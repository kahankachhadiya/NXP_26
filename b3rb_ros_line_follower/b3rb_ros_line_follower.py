# Copyright 2024-2026 NXP
# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import time
from enum import Enum

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import String
from synapse_msgs.msg import EdgeVectors, ServerCommunication

QOS_PROFILE_DEFAULT = 10
PI = math.pi

# ── Speed / steer bounds ────────────────────────────────────────────────────
SPEED_MAX  =  1.0
SPEED_MIN  =  0.0
TURN_MAX   =  1.0
TURN_MIN   = -1.0

# ── Intersection turn override parameters ───────────────────────────────────
TURN_OVERRIDE_DURATION = 2.0   # seconds to hold turn after directional sign
TURN_OVERRIDE_LEFT     =  0.6  # positive = left
TURN_OVERRIDE_RIGHT    = -0.6  # negative = right

# ── ZONE_APPROACH auto-expiry ────────────────────────────────────────────────
ZONE_APPROACH_TIMEOUT  = 3.0   # seconds without a new sign → revert to TRACKING

# ── PID gains ───────────────────────────────────────────────────────────────
KP = 0.6
KI = 0.0
KD = 0.15


# ── FSM States ───────────────────────────────────────────────────────────────
class State(Enum):
    TRACKING          = 0   # Normal lane-following at full speed
    OBSTACLE_DETECTED = 1   # LIDAR triggered — motors stopped
    ZONE_APPROACH     = 2   # Near a location sign — slowed, watching for QR
    SERVER_HANDSHAKE  = 3   # Stopped at QR; waiting for server response
    MISSION_COMPLETE  = 4   # All deliveries done; permanently stopped


# ── Target mapping ───────────────────────────────────────────────────────────
# Maps mission waypoint names to the sign-board label that marks the location.
TARGET_SIGN_MAP = {
    "PATIENT_1":  "A",
    "PATIENT_2":  "B",
    "PATIENT_3":  "C",
    "HOSPITAL_1": "X",
    "HOSPITAL_2": "Y",
    "HOSPITAL_3": "Z",
}

# Ordered patient sequence used to detect when all patients are delivered.
PATIENT_SEQUENCE  = ["PATIENT_1",  "PATIENT_2",  "PATIENT_3"]
HOSPITAL_SEQUENCE = ["HOSPITAL_1", "HOSPITAL_2", "HOSPITAL_3"]

# Directional signs that trigger turn overrides at intersections.
DIRECTIONAL_SIGNS = {"Left", "Right", "Straight"}



class LineFollower(Node):
    """
    FSM-based controller for the B3RB buggy in the NXP Cup Medical Response Challenge.

    FSM States
    ----------
    TRACKING          : PID lane-following at full speed.  Entry state.
    OBSTACLE_DETECTED : LIDAR sees something within 0.8 m → motors halted.
    ZONE_APPROACH     : Relevant sign board seen → half speed, looking for QR.
    SERVER_HANDSHAKE  : QR matched → stopped, waiting for server assignment.
    MISSION_COMPLETE  : All deliveries done → permanently stopped.

    Mission flow
    ------------
    1. Follow lane until the sign matching active_target (or a directional sign) appears.
    2. Slow to half speed (ZONE_APPROACH).  Execute any left/right turn override.
    3. When the QR code containing active_target is seen → stop, notify server.
    4. Server replies with the next destination → update active_target, resume TRACKING.
    5. After 3 patients delivered (server sends "COMPLETED" or equivalent) → park.
    """

    def __init__(self):
        super().__init__('line_follower')

        # ── Subscriptions ──────────────────────────────────────────────────
        self.subscription_vectors = self.create_subscription(
            EdgeVectors, '/edge_vectors',
            self.edge_vectors_callback, QOS_PROFILE_DEFAULT)

        self.subscription_lidar = self.create_subscription(
            LaserScan, '/scan',
            self.lidar_callback, QOS_PROFILE_DEFAULT)

        self.subscription_server = self.create_subscription(
            ServerCommunication, '/ServerCommunication',
            self.server_communication_callback, QOS_PROFILE_DEFAULT)

        self.subscription_qr = self.create_subscription(
            String, '/qr_detection',
            self.qr_detection_callback, QOS_PROFILE_DEFAULT)

        self.subscription_signs = self.create_subscription(
            String, '/sign_board_detection',
            self.sign_board_callback, QOS_PROFILE_DEFAULT)

        # ── Publishers ─────────────────────────────────────────────────────
        self.publisher_joy = self.create_publisher(
            Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT)

        self.publisher_server = self.create_publisher(
            ServerCommunication, '/ServerCommunication', QOS_PROFILE_DEFAULT)


        # ── FSM state ──────────────────────────────────────────────────────
        self.current_state = State.TRACKING
        self.get_logger().info(
            f"[FSM] Initial state: {self.current_state.name}"
        )

        # ── Mission tracking ───────────────────────────────────────────────
        # active_target is the sign/QR label the buggy is currently seeking.
        self.active_target = "PATIENT_1"
        self.patients_delivered = 0          # increments each time a patient QR is confirmed
        self.get_logger().info(
            f"[MISSION] active_target='{self.active_target}', "
            f"sign='{TARGET_SIGN_MAP[self.active_target]}'. "
            f"Full target map: {TARGET_SIGN_MAP}"
        )

        # ── Control outputs ────────────────────────────────────────────────
        self.target_speed = 0.15
        self.target_turn  = 0.0

        # ── PID state ─────────────────────────────────────────────────────
        self.kp         = KP
        self.ki         = KI
        self.kd         = KD
        self.prev_error = 0.0
        self.integral   = 0.0

        # ── Turn override (intersection navigation) ────────────────────────
        # When a directional sign is seen, we temporarily lock the steer output.
        self.turn_override_active    = False
        self.turn_override_value     = 0.0
        self.turn_override_end_time  = 0.0   # wall-clock epoch seconds

        # ── ZONE_APPROACH timeout ──────────────────────────────────────────
        self.last_sign_time = None            # time.time() of last sign detection

        # ── Throttle counters for high-frequency callbacks ─────────────────
        self._pid_log_counter   = 0
        self._lidar_log_counter = 0

        # ── 10 Hz control loop ─────────────────────────────────────────────
        self.control_timer = self.create_timer(0.1, self.publish_drive_commands)

        self.get_logger().info(
            "[INIT] LineFollower FSM node initialised. "
            "Starting in TRACKING state. 10 Hz control loop active."
        )

    # ================================================================== #
    #  10 Hz drive-command publisher                                      #
    # ================================================================== #

    def publish_drive_commands(self):
        """
        Publishes the current speed and steer values to /cerebri/in/joy.

        Priority chain (highest to lowest):
          1. MISSION_COMPLETE or SERVER_HANDSHAKE → always 0,0.
          2. OBSTACLE_DETECTED → always 0,0.
          3. Turn override active (intersection manoeuvre) → use override steer.
          4. Normal PID output.

        msg.axes layout: [0.0, speed, 0.0, turn]
          speed : positive = forward  (range -1..1)
          turn  : positive = left     (range -1..1)
        """
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]

        if self.current_state in (State.MISSION_COMPLETE, State.SERVER_HANDSHAKE,
                                  State.OBSTACLE_DETECTED):
            speed = 0.0
            turn  = 0.0
        else:
            # Check if a turn override is still active.
            if self.turn_override_active:
                if time.time() < self.turn_override_end_time:
                    turn  = self.turn_override_value
                    speed = self.target_speed
                else:
                    # Override expired — cancel it and resume PID steering.
                    self.turn_override_active = False
                    self.get_logger().info(
                        "[DRIVE] Turn override EXPIRED — resuming PID steering."
                    )
                    turn  = max(TURN_MIN,  min(TURN_MAX,  self.target_turn))
                    speed = max(SPEED_MIN, min(SPEED_MAX, self.target_speed))
            else:
                turn  = max(TURN_MIN,  min(TURN_MAX,  self.target_turn))
                speed = max(SPEED_MIN, min(SPEED_MAX, self.target_speed))

        msg.axes = [0.0, speed, 0.0, turn]
        self.publisher_joy.publish(msg)

    # ================================================================== #
    #  Helper: FSM transition                                             #
    # ================================================================== #

    def _transition(self, new_state, reason=""):
        """Log and execute a state transition."""
        old = self.current_state.name
        self.current_state = new_state
        self.get_logger().info(
            f"[FSM] *** STATE CHANGE: {old} → {new_state.name} "
            + (f"| reason: {reason}" if reason else "")
            + " ***"
        )

    # ================================================================== #
    #  Helper: send message to server                                     #
    # ================================================================== #

    def send_server_message(self, text):
        """Publish a message to the Municipality Server (src=1, dest=2)."""
        srv_msg = ServerCommunication()
        srv_msg.src  = 1
        srv_msg.dest = 2
        srv_msg.ack  = 0
        srv_msg.msg  = text
        self.publisher_server.publish(srv_msg)
        self.get_logger().info(
            f"[SERVER] >>> Sent to server: '{text}'"
        )

    # ================================================================== #
    #  Helper: advance mission target                                     #
    # ================================================================== #

    def _set_active_target(self, new_target):
        """Update active_target, log thoroughly, and reset PID integral."""
        old = self.active_target
        self.active_target = new_target
        sign = TARGET_SIGN_MAP.get(new_target, "UNKNOWN")
        self.integral  = 0.0
        self.prev_error = 0.0
        self.get_logger().info(
            f"[MISSION] Target updated: '{old}' → '{new_target}' "
            f"(look for sign '{sign}' / QR containing '{new_target}')."
        )

    # ================================================================== #
    #  Callback: Edge Vectors → PID steering                             #
    # ================================================================== #

    def edge_vectors_callback(self, message):
        """
        PID lane-following controller.

        Active only in TRACKING and ZONE_APPROACH states.  In ZONE_APPROACH,
        maximum speed is capped at 50% of SPEED_MAX.

        Also handles ZONE_APPROACH timeout: if no sign has been seen for
        ZONE_APPROACH_TIMEOUT seconds, revert automatically to TRACKING.
        """
        if self.current_state not in (State.TRACKING, State.ZONE_APPROACH):
            return

        # ── ZONE_APPROACH timeout check ────────────────────────────────
        if (self.current_state == State.ZONE_APPROACH
                and self.last_sign_time is not None
                and time.time() - self.last_sign_time > ZONE_APPROACH_TIMEOUT):
            self._transition(
                State.TRACKING,
                f"No sign seen for {ZONE_APPROACH_TIMEOUT}s — reverting to full speed."
            )

        half_width = message.image_width / 2.0

        # ── Compute centroid_x from available vectors ──────────────────
        if message.vector_count == 2:
            x1 = (message.vector_1[0].x + message.vector_1[1].x) / 2.0
            x2 = (message.vector_2[0].x + message.vector_2[1].x) / 2.0
            centroid_x = (x1 + x2) / 2.0
        elif message.vector_count == 1:
            centroid_x = (message.vector_1[0].x + message.vector_1[1].x) / 2.0
        else:
            self.get_logger().warn(
                "[PID] No edge vectors received — cannot compute steering. "
                "Holding previous output."
            )
            return

        # ── Normalised lateral error ───────────────────────────────────
        # Positive error → centroid is right of centre → steer right (negative turn).
        error = (centroid_x - half_width) / half_width

        self.integral   += error
        derivative       = error - self.prev_error
        u                = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error  = error

        # Negate: rightward error → negative (right) turn value.
        self.target_turn = max(TURN_MIN, min(TURN_MAX, -u))

        # ── Speed: inversely proportional to steering effort ───────────
        speed_cap = SPEED_MAX * 0.5 if self.current_state == State.ZONE_APPROACH else SPEED_MAX
        self.target_speed = max(
            SPEED_MIN,
            min(speed_cap, speed_cap * (1.0 - abs(self.target_turn)))
        )

        # Throttle PID log to once every 30 callbacks.
        self._pid_log_counter += 1
        if self._pid_log_counter >= 30:
            self._pid_log_counter = 0
            self.get_logger().info(
                f"[PID] error={error:.3f}, turn={self.target_turn:.3f}, "
                f"speed={self.target_speed:.3f} (state={self.current_state.name})."
            )

    # ================================================================== #
    #  Callback: LIDAR → obstacle detection                              #
    # ================================================================== #

    def lidar_callback(self, message):
        """
        Monitor the forward sector for obstacles.

        Extracts the front ~22% of the scan arc (indices N*7/18 to N*11/18),
        filters non-finite values, and transitions to OBSTACLE_DETECTED if the
        closest reading is within 0.8 m.  Recovers to TRACKING once clear.

        Does not interrupt SERVER_HANDSHAKE or MISSION_COMPLETE states.
        """
        if self.current_state in (State.SERVER_HANDSHAKE, State.MISSION_COMPLETE):
            return

        N = len(message.ranges)
        front = message.ranges[int(N * 7 / 18) : int(N * 11 / 18)]
        finite_ranges = [r for r in front if math.isfinite(r)]
        min_dist = min(finite_ranges) if finite_ranges else math.inf

        if min_dist < 0.8:
            if self.current_state != State.OBSTACLE_DETECTED:
                self._transition(
                    State.OBSTACLE_DETECTED,
                    f"Obstacle at {min_dist:.2f} m (threshold 0.8 m)."
                )
                self.target_speed = 0.0
                self.target_turn  = 0.0
            # No else — don't log "still blocked" every scan cycle.
        elif self.current_state == State.OBSTACLE_DETECTED:
            self._transition(
                State.TRACKING,
                f"Path clear — nearest obstacle now {min_dist:.2f} m."
            )

    # ================================================================== #
    #  Callback: Sign board → intersection navigation                    #
    # ================================================================== #

    def sign_board_callback(self, message):
        """
        React to a detected traffic sign board.

        Logic
        -----
        - If the sign matches the mapped label for active_target OR is a
          directional sign (Left/Right/Straight), transition to ZONE_APPROACH.
        - Directional signs additionally arm a turn override:
            Left     → +0.6 steer for TURN_OVERRIDE_DURATION seconds.
            Right    → -0.6 steer for TURN_OVERRIDE_DURATION seconds.
            Straight →  no turn override (PID continues normally).
        - Any sign refreshes the ZONE_APPROACH timeout clock.
        - Signs that are irrelevant to the current target are logged but ignored.
        """
        if self.current_state == State.MISSION_COMPLETE:
            return

        sign = message.data
        if not sign:
            return

        expected_sign = TARGET_SIGN_MAP.get(self.active_target, "")

        is_target_sign      = (sign == expected_sign)
        is_directional_sign = (sign in DIRECTIONAL_SIGNS)

        if is_target_sign or is_directional_sign:
            if self.current_state != State.ZONE_APPROACH:
                self._transition(
                    State.ZONE_APPROACH,
                    f"Sign '{sign}' triggered zone approach "
                    f"({'target match' if is_target_sign else 'directional'})."
                )

            # Refresh the ZONE_APPROACH timeout clock.
            self.last_sign_time = time.time()

            # ── Directional turn overrides ─────────────────────────────
            if sign == "Left":
                self.turn_override_active   = True
                self.turn_override_value    = TURN_OVERRIDE_LEFT
                self.turn_override_end_time = time.time() + TURN_OVERRIDE_DURATION
                self.get_logger().info(
                    f"[SIGN] LEFT turn override armed: "
                    f"steer={TURN_OVERRIDE_LEFT} for {TURN_OVERRIDE_DURATION}s."
                )
            elif sign == "Right":
                self.turn_override_active   = True
                self.turn_override_value    = TURN_OVERRIDE_RIGHT
                self.turn_override_end_time = time.time() + TURN_OVERRIDE_DURATION
                self.get_logger().info(
                    f"[SIGN] RIGHT turn override armed: "
                    f"steer={TURN_OVERRIDE_RIGHT} for {TURN_OVERRIDE_DURATION}s."
                )

    # ================================================================== #
    #  Callback: QR code → server handshake trigger                      #
    # ================================================================== #

    def qr_detection_callback(self, message):
        """
        Trigger a server handshake when a QR code matching active_target is scanned.

        Uses a partial-match check so that payloads like "{LOC: PATIENT_1, ...}"
        still match when active_target == "PATIENT_1".

        On match:
          1. Stop the buggy.
          2. Transition to SERVER_HANDSHAKE.
          3. Send the full QR payload to the server (server uses it to assign
             the next destination).

        Non-matching payloads are logged and ignored.
        """
        if self.current_state in (State.SERVER_HANDSHAKE, State.MISSION_COMPLETE):
            self.get_logger().info(
                f"[QR] Ignoring QR scan in state {self.current_state.name}."
            )
            return

        payload = message.data
        if self.current_state in (State.SERVER_HANDSHAKE, State.MISSION_COMPLETE):
            return

        # Partial match: active_target substring inside payload.
        if self.active_target in payload:
            self.get_logger().info(
                f"[QR] MATCH: '{self.active_target}' found in '{payload}' — "
                "stopping and initiating SERVER_HANDSHAKE."
            )
            self.target_speed = 0.0
            self.target_turn  = 0.0
            self._transition(
                State.SERVER_HANDSHAKE,
                f"QR matched active_target='{self.active_target}'."
            )
            self.send_server_message(payload)

    # ================================================================== #
    #  Callback: Server response → mission progression                   #
    # ================================================================== #

    def server_communication_callback(self, message):
        """
        Handle the Municipality Server's response to advance the mission.

        Message filtering
        -----------------
        - Ignore messages where dest != 1 (not addressed to the buggy).

        Mission progression logic
        -------------------------
        1. If the payload contains "COMPLETED" (case-insensitive), all deliveries
           are done → navigate to parking, notify server with "PARKED", enter
           MISSION_COMPLETE (permanent stop).
        2. Otherwise scan the payload for any known target key
           (PATIENT_1/2/3, HOSPITAL_1/2/3).  If found:
             a. Update active_target.
             b. If the new target is a PATIENT, increment patients_delivered.
             c. Transition back to TRACKING to resume driving.
        3. If neither rule matches, log a warning and remain in current state.
        """
        # Only process messages addressed to us (buggy = dest 1).
        if message.dest != 1:
            return

        payload = message.msg.strip()

        # ── Rule 1: Mission complete signal ───────────────────────────
        if "COMPLETED" in payload.upper():
            self.get_logger().info(
                "[SERVER] *** 'COMPLETED' received — all deliveries done! "
                "Navigating to parking area. ***"
            )
            self._navigate_to_parking()
            return

        # ── Rule 2: New target assignment ─────────────────────────────
        for target_key in list(TARGET_SIGN_MAP.keys()):
            if target_key in payload:
                self.get_logger().info(
                    f"[SERVER] *** New target assigned: '{target_key}' "
                    f"(found in payload '{payload}'). ***"
                )
                self._set_active_target(target_key)

                # Track patient deliveries.
                if target_key in PATIENT_SEQUENCE:
                    self.patients_delivered += 1
                    self.get_logger().info(
                        f"[MISSION] Patient delivery #{self.patients_delivered} confirmed. "
                        f"Patients delivered so far: {self.patients_delivered}/3."
                    )

                self._transition(
                    State.TRACKING,
                    f"New target '{target_key}' received from server — resuming."
                )
                return

        # ── Rule 3: Unrecognised payload ──────────────────────────────
        self.get_logger().warn(
            f"[SERVER] Could not parse a known target or 'COMPLETED' "
            f"from payload: '{payload}'. Remaining in {self.current_state.name}."
        )

    # ================================================================== #
    #  Parking helper                                                     #
    # ================================================================== #

    def _navigate_to_parking(self):
        """
        Execute the parking sequence and transition to MISSION_COMPLETE.

        Strategy: drive straight at reduced speed for 3 seconds (configurable),
        then stop.  A real implementation can substitute a dedicated parking
        path planner here.
        """
        self.get_logger().info(
            "[PARK] *** Beginning parking manoeuvre — driving straight for 3 seconds. ***"
        )
        # Drive straight at 30% speed for ~3 s.  The control timer will send
        # these values until we flip to MISSION_COMPLETE.
        self.target_speed = 0.3
        self.target_turn  = 0.0

        # Schedule the MISSION_COMPLETE transition via a one-shot timer.
        self._parking_timer = self.create_timer(3.0, self._finish_parking)

    def _finish_parking(self):
        """Called 3 s after parking begins to halt and finalize the mission."""
        self._parking_timer.cancel()
        self.target_speed = 0.0
        self.target_turn  = 0.0
        self._transition(
            State.MISSION_COMPLETE,
            "Parking complete — buggy stopped permanently."
        )
        self.send_server_message("PARKED")
        self.get_logger().info(
            "[MISSION] *** MISSION COMPLETE — all patients delivered and buggy parked. "
            "Sent 'PARKED' to server. ***"
        )


# =========================================================================== #
#  Entry point                                                                 #
# =========================================================================== #

def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            "[SHUTDOWN] KeyboardInterrupt received — shutting down LineFollower."
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
