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

# ── Directional bias applied to PID error when a turn sign is pending ───────
TURN_BIAS_LEFT     =  0.48
TURN_BIAS_RIGHT    = -0.48

# ── Boundary constraint ──────────────────────────────────────────────────────
BOUNDARY_CORRECTION_TURN = 0.5
BOUNDARY_SPEED_CAP       = 0.24

# ── Speed constants ──────────────────────────────────────────────────────────
NO_VECTOR_SPEED     = 0.18

# ── Obstacle avoidance ───────────────────────────────────────────────────────
AVOIDANCE_SPEED     = 0.18
AVOIDANCE_TURN      = 0.6
AVOIDANCE_THRESHOLD = 0.8

# ── Target approach (LIDAR stop thresholds in TRACKING) ──────────────────────
APPROACH_CREEP_THRESHOLD = 0.6
APPROACH_STOP_THRESHOLD  = 0.35

# ── Directional sign gap timer ────────────────────────────────────────────────
DIR_SIGN_TIMEOUT = 5.0   # seconds without directional sign → gap open

# ── PID gains ───────────────────────────────────────────────────────────────
KP = 0.25
KI = 0.0
KD = 0.08

# ── Boundary proximity steering ──────────────────────────────────────────────
BOUNDARY_ZONE = 0.35

# ── Base straight speed ────────────────────────────────────────────────────
STRAIGHT_SPEED = 0.18

# ── Speed control ────────────────────────────────────────────────────────────
SPEED_REDUCTION_THRESHOLD = 0.3


# ── FSM States ───────────────────────────────────────────────────────────────
class State(Enum):
    TRACKING           = 0   # Normal lane-following, PID with boundary proximity
    OBSTACLE_AVOIDANCE = 1   # Steering around an obstacle using LIDAR + edge vectors
    SERVER_HANDSHAKE   = 2   # Stopped at QR destination, waiting for server reply
    MISSION_COMPLETE   = 3   # All done, stopped permanently


# ── Target mapping ───────────────────────────────────────────────────────────
TARGET_SIGN_MAP = {
    "PATIENT_1":  "A",
    "PATIENT_2":  "B",
    "PATIENT_3":  "C",
    "HOSPITAL_1": "X",
    "HOSPITAL_2": "Y",
    "HOSPITAL_3": "Z",
}

PATIENT_SEQUENCE  = ["PATIENT_1",  "PATIENT_2",  "PATIENT_3"]
HOSPITAL_SEQUENCE = ["HOSPITAL_1", "HOSPITAL_2", "HOSPITAL_3"]

DIRECTIONAL_SIGNS = {"Left", "Right", "Straight"}


class LineFollower(Node):
    """
    FSM-based controller for the B3RB buggy in the NXP Cup Medical Response Challenge.

    FSM States
    ----------
    TRACKING          : PID lane-following, directional bias, handles all driving.
    OBSTACLE_AVOIDANCE: LIDAR-triggered, edge-vector-constrained steering around obstacle.
    SERVER_HANDSHAKE  : Stopped at QR; waiting for server assignment.
    MISSION_COMPLETE  : All deliveries done; permanently stopped.

    Mission flow
    ------------
    1. Follow lane until QR code containing active_target appears — stop.
    2. Transition to SERVER_HANDSHAKE, send QR payload to server.
    3. Server replies with the next destination → update active_target, resume TRACKING.
    4. After all deliveries (server sends "COMPLETED") → park.
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
        self.active_target      = "PATIENT_1"
        self.patients_delivered = 0
        self.get_logger().info(
            f"[MISSION] active_target='{self.active_target}', "
            f"sign='{TARGET_SIGN_MAP[self.active_target]}'. "
            f"Full target map: {TARGET_SIGN_MAP}"
        )

        # ── Control outputs ────────────────────────────────────────────────
        self.target_speed = STRAIGHT_SPEED
        self.target_turn  = 0.0

        # ── PID state ─────────────────────────────────────────────────────
        self.kp         = KP
        self.ki         = KI
        self.kd         = KD
        self.prev_error = 0.0
        self.integral   = 0.0

        # ── Pending direction from last directional sign ───────────────────
        # 'Left', 'Right', 'Straight', or None
        self.pending_direction = None

        # ── Obstacle avoidance steer direction ────────────────────────────
        self.avoidance_turn_value = 0.0

        # ── Directional sign gap tracking ─────────────────────────────────
        self.dir_sign_visible    = False
        self.dir_sign_gap_open   = False
        self._last_dir_sign_time = None

        # ── Throttle counters ──────────────────────────────────────────────
        self._pid_log_counter = 0

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
        """Publish current speed and steer values to /cerebri/in/joy."""
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]

        if self.current_state in (State.SERVER_HANDSHAKE, State.MISSION_COMPLETE):
            speed = 0.0
            turn  = 0.0
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
        if new_state == State.TRACKING:
            self.integral   = 0.0
            self.prev_error = 0.0

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
        """Update active_target and reset navigation state."""
        old  = self.active_target
        sign = TARGET_SIGN_MAP.get(new_target, "UNKNOWN")
        self.active_target       = new_target
        self.integral            = 0.0
        self.prev_error          = 0.0
        self.pending_direction   = None
        self.dir_sign_visible    = False
        self.dir_sign_gap_open   = False
        self._last_dir_sign_time = None
        self.get_logger().info(
            f"[MISSION] Target updated: '{old}' → '{new_target}' "
            f"(look for sign '{sign}' / QR containing '{new_target}')."
        )

    # ================================================================== #
    #  Callback: Sign board → directional bias or target logging         #
    # ================================================================== #

    def sign_board_callback(self, message):
        """
        React to a detected traffic sign board.

        Directional signs (Left/Right/Straight):
          - Ignored in MISSION_COMPLETE or OBSTACLE_AVOIDANCE.
          - Gap logic: if sign reappears after a gap → clear pending_direction,
            lock the new direction.
          - Sets pending_direction, never changes FSM state.

        Location signs (A/B/C/X/Y/Z) matching active_target:
          - Ignored in MISSION_COMPLETE, OBSTACLE_AVOIDANCE, or SERVER_HANDSHAKE.
          - Just logged. QR detector handles the actual stop + handshake.

        Any other sign: ignored silently.
        """
        sign = message.data
        if not sign:
            return

        expected_sign       = TARGET_SIGN_MAP.get(self.active_target, "")
        is_target_sign      = (sign == expected_sign)
        is_directional_sign = (sign in DIRECTIONAL_SIGNS)

        if not (is_target_sign or is_directional_sign):
            return

        # ── Directional sign (green board) ────────────────────────────────
        if is_directional_sign:
            if self.current_state in (State.MISSION_COMPLETE, State.OBSTACLE_AVOIDANCE):
                return

            now = time.time()

            # Gap logic: if visible→gone→visible, it's a fresh board
            if not self.dir_sign_visible and self.dir_sign_gap_open:
                self.pending_direction = None
                self.dir_sign_gap_open = False
                self.get_logger().info(
                    f"[SIGN] New green board after gap — direction reset, locking '{sign}'."
                )

            self.dir_sign_visible    = True
            self._last_dir_sign_time = now

            if self.pending_direction is None:
                self.pending_direction = sign
                self.get_logger().info(
                    f"[SIGN] Pending direction set to '{sign}' — "
                    "PID will bias toward this side through the junction."
                )
            return

        # ── Location sign matching active_target ──────────────────────────
        if is_target_sign:
            if self.current_state in (State.MISSION_COMPLETE, State.OBSTACLE_AVOIDANCE,
                                      State.SERVER_HANDSHAKE):
                return
            self.get_logger().info(
                f"[SIGN] Target location sign '{sign}' seen for '{self.active_target}' — "
                "QR detector will handle the stop."
            )

    # ================================================================== #
    #  Callback: QR code → server handshake trigger                      #
    # ================================================================== #

    def qr_detection_callback(self, message):
        """
        Trigger a server handshake when a QR code matching active_target is scanned.

        On match:
          1. Stop the buggy.
          2. Transition to SERVER_HANDSHAKE.
          3. Send the full QR payload to the server.
        """
        if self.current_state in (State.SERVER_HANDSHAKE, State.MISSION_COMPLETE):
            return

        payload = message.data

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
    #  Callback: Edge Vectors → PID steering                             #
    # ================================================================== #

    def edge_vectors_callback(self, message):
        """
        PID lane-following controller active in TRACKING and OBSTACLE_AVOIDANCE.

        OBSTACLE_AVOIDANCE handling (checked first):
          2 vectors straddling centre → transition back to TRACKING, fall through to PID.
          2 vectors same side         → steer away, AVOIDANCE_SPEED, return.
          1 vector                    → steer away from boundary, AVOIDANCE_SPEED, return.
          0 vectors                   → creep in avoidance_turn_value dir, AVOIDANCE_SPEED, return.

        TRACKING:
          Directional sign gap timer check.
          Bias: Left→TURN_BIAS_LEFT, Right→TURN_BIAS_RIGHT, else 0.
          Bias dropped if vectors straddling (both sides → on new road).
          CASE 0: creep at NO_VECTOR_SPEED with bias clamped.
          CASE 1: boundary correction ± bias (boundary wins on conflict).
          CASE 2: PID on boundary-proximity error + bias; speed scales with turn.
        """
        if self.current_state not in (State.TRACKING, State.OBSTACLE_AVOIDANCE):
            return

        half_width = float(message.image_width) / 2.0

        # ── OBSTACLE_AVOIDANCE: edge-vector-constrained recovery ──────────
        if self.current_state == State.OBSTACLE_AVOIDANCE:
            if message.vector_count == 2:
                ax1 = (message.vector_1[0].x + message.vector_1[1].x) / 2.0
                ax2 = (message.vector_2[0].x + message.vector_2[1].x) / 2.0
                straddling = (ax1 < half_width) != (ax2 < half_width)

                if straddling:
                    self._transition(
                        State.TRACKING,
                        "Vectors on both sides — back inside track after avoidance."
                    )
                    # Fall through to TRACKING PID below.
                else:
                    # Both on same side — keep steering away.
                    vec_x = (ax1 + ax2) / 2.0
                    if vec_x < half_width:
                        self.target_turn = AVOIDANCE_TURN   # left boundary → steer right? No:
                        # left boundary (vec_x < half_width) → steer right = negative turn
                        self.target_turn = -BOUNDARY_CORRECTION_TURN
                    else:
                        self.target_turn =  BOUNDARY_CORRECTION_TURN
                    self.target_speed = AVOIDANCE_SPEED
                    return

            elif message.vector_count == 1:
                vec_x = (message.vector_1[0].x + message.vector_1[1].x) / 2.0
                if vec_x < half_width:
                    self.target_turn = -BOUNDARY_CORRECTION_TURN
                else:
                    self.target_turn =  BOUNDARY_CORRECTION_TURN
                self.target_speed = AVOIDANCE_SPEED
                return

            else:
                # No vectors — creep in avoidance direction.
                self.target_speed = AVOIDANCE_SPEED
                self.target_turn  = self.avoidance_turn_value
                return

        # ── TRACKING: directional sign gap timer ──────────────────────────
        if (self.dir_sign_visible
                and self._last_dir_sign_time is not None
                and time.time() - self._last_dir_sign_time > DIR_SIGN_TIMEOUT):
            self.dir_sign_visible  = False
            self.dir_sign_gap_open = True
            self.get_logger().info(
                "[SIGN] Green board gone — gap open, next board detection "
                "will reset direction and lock the new sign."
            )

        # ── Directional bias for this frame ───────────────────────────────
        if self.pending_direction == 'Left':
            bias = TURN_BIAS_LEFT
        elif self.pending_direction == 'Right':
            bias = TURN_BIAS_RIGHT
        else:
            bias = 0.0

        speed_cap = SPEED_MAX

        # ════════════════════════════════════════════════════════════════
        #  CASE 0: No edge vectors
        # ════════════════════════════════════════════════════════════════
        if message.vector_count == 0:
            self.target_speed = NO_VECTOR_SPEED
            self.target_turn  = max(TURN_MIN, min(TURN_MAX, bias))
            return

        # ════════════════════════════════════════════════════════════════
        #  CASE 1: Single vector — near one boundary, steer away
        # ════════════════════════════════════════════════════════════════
        if message.vector_count == 1:
            vec_x = (message.vector_1[0].x + message.vector_1[1].x) / 2.0

            if vec_x < half_width:
                boundary_turn = -BOUNDARY_CORRECTION_TURN   # left boundary → steer right
            else:
                boundary_turn =  BOUNDARY_CORRECTION_TURN   # right boundary → steer left

            # Boundary wins if it conflicts with bias direction.
            if abs(bias) > 0 and (bias * boundary_turn < 0):
                self.target_turn = max(TURN_MIN, min(TURN_MAX, boundary_turn))
            else:
                self.target_turn = max(TURN_MIN, min(TURN_MAX, boundary_turn + bias))

            self.target_speed = BOUNDARY_SPEED_CAP
            return

        # ════════════════════════════════════════════════════════════════
        #  CASE 2: Both vectors visible
        # ════════════════════════════════════════════════════════════════
        x1 = (message.vector_1[0].x + message.vector_1[1].x) / 2.0
        x2 = (message.vector_2[0].x + message.vector_2[1].x) / 2.0
        centroid_x = (x1 + x2) / 2.0

        # Drop bias once vectors straddle centre — buggy is on the new road.
        vectors_straddling = (x1 < half_width) != (x2 < half_width)
        if vectors_straddling:
            bias = 0.0

        # Boundary-proximity error: only non-zero when near an edge.
        norm_pos   = (centroid_x - half_width) / half_width   # -1..+1
        safe_limit = 1.0 - BOUNDARY_ZONE

        if norm_pos > safe_limit:
            error = norm_pos - safe_limit
        elif norm_pos < -safe_limit:
            error = norm_pos + safe_limit
        else:
            error = 0.0

        error += bias

        self.integral  += error
        derivative      = error - self.prev_error
        u               = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        self.target_turn = max(TURN_MIN, min(TURN_MAX, -u))

        excess = max(0.0, abs(self.target_turn) - SPEED_REDUCTION_THRESHOLD)
        scale  = 1.0 - excess / (1.0 - SPEED_REDUCTION_THRESHOLD + 1e-6)
        self.target_speed = max(SPEED_MIN, min(speed_cap, speed_cap * scale))

        self._pid_log_counter += 1
        if self._pid_log_counter >= 30:
            self._pid_log_counter = 0
            self.get_logger().info(
                f"[PID] error={error:.3f} (bias={bias:+.2f}), "
                f"turn={self.target_turn:.3f}, speed={self.target_speed:.3f} "
                f"(state={self.current_state.name}, dir={self.pending_direction})."
            )

    # ================================================================== #
    #  Callback: LIDAR → obstacle detection / approach stop              #
    # ================================================================== #

    def lidar_callback(self, message):
        """
        LIDAR handler — only active in TRACKING.

        Ignored in SERVER_HANDSHAKE, MISSION_COMPLETE, OBSTACLE_AVOIDANCE.

        TRACKING:
          <= APPROACH_STOP_THRESHOLD  : stop for QR scan.
          <= APPROACH_CREEP_THRESHOLD : slow to 0.1 m/s creep.
          <  AVOIDANCE_THRESHOLD      : transition to OBSTACLE_AVOIDANCE.
        """
        if self.current_state in (State.SERVER_HANDSHAKE, State.MISSION_COMPLETE,
                                   State.OBSTACLE_AVOIDANCE):
            return

        N            = len(message.ranges)
        front_start  = int(N * 7 / 18)
        front_end    = int(N * 11 / 18)
        front        = message.ranges[front_start:front_end]
        finite_front = [r for r in front if math.isfinite(r)]
        min_dist     = min(finite_front) if finite_front else math.inf

        # ── TRACKING ─────────────────────────────────────────────────────
        if self.current_state == State.TRACKING:
            if min_dist <= APPROACH_STOP_THRESHOLD:
                self.target_speed = 0.0
                self.target_turn  = 0.0
                self.get_logger().info(
                    f"[APPROACH] Stopped at {min_dist:.2f} m — waiting for QR scan."
                )
                return

            if min_dist <= APPROACH_CREEP_THRESHOLD:
                self.target_speed = 0.1
                return

            if min_dist < AVOIDANCE_THRESHOLD:
                left_ranges  = [r for r in message.ranges[:front_start] if math.isfinite(r)]
                right_ranges = [r for r in message.ranges[front_end:]   if math.isfinite(r)]
                left_clear   = sum(left_ranges)  / len(left_ranges)  if left_ranges  else 0.0
                right_clear  = sum(right_ranges) / len(right_ranges) if right_ranges else 0.0

                if left_clear >= right_clear:
                    self.avoidance_turn_value =  AVOIDANCE_TURN
                    side = "LEFT"
                else:
                    self.avoidance_turn_value = -AVOIDANCE_TURN
                    side = "RIGHT"

                self._transition(
                    State.OBSTACLE_AVOIDANCE,
                    f"Obstacle at {min_dist:.2f} m — steering {side} to avoid."
                )
                self.target_speed = AVOIDANCE_SPEED
                self.target_turn  = self.avoidance_turn_value

    # ================================================================== #
    #  Callback: Server response → mission progression                   #
    # ================================================================== #

    def server_communication_callback(self, message):
        """
        Handle the Municipality Server's response to advance the mission.

        Ignore messages where dest != 1.
        "COMPLETED" in payload → navigate to parking.
        Known target key in payload → set active target, resume TRACKING.
        """
        if message.dest != 1:
            return

        payload = message.msg.strip()

        if "COMPLETED" in payload.upper():
            self.get_logger().info(
                "[SERVER] *** 'COMPLETED' received — all deliveries done! "
                "Navigating to parking area. ***"
            )
            self._navigate_to_parking()
            return

        for target_key in list(TARGET_SIGN_MAP.keys()):
            if target_key in payload:
                self.get_logger().info(
                    f"[SERVER] *** New target assigned: '{target_key}' "
                    f"(found in payload '{payload}'). ***"
                )
                self._set_active_target(target_key)

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

        self.get_logger().warn(
            f"[SERVER] Could not parse a known target or 'COMPLETED' "
            f"from payload: '{payload}'. Remaining in {self.current_state.name}."
        )

    # ================================================================== #
    #  Parking helpers                                                    #
    # ================================================================== #

    def _navigate_to_parking(self):
        """Drive straight at reduced speed for 3 seconds, then stop."""
        self.get_logger().info(
            "[PARK] *** Beginning parking manoeuvre — driving straight for 3 seconds. ***"
        )
        self.target_speed = 0.3
        self.target_turn  = 0.0
        self._parking_timer = self.create_timer(3.0, self._finish_parking)

    def _finish_parking(self):
        """Called 3 s after parking begins — halt and finalise the mission."""
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
