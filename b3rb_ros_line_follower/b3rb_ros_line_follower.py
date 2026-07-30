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

# ── Speed / steer bounds ─────────────────────────────────────────────────────
SPEED_MAX = 1.0
SPEED_MIN = 0.0
TURN_MAX  = 1.0
TURN_MIN  = -1.0

# ── Speed constants ───────────────────────────────────────────────────────────
NORMAL_SPEED    = 0.32   # cruising speed on straights
TURNING_SPEED   = 0.20   # speed while executing a directional turn
NO_VECTOR_SPEED = 0.20   # speed when no edge vectors visible
AVOIDANCE_SPEED = 0.25   # speed during obstacle avoidance
QR_CREEP_SPEED  = 0.15   # speed while QR is visible

# ── Edge safety (ABSOLUTE TOP PRIORITY) ──────────────────────────────────────
# If any edge vector midpoint is within this fraction of the image width
# from its wall side, perform a hard linear recovery steer immediately.
# Nothing overrides this rule.
EDGE_SAFE_MARGIN  = 0.25   # min fraction of image width from each wall
EDGE_RECOVERY_TURN = 0.80  # steer magnitude during recovery

# ── Obstacle avoidance ────────────────────────────────────────────────────────
AVOIDANCE_TURN      = 0.60
AVOIDANCE_THRESHOLD = 0.80  # metres

# ── Target approach ───────────────────────────────────────────────────────────
QR_CREEP_SPEED = 0.15   # m/s while QR matched and visible

# ── Directional sign: frames of centred driving before direction consumed ─────
TURN_DONE_FRAMES = 8

# ── PID gains ─────────────────────────────────────────────────────────────────
KP = 0.55
KI = 0.00
KD = 0.18

# ── Lane tracking targets ─────────────────────────────────────────────────────
# When following a direction, the PID targets this normalised position
# (-1=full left, 0=centre, +1=full right) inside the lane.
# "Left" means follow the left edge → target is left of centre.
# "Right" means follow the right edge → target is right of centre.
FOLLOW_LEFT_TARGET   = -0.35   # norm_pos target when direction is Left
FOLLOW_RIGHT_TARGET  =  0.35   # norm_pos target when direction is Right
FOLLOW_CENTRE_TARGET =  0.0    # norm_pos target for centre / straight

# ── Integral windup clamp ─────────────────────────────────────────────────────
INTEGRAL_CLAMP = 0.20

# ── Speed reduction on turns (linear) ────────────────────────────────────────
# speed = NORMAL_SPEED - SPEED_TURN_SCALE * |turn|
# clamped to [TURNING_SPEED, NORMAL_SPEED]
SPEED_TURN_SCALE = 0.40


# ── FSM States ───────────────────────────────────────────────────────────────
class State(Enum):
    TRACKING         = 0   # All driving — PID, avoidance, sign reading
    SERVER_HANDSHAKE = 1   # Stopped at QR, waiting for server reply
    MISSION_COMPLETE = 2   # All done, stopped permanently


# ── Target mapping ───────────────────────────────────────────────────────────
TARGET_SIGN_MAP = {
    "PATIENT_1":  "A",
    "PATIENT_2":  "B",
    "PATIENT_3":  "C",
    "HOSPITAL_1": "X",
    "HOSPITAL_2": "Y",
    "HOSPITAL_3": "Z",
}

# Reverse map: sign letter → target key (used to parse server reply)
SIGN_TO_TARGET = {v: k for k, v in TARGET_SIGN_MAP.items()}

PATIENT_SEQUENCE  = ["PATIENT_1", "PATIENT_2", "PATIENT_3"]
HOSPITAL_SEQUENCE = ["HOSPITAL_1", "HOSPITAL_2", "HOSPITAL_3"]

DIRECTIONAL_SIGNS = {"Left", "Right", "Straight"}


class LineFollower(Node):
    """
    FSM-based controller for the B3RB buggy in the NXP Cup Medical Response Challenge.

    FSM States (3 total — obstacle avoidance is inline in TRACKING):
    ----------
    TRACKING         : PID lane-following, directional bias, inline obstacle avoidance.
                       Listens to all sign/QR/server events at all times.
    SERVER_HANDSHAKE : Stopped at QR destination; waiting for server reply.
    MISSION_COMPLETE : All deliveries done; permanently stopped.

    Mission flow:
    1. Drive, read green board → set pending_direction bias.
    2. QR code matching active_target scanned → stop, SERVER_HANDSHAKE.
    3. Server sends next target sign letter (e.g. "B") → resume TRACKING.
    4. Server sends "COMPLETED" → park.

    Server payload format (recv):
      { "src": 2, "dest": 1, "uid": 0, "ack": 0, "msg": "<letter_or_COMPLETED>" }
    Server payload format (send):
      { "src": 1, "dest": 2, "uid": 0, "ack": 0, "msg": "<qr_payload>" }
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
        self.get_logger().info(f"[FSM] Initial state: {self.current_state.name}")

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
        self.pending_direction = None   # 'Left', 'Right', 'Straight', or None
        # Counts consecutive frames where bias was dropped (straddling).
        # When it reaches TURN_DONE_FRAMES the direction is consumed — next
        # board detection can overwrite pending_direction.
        self._turn_done_count  = 0

        # ── Inline obstacle avoidance ──────────────────────────────────────
        self.avoiding             = False
        self.avoidance_turn_value = 0.0

        # ── Target sign seen flag ──────────────────────────────────────────
        # True once the target location sign (A/B/C/X/Y/Z) has been detected.
        # Enables the creep-and-stop approach logic in lidar_callback.
        self.target_sign_seen = False

        # ── QR pending payload ─────────────────────────────────────────────
        # Set by qr_detection_callback once a matching QR is seen.
        # Cleared after the server handshake is triggered.
        self._qr_pending_payload = None
        # One-shot ROS timer started when QR disappears; cancelled if QR reappears.
        self._qr_lost_timer = None
        self._qr_lost_since = None  # kept for _set_active_target reset compat

        # ── Throttle counters ──────────────────────────────────────────────
        self._pid_log_counter = 0

        # ── 10 Hz control loop ─────────────────────────────────────────────
        self.control_timer = self.create_timer(0.1, self.publish_drive_commands)

        self.get_logger().info(
            "[INIT] LineFollower node ready. 10 Hz control loop active."
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
        old = self.current_state.name
        self.current_state = new_state
        self.get_logger().info(
            f"[FSM] {old} → {new_state.name}"
            + (f" | {reason}" if reason else "")
        )
        if new_state == State.TRACKING:
            self.integral   = 0.0
            self.prev_error = 0.0

    # ================================================================== #
    #  Helper: send message to server                                     #
    # ================================================================== #

    def send_server_message(self, text):
        """Publish a message to the server. Format: src=1, dest=2."""
        srv_msg = ServerCommunication()
        srv_msg.src  = 1
        srv_msg.dest = 2
        srv_msg.uid  = 0
        srv_msg.ack  = 0
        srv_msg.msg  = text
        self.publisher_server.publish(srv_msg)
        self.get_logger().info(f"[SERVER] >>> Sent: '{text}'")

    # ================================================================== #
    #  Helper: advance mission target                                     #
    # ================================================================== #

    def _set_active_target(self, new_target):
        old  = self.active_target
        sign = TARGET_SIGN_MAP.get(new_target, "?")
        self.active_target       = new_target
        self.integral            = 0.0
        self.prev_error          = 0.0
        self.pending_direction   = None
        self._turn_done_count    = 0
        self.avoiding            = False
        self.target_sign_seen    = False
        self._qr_pending_payload = None
        self._qr_lost_since      = None
        if self._qr_lost_timer is not None:
            self._qr_lost_timer.cancel()
            self._qr_lost_timer = None
        self.get_logger().info(
            f"[MISSION] {old} → {new_target} (look for sign '{sign}' / QR '{new_target}')"
        )

    # ================================================================== #
    #  Callback: Sign board                                               #
    # ================================================================== #

    def sign_board_callback(self, message):
        """
        Parses the board map published by ObjectRecognizer and:

        1. Extracts the direction paired with the active target's character.
           e.g. active_target='PATIENT_1' → expected_sign='A'
                MAP message "MAP:A=Left,B=Left,C=Straight,X=Straight,Y=Right,Z=Right"
                → pending_direction = 'Left'

        2. Marks target_sign_seen=True when the active target's character
           is present on the board (enables creep-and-stop approach via LIDAR).

        Message format from ObjectRecognizer:
          "MAP:<char>=<dir>,<char>=<dir>,..."
          e.g. "MAP:A=Left,B=Left,C=Straight,X=Straight,Y=Right,Z=Right"
        """
        if self.current_state == State.MISSION_COMPLETE:
            return

        raw = message.data
        if not raw:
            return

        # ── Parse MAP message ────────────────────────────────────────────
        if not raw.startswith("MAP:"):
            return

        # Build dict from "A=Left,B=Left,C=Straight,..."
        char_dir_map = {}
        try:
            pairs = raw[4:].split(",")
            for pair in pairs:
                if "=" in pair:
                    ch, dr = pair.split("=", 1)
                    char_dir_map[ch.strip()] = dr.strip()
        except Exception:
            self.get_logger().warn(f"[SIGN] Could not parse board message: '{raw}'")
            return

        if not char_dir_map:
            return

        expected_sign = TARGET_SIGN_MAP.get(self.active_target, "")

        # ── Check if our target character is on the board ────────────────
        if expected_sign not in char_dir_map:
            return

        # ── Mark approach mode ───────────────────────────────────────────
        if self.current_state != State.SERVER_HANDSHAKE and not self.target_sign_seen:
            self.target_sign_seen = True
            self.get_logger().info(
                f"[SIGN] Target '{expected_sign}' seen on board — approach mode active."
            )

        # ── Extract and apply direction for our target ───────────────────
        direction = char_dir_map[expected_sign]

        if direction not in DIRECTIONAL_SIGNS:
            self.get_logger().warn(
                f"[SIGN] Unknown direction '{direction}' for target '{expected_sign}'."
            )
            return

        if self.pending_direction != direction:
            self.pending_direction = direction
            self._turn_done_count  = 0
            self.get_logger().info(
                f"[SIGN] Target '{expected_sign}' → Direction: '{direction}' "
                f"(full board: {char_dir_map})."
            )


    # ================================================================== #
    #  Callback: QR code                                                  #
    # ================================================================== #

    def qr_detection_callback(self, message):
        """
        Stop-under-QR logic with 1-second disappear debounce:
          1. QR matching active_target seen → slow to QR_CREEP_SPEED,
             set _qr_pending_payload, cancel any in-progress lost timer.
          2. QR gone (empty payload) while _qr_pending_payload set →
             start a one-shot 1-second ROS timer.
             If QR reappears before timer fires, cancel the timer.
             When timer fires → stop + handshake.
        """
        if self.current_state in (State.SERVER_HANDSHAKE, State.MISSION_COMPLETE):
            return

        payload = message.data

        # ── QR disappeared ────────────────────────────────────────────────
        if payload == "":
            if self._qr_pending_payload is None:
                return   # never saw a matching QR — ignore
            if self._qr_lost_timer is None:
                # Start a one-shot 1 s timer.
                self._qr_lost_timer = self.create_timer(1.0, self._on_qr_lost_confirmed)
                self.get_logger().info("[QR] QR out of view — 1 s timer started.")
            return

        # ── QR visible ───────────────────────────────────────────────────
        if self.active_target not in payload:
            return

        # QR is back — cancel the lost timer if running.
        if self._qr_lost_timer is not None:
            self._qr_lost_timer.cancel()
            self._qr_lost_timer = None
            self.get_logger().info("[QR] QR reappeared — lost timer cancelled (dropout ignored).")

        if self._qr_pending_payload is None:
            self.get_logger().info(
                f"[QR] MATCH '{self.active_target}' in '{payload}' — "
                "creeping forward to pass under QR."
            )
            self.target_speed = min(self.target_speed, QR_CREEP_SPEED)

        self._qr_pending_payload = payload

    def _on_qr_lost_confirmed(self):
        """Fired 1 second after QR disappeared — buggy is under the post."""
        self._qr_lost_timer.cancel()
        self._qr_lost_timer = None

        if self._qr_pending_payload is None:
            return   # already handled or reset

        if self.current_state in (State.SERVER_HANDSHAKE, State.MISSION_COMPLETE):
            return

        qr_payload = self._qr_pending_payload
        self._qr_pending_payload = None
        self._qr_lost_since      = None

        self.get_logger().info(
            f"[QR] 1 s confirmed absent — stopped under post. "
            f"Sending '{qr_payload}' to server."
        )
        self.target_speed = 0.0
        self.target_turn  = 0.0
        self.avoiding     = False
        self._transition(
            State.SERVER_HANDSHAKE,
            f"Arrived under QR '{self.active_target}'."
        )
        self.send_server_message(qr_payload)

    # ================================================================== #
    #  Callback: Edge Vectors -> PID steering                            #
    # ================================================================== #

    def edge_vectors_callback(self, message):
        """
        Simple, robust lane controller.

        Logic:
          1. EDGE VIOLATION (absolute)
             Any edge vector within EDGE_SAFE_MARGIN of its wall?
             → hard linear steer away, slow to TURNING_SPEED, return.
             Nothing overrides this.

          2. OBSTACLE AVOIDANCE
             LIDAR flagged an obstacle → steer around it.

          3. DIRECTION-BASED PID TARGET
             Sign says Left  → PID targets FOLLOW_LEFT_TARGET  (-0.35)
             Sign says Right → PID targets FOLLOW_RIGHT_TARGET (+0.35)
             Else            → PID targets FOLLOW_CENTRE_TARGET (0.0)
             PID drives norm_pos toward that target.
             Speed reduced linearly with |turn| (TURNING_SPEED floor).

          4. NO VECTORS → creep with last known turn or straight.
        """
        if self.current_state != State.TRACKING:
            return

        img_w      = float(message.image_width)
        half_width = img_w / 2.0

        # Collect edge x-midpoints.
        edge_xs = []
        if message.vector_count >= 1:
            edge_xs.append((message.vector_1[0].x + message.vector_1[1].x) / 2.0)
        if message.vector_count == 2:
            edge_xs.append((message.vector_2[0].x + message.vector_2[1].x) / 2.0)

        # ════════════════════════════════════════════════════════════════
        #  1. EDGE VIOLATION — absolute, unconditional
        # ════════════════════════════════════════════════════════════════
        if edge_xs:
            margin_px = img_w * EDGE_SAFE_MARGIN
            for ex in edge_xs:
                if ex < margin_px:
                    # Left edge too close — steer right hard.
                    self.target_turn  =  EDGE_RECOVERY_TURN
                    self.target_speed =  TURNING_SPEED
                    self.integral     =  0.0
                    self.get_logger().info(f"[EDGE] Left wall at {ex:.0f}px — recovery RIGHT")
                    return
                if ex > img_w - margin_px:
                    # Right edge too close — steer left hard.
                    self.target_turn  = -EDGE_RECOVERY_TURN
                    self.target_speed =  TURNING_SPEED
                    self.integral     =  0.0
                    self.get_logger().info(f"[EDGE] Right wall at {ex:.0f}px — recovery LEFT")
                    return

        # ════════════════════════════════════════════════════════════════
        #  2. OBSTACLE AVOIDANCE
        # ════════════════════════════════════════════════════════════════
        if self.avoiding:
            if edge_xs:
                mid = sum(edge_xs) / len(edge_xs)
                self.target_turn  = 0.5 if mid < half_width else -0.5
                self.target_speed = AVOIDANCE_SPEED
            else:
                self.target_speed = AVOIDANCE_SPEED
                self.target_turn  = self.avoidance_turn_value
            return

        # ════════════════════════════════════════════════════════════════
        #  3. DIRECTION-BASED PID
        # ════════════════════════════════════════════════════════════════
        # Determine PID target from sign direction.
        if self.pending_direction == "Left":
            pid_target = FOLLOW_LEFT_TARGET
        elif self.pending_direction == "Right":
            pid_target = FOLLOW_RIGHT_TARGET
        else:
            pid_target = FOLLOW_CENTRE_TARGET

        # No edge vectors — creep with bias or damp.
        if not edge_xs:
            self.target_speed = NO_VECTOR_SPEED
            if self.pending_direction in ("Left", "Right"):
                # Hold last turn direction at creep speed.
                self.target_turn = max(TURN_MIN, min(TURN_MAX, pid_target * 1.5))
            else:
                self.target_turn = self.target_turn * 0.5   # damp toward straight
            return

        # Compute centroid (or use single edge if only one visible).
        if len(edge_xs) == 2:
            centroid_x = (edge_xs[0] + edge_xs[1]) / 2.0
        else:
            # Single edge: infer lane centre as one lane-width away from the edge.
            # Use same PID but with the edge position as proxy.
            centroid_x = edge_xs[0]

        norm_pos = (centroid_x - half_width) / half_width   # -1 (left) .. +1 (right)

        # PID error = distance from target position.
        error = norm_pos - pid_target

        # Turn done detection: when centred near target, increment counter.
        if abs(error) < 0.1:
            self._turn_done_count = min(self._turn_done_count + 1, TURN_DONE_FRAMES)
        else:
            self._turn_done_count = 0

        self.integral  += error
        self.integral   = max(-INTEGRAL_CLAMP, min(INTEGRAL_CLAMP, self.integral))
        derivative      = error - self.prev_error
        u               = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        self.target_turn = max(TURN_MIN, min(TURN_MAX, -u))

        # Linear speed reduction with turn magnitude.
        speed = NORMAL_SPEED - SPEED_TURN_SCALE * abs(self.target_turn)
        self.target_speed = max(TURNING_SPEED, min(NORMAL_SPEED, speed))

        self._pid_log_counter += 1
        if self._pid_log_counter >= 30:
            self._pid_log_counter = 0
            self.get_logger().info(
                f"[PID] dir={self.pending_direction} target={pid_target:+.2f} "
                f"norm={norm_pos:+.3f} err={error:+.3f} "
                f"turn={self.target_turn:+.3f} spd={self.target_speed:.3f}"
            )

    # ================================================================== #
    #  Callback: LIDAR                                                    #
    # ================================================================== #

    def lidar_callback(self, message):
        """
        LIDAR — only active in TRACKING.

        Destination arrival is now handled entirely by the QR
        disappear-from-view logic in qr_detection_callback.
        LIDAR is only used for obstacle avoidance here.

        < AVOIDANCE_THRESHOLD : steer around obstacle.
        >= AVOIDANCE_THRESHOLD: cancel avoidance.
        """
        if self.current_state != State.TRACKING:
            return

        N           = len(message.ranges)
        front_start = int(N * 7 / 18)
        front_end   = int(N * 11 / 18)
        front_vals  = [r for r in message.ranges[front_start:front_end]
                       if math.isfinite(r)]
        min_dist    = min(front_vals) if front_vals else math.inf

        if min_dist < AVOIDANCE_THRESHOLD:
            if not self.avoiding:
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

                self.avoiding = True
                self.get_logger().info(
                    f"[AVOID] Obstacle at {min_dist:.2f} m — steering {side}."
                )
                self.target_speed = AVOIDANCE_SPEED
                self.target_turn  = self.avoidance_turn_value
        else:
            if self.avoiding:
                self.avoiding = False
                self.get_logger().info("[AVOID] Path clear — avoidance cancelled.")

    # ================================================================== #
    #  Callback: Server response                                          #
    # ================================================================== #

    def server_communication_callback(self, message):
        """
        Handle server reply.

        Payload format: { src:2, dest:1, uid:0, ack:0, msg:"<letter>" }

        msg field:
          Single sign letter (A/B/C/X/Y/Z) → next target.
          "COMPLETED" (case-insensitive)    → all done, park.
          Anything else                     → acknowledgement, ignore.
        """
        if message.dest != 1:
            return

        msg = message.msg.strip()

        if not msg:
            return

        # ── Mission complete ───────────────────────────────────────────────
        if msg.upper() == "COMPLETED":
            self.get_logger().info("[SERVER] COMPLETED — navigating to parking.")
            self._navigate_to_parking()
            return

        # ── Next target as sign letter ─────────────────────────────────────
        if msg in SIGN_TO_TARGET:
            target_key = SIGN_TO_TARGET[msg]
            self.get_logger().info(
                f"[SERVER] Next target: '{msg}' → '{target_key}'"
            )
            self._set_active_target(target_key)

            if target_key in PATIENT_SEQUENCE:
                self.patients_delivered += 1
                self.get_logger().info(
                    f"[MISSION] Patient {self.patients_delivered}/3 delivered."
                )

            self._transition(
                State.TRACKING,
                f"New target '{target_key}' — resuming."
            )
            return

        # ── Acknowledgement or unknown — ignore silently ───────────────────
        self.get_logger().info(f"[SERVER] Acknowledgement received: '{msg}' — continuing.")

    # ================================================================== #
    #  Parking helpers                                                    #
    # ================================================================== #

    def _navigate_to_parking(self):
        self.get_logger().info("[PARK] Parking — driving straight 3 s.")
        self.target_speed = 0.68  # was 0.45 → +50%
        self.target_turn  = 0.0
        self._parking_timer = self.create_timer(3.0, self._finish_parking)

    def _finish_parking(self):
        self._parking_timer.cancel()
        self.target_speed = 0.0
        self.target_turn  = 0.0
        self._transition(State.MISSION_COMPLETE, "Parked.")
        self.send_server_message("PARKED")
        self.get_logger().info("[MISSION] COMPLETE — all delivered, buggy parked.")


# =========================================================================== #
#  Entry point                                                                 #
# =========================================================================== #

def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[SHUTDOWN] KeyboardInterrupt — shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
