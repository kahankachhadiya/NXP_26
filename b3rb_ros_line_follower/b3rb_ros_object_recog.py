# Copyright 2024-2026 NXP
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

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np


# Confidence threshold: detections below this score are discarded.
CONFIDENCE_THRESHOLD = 0.7

# Input resolution expected by the ONNX model.
MODEL_INPUT_SIZE = (512, 512)

# Class labels matching model output rows 4-12 (row index = class index).
CLASS_NAMES = ['A', 'B', 'C', 'Left', 'Right', 'Straight', 'X', 'Y', 'Z']


class ObjectRecognizer(Node):
    """
    ROS 2 Node that processes raw camera images to recognize NXP Cup traffic sign boards.

    Pipeline per frame:
      1. Crop image to top 50% (sign boards appear above the track floor).
      2. Build a 512x512 normalized blob.
      3. Run forward inference through an ONNX YOLOv8-style model.
      4. The model outputs shape (1, 13, 5376):
           - rows 0-3  : bounding box coordinates (skipped)
           - rows 4-12 : 9 per-class confidence scores
      5. If max confidence >= 0.7, publish class label to /sign_board_detection.

    Hardware: CUDA backend is attempted first; falls back to CPU if unavailable.
    """

    def __init__(self):
        super().__init__('object_recognizer')

        # ------------------------------------------------------------------ #
        #  Load ONNX model                                                    #
        # ------------------------------------------------------------------ #
        self.net = None
        dir_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(dir_path, 'best.onnx')

        self.get_logger().info(f"[INIT] Looking for ONNX model at: {model_path}")

        if not os.path.exists(model_path):
            self.get_logger().error(
                f"[INIT] CRITICAL — model file NOT found: {model_path}. "
                "Sign recognition will be disabled."
            )
        else:
            try:
                self.net = cv2.dnn.readNetFromONNX(model_path)
                self.get_logger().info("[INIT] ONNX model loaded successfully.")

                # Attempt CUDA acceleration; fall back to CPU if unavailable.
                cuda_available = False
                try:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    # Probe with a tiny dummy blob to confirm CUDA actually works.
                    dummy = cv2.dnn.blobFromImage(
                        np.zeros((8, 8, 3), dtype=np.uint8),
                        scalefactor=1.0 / 255.0,
                        size=(8, 8),
                        swapRB=True,
                        crop=False,
                    )
                    self.net.setInput(dummy)
                    self.net.forward()
                    cuda_available = True
                    self.get_logger().info("[INIT] CUDA backend confirmed — using GPU inference.")
                except Exception:
                    # CUDA not available or not working; fall back to CPU.
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    self.get_logger().info("[INIT] CUDA unavailable — falling back to CPU inference.")

            except Exception as exc:
                self.get_logger().error(
                    f"[INIT] Failed to load DNN model: {exc}. "
                    "Sign recognition will be disabled."
                )
                self.net = None

        # ------------------------------------------------------------------ #
        #  Subscription: compressed camera feed                               #
        # ------------------------------------------------------------------ #
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10,
        )

        # ------------------------------------------------------------------ #
        #  Publisher: detected sign label                                     #
        # ------------------------------------------------------------------ #
        self.publisher_sign = self.create_publisher(
            String,
            '/sign_board_detection',
            10,
        )

        # Frame counter — reduces log spam (log every 30th frame at INFO level).
        self._frame_count = 0

        # Track last published sign to avoid duplicate log lines.
        self._last_sign = None

        self.get_logger().info(
            "[INIT] ObjectRecognizer node ready. "
            f"Class map: {CLASS_NAMES}. "
            f"Confidence threshold: {CONFIDENCE_THRESHOLD}."
        )

    # ---------------------------------------------------------------------- #
    #  Camera callback                                                        #
    # ---------------------------------------------------------------------- #

    def camera_image_callback(self, message):
        """Decode compressed image and run sign classification."""
        self._frame_count += 1

        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            self.get_logger().warn("[CAM] Failed to decode compressed image — skipping frame.")
            return

        if self._frame_count % 30 == 0:
            self.get_logger().info(
                f"[CAM] Processing frame #{self._frame_count} "
                f"(size {image.shape[1]}x{image.shape[0]})."
            )

        sign_label, max_score = self.classify_sign(image)

        if sign_label is not None:
            msg = String()
            msg.data = sign_label
            self.publisher_sign.publish(msg)
            # Only log when the detected sign changes.
            if sign_label != self._last_sign:
                self.get_logger().info(
                    f"[SIGN] DETECTED: '{sign_label}' (confidence={max_score:.3f})"
                )
                self._last_sign = sign_label
        else:
            if self._last_sign is not None:
                self.get_logger().info("[SIGN] No sign in view.")
                self._last_sign = None

    # ---------------------------------------------------------------------- #
    #  Inference logic                                                        #
    # ---------------------------------------------------------------------- #

    def classify_sign(self, image):
        """
        Run ONNX inference on the top 50% of the input frame.

        Returns
        -------
        (label: str | None, max_score: float)
            label     : class name when confidence >= CONFIDENCE_THRESHOLD, else None.
            max_score : highest confidence value found across all classes and anchors.
        """
        if image is None or self.net is None:
            return None, 0.0

        # 1. Crop to top 50% — sign boards are above the track surface.
        h = image.shape[0]
        cropped = image[0 : h // 2, :, :]

        # 2. Build normalized blob: scale=1/255, size=512x512, swap BGR→RGB, no crop.
        blob = cv2.dnn.blobFromImage(
            cropped,
            scalefactor=1.0 / 255.0,
            size=MODEL_INPUT_SIZE,
            swapRB=True,
            crop=False,
        )

        # 3. Forward pass — expected output shape: (1, 13, 5376).
        self.net.setInput(blob)
        try:
            preds = self.net.forward()
        except Exception as exc:
            # Only log forward-pass errors once per 30 frames to avoid spam.
            if self._frame_count % 30 == 0:
                self.get_logger().error(f"[INFER] Forward pass failed: {exc}")
            return None, 0.0

        if preds.shape[1] < 13:
            if self._frame_count % 30 == 0:
                self.get_logger().error(
                    f"[INFER] Unexpected output shape {preds.shape} — expected (1, 13, N)."
                )
            return None, 0.0

        # 4. Class confidence matrix: skip first 4 rows (bbox), take rows 4-12 → shape (9, N).
        scores_matrix = preds[0][4:, :]   # shape: (9, 5376)

        max_score = float(np.max(scores_matrix))

        if max_score >= CONFIDENCE_THRESHOLD:
            class_idx, _ = np.unravel_index(
                np.argmax(scores_matrix), scores_matrix.shape
            )
            return CLASS_NAMES[class_idx], max_score

        return None, max_score


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #

def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[SHUTDOWN] KeyboardInterrupt received — shutting down ObjectRecognizer.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
