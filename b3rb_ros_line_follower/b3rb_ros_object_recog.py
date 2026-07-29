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

                # Attempt CUDA acceleration; fall back to CPU transparently.
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.get_logger().info(
                    "[INIT] Backend set to CUDA. "
                    "(If no CUDA GPU is present, OpenCV will silently fall back to CPU.)"
                )
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
            self.get_logger().info(
                f"[SIGN] *** DETECTED: '{sign_label}' "
                f"(confidence={max_score:.3f}) — published to /sign_board_detection ***"
            )
        else:
            if self._frame_count % 30 == 0:
                self.get_logger().info(
                    f"[SIGN] No sign detected this frame "
                    f"(best score={max_score:.3f} < threshold={CONFIDENCE_THRESHOLD})."
                )

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
            self.get_logger().warn(
                "[INFER] Skipping inference — image is None or model not loaded."
            )
            return None, 0.0

        # 1. Crop to top 50% — sign boards are above the track surface.
        h = image.shape[0]
        cropped = image[0 : h // 2, :, :]
        self.get_logger().info(
            f"[INFER] Cropped to top 50%: {cropped.shape[1]}x{cropped.shape[0]} px."
        )

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
            self.get_logger().error(f"[INFER] Forward pass failed: {exc}")
            return None, 0.0

        self.get_logger().info(
            f"[INFER] Raw output shape: {preds.shape}. "
            "Extracting class scores from rows 4-12."
        )

        if preds.shape[1] < 13:
            self.get_logger().error(
                f"[INFER] Unexpected output shape {preds.shape} — expected (1, 13, N). "
                "Cannot extract class scores."
            )
            return None, 0.0

        # 4. Class confidence matrix: skip first 4 rows (bbox), take rows 4-12 → shape (9, N).
        scores_matrix = preds[0][4:, :]   # shape: (9, 5376)

        max_score = float(np.max(scores_matrix))
        self.get_logger().info(
            f"[INFER] Max confidence across all classes/anchors: {max_score:.4f}."
        )

        if max_score >= CONFIDENCE_THRESHOLD:
            # Find which class row and anchor column produced the max score.
            class_idx, anchor_idx = np.unravel_index(
                np.argmax(scores_matrix), scores_matrix.shape
            )
            label = CLASS_NAMES[class_idx]
            self.get_logger().info(
                f"[INFER] Threshold PASSED → class_idx={class_idx}, "
                f"label='{label}', anchor={anchor_idx}, score={max_score:.4f}."
            )
            return label, max_score

        self.get_logger().info(
            f"[INFER] Threshold NOT met ({max_score:.4f} < {CONFIDENCE_THRESHOLD}). "
            "No sign published."
        )
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
