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
CONFIDENCE_THRESHOLD = 0.5

# Input resolution expected by the ONNX model.
MODEL_INPUT_SIZE = (512, 512)

# Class labels — index matches model output rows 4-12.
CLASS_NAMES = ['A', 'B', 'C', 'Left', 'Right', 'Straight', 'X', 'Y', 'Z']

# Which labels are location characters vs direction arrows.
CHAR_CLASSES = {'A', 'B', 'C', 'X', 'Y', 'Z'}
DIR_CLASSES  = {'Left', 'Right', 'Straight'}

# IoU threshold for Non-Maximum Suppression.
NMS_IOU_THRESHOLD = 0.45

# Maximum pixel distance (in model-input-space) to pair a char with a direction.
PAIR_X_TOLERANCE = 60   # pixels at 512-wide input scale


class ObjectRecognizer(Node):
    """
    ROS 2 Node that processes raw camera images to recognise NXP Cup traffic sign boards.

    Pipeline per frame
    ------------------
    1. Crop image to top 50% (sign boards appear above the track floor).
    2. Build a 512×512 normalised blob.
    3. Run forward inference through an ONNX YOLOv8-style model.
    4. The model outputs shape (1, 13, N):
         - rows 0-3  : cx, cy, w, h  (centre-format bounding box)
         - rows 4-12 : 9 per-class confidence scores
    5. Decode all detections, apply NMS per class.
    6. Pair each character detection with the direction detection whose
       x-centre is closest (within PAIR_X_TOLERANCE pixels).
    7. Publish one message per pair: "<char>:<direction>"  e.g. "A:Left"
       Also publish the full board map once per frame:
         "MAP:A=Left,B=Left,C=Straight,X=Straight,Y=Right,Z=Right"

    Topic published: /sign_board_detection  (std_msgs/String)
    """

    def __init__(self):
        super().__init__('object_recognizer')

        # ------------------------------------------------------------------ #
        #  Load ONNX model                                                    #
        # ------------------------------------------------------------------ #
        self.net = None
        dir_path   = os.path.dirname(os.path.abspath(__file__))
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

                # Attempt CUDA; fall back to CPU.
                try:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    dummy = cv2.dnn.blobFromImage(
                        np.zeros((8, 8, 3), dtype=np.uint8),
                        scalefactor=1.0 / 255.0,
                        size=(8, 8), swapRB=True, crop=False,
                    )
                    self.net.setInput(dummy)
                    self.net.forward()
                    self.get_logger().info("[INIT] CUDA backend confirmed — using GPU inference.")
                except Exception:
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
        #  Subscription / Publisher                                           #
        # ------------------------------------------------------------------ #
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10,
        )

        self.publisher_sign = self.create_publisher(
            String,
            '/sign_board_detection',
            10,
        )

        self._frame_count = 0
        self._last_map_str = ""   # last published MAP string — avoid spam

        self.get_logger().info(
            "[INIT] ObjectRecognizer ready. "
            f"Classes: {CLASS_NAMES}. Conf threshold: {CONFIDENCE_THRESHOLD}."
        )

    # ---------------------------------------------------------------------- #
    #  Camera callback                                                        #
    # ---------------------------------------------------------------------- #

    def camera_image_callback(self, message):
        self._frame_count += 1

        np_arr = np.frombuffer(message.data, np.uint8)
        image  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            self.get_logger().warn("[CAM] Failed to decode compressed image — skipping.")
            return

        if self._frame_count % 30 == 0:
            self.get_logger().info(
                f"[CAM] Frame #{self._frame_count} "
                f"({image.shape[1]}x{image.shape[0]})"
            )

        char_dir_map = self.detect_sign_map(image)

        if not char_dir_map:
            if self._last_map_str:
                self.get_logger().info("[SIGN] Board no longer in view.")
                self._last_map_str = ""
            return

        # ── Publish MAP message (full board) ──────────────────────────────
        map_str = "MAP:" + ",".join(
            f"{ch}={dr}" for ch, dr in sorted(char_dir_map.items())
        )
        if map_str != self._last_map_str:
            self.get_logger().info(f"[SIGN] Board map: {map_str}")
            self._last_map_str = map_str

        msg      = String()
        msg.data = map_str
        self.publisher_sign.publish(msg)

    # ---------------------------------------------------------------------- #
    #  Full detection pipeline                                                #
    # ---------------------------------------------------------------------- #

    def detect_sign_map(self, image):
        """
        Run YOLOv8 inference and return a dict mapping each detected
        character to its paired direction.

        Returns
        -------
        dict[str, str]  e.g. {'A': 'Left', 'B': 'Left', 'C': 'Straight'}
        Empty dict when nothing is detected or model is not loaded.
        """
        if image is None or self.net is None:
            return {}

        # 1. Crop top 50%.
        h        = image.shape[0]
        cropped  = image[0 : h // 2, :, :]
        crop_h, crop_w = cropped.shape[:2]

        # 2. Build blob.
        blob = cv2.dnn.blobFromImage(
            cropped,
            scalefactor=1.0 / 255.0,
            size=MODEL_INPUT_SIZE,
            swapRB=True,
            crop=False,
        )

        # 3. Forward pass.
        self.net.setInput(blob)
        try:
            preds = self.net.forward()   # shape: (1, 13, N)
        except Exception as exc:
            if self._frame_count % 30 == 0:
                self.get_logger().error(f"[INFER] Forward pass failed: {exc}")
            return {}

        if preds.ndim != 3 or preds.shape[1] < 13:
            if self._frame_count % 30 == 0:
                self.get_logger().error(
                    f"[INFER] Unexpected output shape {preds.shape}."
                )
            return {}

        # 4. Decode detections.
        #    preds[0] shape: (13, N)
        #    rows 0-3 : cx, cy, w, h  (normalised to MODEL_INPUT_SIZE)
        #    rows 4-12: class confidence scores
        output   = preds[0]           # (13, N)
        boxes_xywh  = output[:4, :].T   # (N, 4)  cx cy w h
        scores_all  = output[4:,  :].T  # (N, 9)

        detections = self._decode_detections(boxes_xywh, scores_all)

        if not detections:
            return {}

        # 5. Separate characters from directions.
        chars = [(label, cx, score) for label, cx, score in detections if label in CHAR_CLASSES]
        dirs  = [(label, cx, score) for label, cx, score in detections if label in DIR_CLASSES]

        if self._frame_count % 30 == 0:
            self.get_logger().info(
                f"[DETECT] chars={[(l,round(x)) for l,x,_ in chars]} "
                f"dirs={[(l,round(x)) for l,x,_ in dirs]}"
            )

        if not chars or not dirs:
            return {}

        # 6. Pair each character to the closest direction by x-centre.
        char_dir_map = {}
        for char_label, char_cx, _ in chars:
            best_dir   = None
            best_dist  = float('inf')
            for dir_label, dir_cx, _ in dirs:
                dist = abs(char_cx - dir_cx)
                if dist < best_dist:
                    best_dist = dist
                    best_dir  = dir_label
            if best_dir is not None and best_dist <= PAIR_X_TOLERANCE:
                char_dir_map[char_label] = best_dir

        return char_dir_map

    # ---------------------------------------------------------------------- #
    #  Decode + NMS helper                                                    #
    # ---------------------------------------------------------------------- #

    def _decode_detections(self, boxes_xywh, scores_all):
        """
        Apply per-class NMS and return a flat list of
        (class_label, cx_in_model_pixels, confidence).

        Parameters
        ----------
        boxes_xywh : np.ndarray  (N, 4)  — cx, cy, w, h normalised 0-1
        scores_all : np.ndarray  (N, 9)  — per-class confidence scores
        """
        results = []

        # Convert normalised cx/cy/w/h → pixel x1,y1,x2,y2 in model space.
        W = MODEL_INPUT_SIZE[0]
        H = MODEL_INPUT_SIZE[1]

        cx = boxes_xywh[:, 0] * W
        cy = boxes_xywh[:, 1] * H
        w  = boxes_xywh[:, 2] * W
        h  = boxes_xywh[:, 3] * H

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        # Per-class NMS.
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            cls_scores = scores_all[:, cls_idx]
            mask       = cls_scores >= CONFIDENCE_THRESHOLD

            if not np.any(mask):
                continue

            cls_boxes  = np.stack([x1[mask], y1[mask], x2[mask], y2[mask]], axis=1)
            cls_confs  = cls_scores[mask]
            cls_cx     = cx[mask]

            # cv2.dnn.NMSBoxes expects list of [x, y, w, h].
            nms_input = [[float(b[0]), float(b[1]),
                          float(b[2] - b[0]), float(b[3] - b[1])]
                         for b in cls_boxes]

            indices = cv2.dnn.NMSBoxes(
                nms_input,
                cls_confs.tolist(),
                CONFIDENCE_THRESHOLD,
                NMS_IOU_THRESHOLD,
            )

            if len(indices) == 0:
                continue

            # indices may be (N,1) or (N,) depending on OpenCV version.
            indices = indices.flatten() if hasattr(indices, 'flatten') else list(indices)

            for i in indices:
                results.append((cls_name, float(cls_cx[i]), float(cls_confs[i])))

        return results


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #

def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[SHUTDOWN] KeyboardInterrupt — shutting down ObjectRecognizer.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
