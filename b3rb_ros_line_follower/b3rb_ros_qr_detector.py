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

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np
from pyzbar import pyzbar


class QRDetector(Node):
    """
    ROS 2 Node that processes raw camera images to scan for QR codes.

    Pipeline per frame:
      1. Crop image to top 50% (QR codes appear on signs above the floor).
      2. Convert cropped region to grayscale.
      3. Decode with pyzbar.decode().
      4. On the first QRCODE barcode found, publish the UTF-8 payload to
         /qr_detection and return (one publish per callback invocation).

    Publishes to : /qr_detection  (std_msgs/String)
    Subscribes to: /camera/image_raw/compressed  (sensor_msgs/CompressedImage)
    """

    def __init__(self):
        super().__init__('qr_detector')

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
        #  Publisher: decoded QR payload                                      #
        # ------------------------------------------------------------------ #
        self.publisher_qr = self.create_publisher(
            String,
            '/qr_detection',
            10,
        )

        # Frame counter used to throttle repetitive "no QR found" log lines.
        self._frame_count = 0

        # Track the last published payload to avoid spamming identical messages.
        self._last_published = None

        self.get_logger().info(
            "[INIT] QRDetector node ready. "
            "Using pyzbar for decoding. Publishing to /qr_detection."
        )

    # ---------------------------------------------------------------------- #
    #  Camera callback                                                        #
    # ---------------------------------------------------------------------- #

    def camera_image_callback(self, message):
        """Decode compressed image and attempt QR code detection."""
        self._frame_count += 1

        # Decode JPEG bytes → OpenCV BGR image.
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            self.get_logger().warn("[CAM] Failed to decode compressed image — skipping frame.")
            return

        if self._frame_count % 60 == 0:
            self.get_logger().info(
                f"[CAM] Frame #{self._frame_count}: "
                f"image size {image.shape[1]}x{image.shape[0]}."
            )

        qr_payload = self.detect_qr_code(image)

        if qr_payload is not None:
            # Only log and publish when the payload is new (avoid log/topic spam).
            if qr_payload != self._last_published:
                self.get_logger().info(
                    f"[QR] NEW QR CODE DETECTED: '{qr_payload}' — publishing to /qr_detection"
                )
                self._last_published = qr_payload

            msg = String()
            msg.data = qr_payload
            self.publisher_qr.publish(msg)

        else:
            # Clear last published so the same code gets logged again if re-scanned later.
            if self._last_published is not None:
                self.get_logger().info("[QR] QR code no longer in view.")
                self._last_published = None

    # ---------------------------------------------------------------------- #
    #  Detection logic                                                        #
    # ---------------------------------------------------------------------- #

    def detect_qr_code(self, image):
        """
        Detect and decode a QR code in the top half of the frame.

        Steps
        -----
        1. Crop to top 50% of the image.
        2. Convert cropped region to grayscale.
        3. Pass to pyzbar.decode().
        4. Return the UTF-8 string of the first QRCODE barcode found, or None.

        Returns
        -------
        str | None
        """
        h = image.shape[0]
        cropped = image[0 : h // 2, :, :]

        # Convert to grayscale — pyzbar works on single-channel images.
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        try:
            barcodes = pyzbar.decode(gray)
        except Exception as exc:
            self.get_logger().warn(f"[QR] pyzbar.decode raised an exception: {exc}")
            return None

        if not barcodes:
            return None

        for barcode in barcodes:
            if barcode.type == 'QRCODE':
                return barcode.data.decode('utf-8')

        return None


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #

def main(args=None):
    rclpy.init(args=args)
    node = QRDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[SHUTDOWN] KeyboardInterrupt received — shutting down QRDetector.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
