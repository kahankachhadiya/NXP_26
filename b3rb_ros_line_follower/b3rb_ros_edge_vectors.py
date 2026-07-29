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
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
from synapse_msgs.msg import EdgeVectors


QOS_PROFILE_DEFAULT = 10
PI = math.pi

# ── HSV black-line segmentation bounds ──────────────────────────────────────
# Hue   : 0–180  (any hue — black has no meaningful hue)
# Sat   : 0–255  (any saturation)
# Value : 0–50   (very dark pixels only → isolates the black track lines)
LOWER_BLACK = np.array([0,   0,   0],  dtype=np.uint8)
UPPER_BLACK = np.array([180, 255, 50], dtype=np.uint8)

# Debug overlay colours (BGR)
RED_COLOR   = (0,   0,   255)
BLUE_COLOR  = (255, 0,   0)
GREEN_COLOR = (0,   255, 0)

# What fraction of the image height (from the bottom) is analyzed.
# Smaller value → looks further ahead; larger value → looks directly below buggy.
VECTOR_IMAGE_HEIGHT_PERCENTAGE = 0.225

# Discard tiny contour vectors that are likely noise.
VECTOR_MAGNITUDE_MINIMUM = 2.25


class EdgeVectorsPublisher(Node):
    """
    ROS 2 Node that detects lane boundary vectors from a compressed camera feed.

    Processing pipeline (per frame):
      1. GaussianBlur  — removes pixel-level noise before color segmentation.
      2. BGR → HSV     — color space that makes black separation reliable under
                         varying illumination.
      3. inRange mask  — isolates black track lines via LOWER/UPPER_BLACK bounds.
      4. Crop bottom N%— focuses analysis on the region just ahead of the buggy.
      5. findContours  — finds blobs of black pixels.
      6. Vector extraction — each contour becomes a directional vector (top-point
                             to bottom-point).  Short/noisy vectors are dropped.
      7. Left/Right split— vectors are split by image centre-line; the closest
                           one on each side is selected as the lane boundary.
      8. Publish EdgeVectors + debug images.

    Publishes:
        /edge_vectors               (synapse_msgs/EdgeVectors)
        /debug_images/thresh_image  (sensor_msgs/CompressedImage)
        /debug_images/vector_image  (sensor_msgs/CompressedImage)
    """

    def __init__(self):
        super().__init__('edge_vectors_publisher')

        # ------------------------------------------------------------------ #
        #  Subscription                                                       #
        # ------------------------------------------------------------------ #
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            QOS_PROFILE_DEFAULT,
        )

        # ------------------------------------------------------------------ #
        #  Publishers                                                         #
        # ------------------------------------------------------------------ #
        self.publisher_edge_vectors = self.create_publisher(
            EdgeVectors,
            '/edge_vectors',
            QOS_PROFILE_DEFAULT,
        )
        self.publisher_thresh_image = self.create_publisher(
            CompressedImage,
            '/debug_images/thresh_image',
            QOS_PROFILE_DEFAULT,
        )
        self.publisher_vector_image = self.create_publisher(
            CompressedImage,
            '/debug_images/vector_image',
            QOS_PROFILE_DEFAULT,
        )

        # Image geometry — populated on first frame.
        self.image_height = 0
        self.image_width  = 0
        self.lower_image_height = 0
        self.upper_image_height = 0

        # Frame counter for throttled logging.
        self._frame_count = 0

        self.get_logger().info(
            "[INIT] EdgeVectorsPublisher ready. "
            f"HSV black range: {LOWER_BLACK.tolist()} → {UPPER_BLACK.tolist()}. "
            f"Crop window: bottom {VECTOR_IMAGE_HEIGHT_PERCENTAGE*100:.1f}% of frame."
        )

    # ---------------------------------------------------------------------- #
    #  Debug image helper                                                     #
    # ---------------------------------------------------------------------- #

    def publish_debug_image(self, publisher, image):
        """Encode an OpenCV image as JPEG and publish as CompressedImage."""
        msg = CompressedImage()
        _, encoded = cv2.imencode('.jpg', image)
        msg.format = 'jpeg'
        msg.data   = encoded.tobytes()
        publisher.publish(msg)

    # ---------------------------------------------------------------------- #
    #  Geometry helper                                                        #
    # ---------------------------------------------------------------------- #

    def get_vector_angle_in_radians(self, vector):
        """
        Return the slope angle (in radians) of the vector from point[0] to point[1].
        Handles the vertical-line edge case (dx == 0) by returning PI/2.
        """
        dx = vector[0][0] - vector[1][0]
        if dx == 0:
            return PI / 2
        slope = (vector[1][1] - vector[0][1]) / dx
        return math.atan(slope)

    # ---------------------------------------------------------------------- #
    #  Contour → vector extraction                                            #
    # ---------------------------------------------------------------------- #

    def compute_vectors_from_image(self, image, thresh):
        """
        Extract lane edge vectors from the binary threshold mask.

        For each external contour:
          - Find the top-most and bottom-most coordinate pairs.
          - Compute vector magnitude; discard if < VECTOR_MAGNITUDE_MINIMUM.
          - Record distance from the buggy's viewpoint (image bottom-centre).
          - Adjust the representative point based on slope direction.

        Returns
        -------
        vectors : list of [top_coord, bottom_coord, distance]
        image   : debug BGR image with all raw vectors drawn in blue.
        """
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        self.get_logger().info(
            f"[CONTOUR] Found {len(contours)} raw contour(s) in cropped region."
        )

        vectors = []
        for i, contour in enumerate(contours):
            coords = contour[:, 0, :]          # shape: (N, 2) — (x, y) pairs

            min_y = int(np.min(coords[:, 1]))
            max_y = int(np.max(coords[:, 1]))

            min_y_pts = coords[coords[:, 1] == min_y]
            max_y_pts = coords[coords[:, 1] == max_y]

            top_pt    = list(min_y_pts[0])
            bottom_pt = list(max_y_pts[0])

            magnitude = float(np.linalg.norm(
                np.array(top_pt, dtype=float) - np.array(bottom_pt, dtype=float)
            ))

            if magnitude <= VECTOR_MAGNITUDE_MINIMUM:
                self.get_logger().info(
                    f"[CONTOUR] Contour #{i} discarded — magnitude={magnitude:.2f} "
                    f"< min={VECTOR_MAGNITUDE_MINIMUM}."
                )
                continue

            # Distance from the bottom-centre of the crop window (buggy's viewpoint).
            rover_pt   = np.array([self.image_width / 2.0, self.lower_image_height])
            mid_pt     = (np.array(top_pt, dtype=float) + np.array(bottom_pt, dtype=float)) / 2.0
            distance   = float(np.linalg.norm(mid_pt - rover_pt))

            # Correct representative point based on slope direction.
            angle = self.get_vector_angle_in_radians([top_pt, bottom_pt])
            if angle > 0:
                top_pt[0] = int(np.max(min_y_pts[:, 0]))
            else:
                bottom_pt[0] = int(np.max(max_y_pts[:, 0]))

            self.get_logger().info(
                f"[CONTOUR] Contour #{i} accepted — "
                f"top={top_pt}, bottom={bottom_pt}, "
                f"magnitude={magnitude:.2f}, distance={distance:.2f}, angle={math.degrees(angle):.1f}°."
            )

            vectors.append([top_pt, bottom_pt, distance])

            # Draw raw vector in blue on debug image.
            cv2.line(image, tuple(top_pt), tuple(bottom_pt), BLUE_COLOR, 2)

        return vectors, image

    # ---------------------------------------------------------------------- #
    #  Full processing pipeline                                               #
    # ---------------------------------------------------------------------- #

    def process_image_for_edge_vectors(self, image):
        """
        Full HSV segmentation and vector extraction pipeline.

        Steps
        -----
        1. Measure image geometry and compute crop boundaries.
        2. GaussianBlur (5×5) — suppress noise before HSV conversion.
        3. BGR → HSV.
        4. inRange mask with LOWER/UPPER_BLACK → binary thresh.
        5. Crop thresh and colour image to bottom VECTOR_IMAGE_HEIGHT_PERCENTAGE.
        6. Extract contour vectors, sort by distance (closest first).
        7. Split vectors left/right; select the closest from each side.
        8. Remap y-coordinates back to full-image space.
        9. Publish debug images.

        Returns
        -------
        final_vectors : list of at most 2 vectors, each [[x0,y0],[x1,y1]].
        """
        self.image_height, self.image_width, _ = image.shape
        self.lower_image_height = int(self.image_height * VECTOR_IMAGE_HEIGHT_PERCENTAGE)
        self.upper_image_height = self.image_height - self.lower_image_height

        self.get_logger().info(
            f"[PIPELINE] Image {self.image_width}x{self.image_height}. "
            f"Crop rows {self.upper_image_height}–{self.image_height} "
            f"({self.lower_image_height}px tall)."
        )

        # Step 1: Gaussian blur — reduces pixel noise before HSV conversion.
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        self.get_logger().info("[PIPELINE] GaussianBlur (5×5) applied.")

        # Step 2: Convert to HSV.
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        self.get_logger().info("[PIPELINE] Converted BGR → HSV.")

        # Step 3: Threshold — isolate black pixels.
        thresh = cv2.inRange(hsv, LOWER_BLACK, UPPER_BLACK)
        black_pixel_count = int(np.sum(thresh > 0))
        self.get_logger().info(
            f"[PIPELINE] HSV inRange complete — {black_pixel_count} black pixels detected."
        )

        # Step 4: Crop both images to the analysis strip near the buggy.
        thresh_cropped = thresh[self.upper_image_height:, :]
        image_cropped  = image[self.upper_image_height:, :].copy()

        # Step 5: Extract vectors from contours.
        vectors, debug_img = self.compute_vectors_from_image(image_cropped, thresh_cropped)
        self.get_logger().info(
            f"[PIPELINE] {len(vectors)} valid vector(s) extracted after contour processing."
        )

        # Step 6: Sort by distance (closest to buggy first).
        vectors = sorted(vectors, key=lambda v: v[2])

        # Step 7: Split by image centre-line.
        half_width     = self.image_width / 2.0
        vectors_left   = [v for v in vectors if (v[0][0] + v[1][0]) / 2.0 <  half_width]
        vectors_right  = [v for v in vectors if (v[0][0] + v[1][0]) / 2.0 >= half_width]

        self.get_logger().info(
            f"[PIPELINE] Split → left={len(vectors_left)}, right={len(vectors_right)} vector(s)."
        )

        final_vectors = []
        side_labels   = ['LEFT', 'RIGHT']

        for label, side_vectors in zip(side_labels, [vectors_left, vectors_right]):
            if not side_vectors:
                self.get_logger().info(
                    f"[PIPELINE] No {label} vector found — boundary missing."
                )
                continue

            best = side_vectors[0]

            # Draw the selected lane boundary in green.
            cv2.line(debug_img, tuple(best[0]), tuple(best[1]), GREEN_COLOR, 2)

            self.get_logger().info(
                f"[PIPELINE] Best {label} vector: "
                f"top={best[0]}, bottom={best[1]}, dist={best[2]:.2f}."
            )

            # Remap y-coordinates from cropped to full image space.
            best[0][1] += self.upper_image_height
            best[1][1] += self.upper_image_height

            final_vectors.append(best[:2])

        self.get_logger().info(
            f"[PIPELINE] Publishing {len(final_vectors)} edge vector(s)."
        )

        # Publish debug images (viewable in Foxglove / RViz).
        self.publish_debug_image(self.publisher_thresh_image, thresh_cropped)
        self.publish_debug_image(self.publisher_vector_image, debug_img)

        return final_vectors

    # ---------------------------------------------------------------------- #
    #  Camera callback                                                        #
    # ---------------------------------------------------------------------- #

    def camera_image_callback(self, message):
        """Decode compressed image, extract edge vectors, and publish."""
        self._frame_count += 1

        np_arr = np.frombuffer(message.data, np.uint8)
        image  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            self.get_logger().warn("[CAM] Failed to decode compressed image — skipping frame.")
            return

        if self._frame_count % 30 == 0:
            self.get_logger().info(
                f"[CAM] Processing frame #{self._frame_count}."
            )

        vectors = self.process_image_for_edge_vectors(image)

        # ── Construct EdgeVectors message ──────────────────────────────── #
        msg = EdgeVectors()
        msg.image_height  = image.shape[0]
        msg.image_width   = image.shape[1]
        msg.vector_count  = 0

        if len(vectors) > 0:
            msg.vector_1[0].x = float(vectors[0][0][0])
            msg.vector_1[0].y = float(vectors[0][0][1])
            msg.vector_1[1].x = float(vectors[0][1][0])
            msg.vector_1[1].y = float(vectors[0][1][1])
            msg.vector_count += 1
            self.get_logger().info(
                f"[PUBLISH] Vector 1 (LEFT boundary): "
                f"({msg.vector_1[0].x:.1f},{msg.vector_1[0].y:.1f}) → "
                f"({msg.vector_1[1].x:.1f},{msg.vector_1[1].y:.1f})."
            )

        if len(vectors) > 1:
            msg.vector_2[0].x = float(vectors[1][0][0])
            msg.vector_2[0].y = float(vectors[1][0][1])
            msg.vector_2[1].x = float(vectors[1][1][0])
            msg.vector_2[1].y = float(vectors[1][1][1])
            msg.vector_count += 1
            self.get_logger().info(
                f"[PUBLISH] Vector 2 (RIGHT boundary): "
                f"({msg.vector_2[0].x:.1f},{msg.vector_2[0].y:.1f}) → "
                f"({msg.vector_2[1].x:.1f},{msg.vector_2[1].y:.1f})."
            )

        if msg.vector_count == 0:
            self.get_logger().warn(
                "[PUBLISH] No edge vectors detected this frame — "
                "buggy may be off-track or lane markings are not visible."
            )

        self.publisher_edge_vectors.publish(msg)


# --------------------------------------------------------------------------- #
#  Entry point                                                                 #
# --------------------------------------------------------------------------- #

def main(args=None):
    rclpy.init(args=args)
    node = EdgeVectorsPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            "[SHUTDOWN] KeyboardInterrupt received — shutting down EdgeVectorsPublisher."
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
