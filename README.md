# B3RB ROS 2 Hackathon Package (`b3rb_ros_line_follower`)

Welcome to the B3RB Autonomous Buggy Hackathon! This ROS 2 package serves as your starter template for programming a simulated ambulance buggy.

Your goal is to safely navigate a track, avoid static obstacles, identify traffic sign boards, detect patients and hospitals by scanning QR codes, interact with a coordinate server, and park at the final destination.

---

## 1. ROS Node Architecture

Below is the ROS 2 node architecture of the buggy. You are responsible for completing the implementation of these nodes.

![ROS Graph Diagram](ros_graph_diagram.jpg)

### Node Summary Table

| Node Name | Executable Name | Python File | Description |
| :--- | :--- | :--- | :--- |
| `/edge_vectors_publisher` | `vectors` | `b3rb_ros_edge_vectors.py` | Extracts track lane edge vectors from the camera feed. |
| `/line_follower` | `runner` | `b3rb_ros_line_follower.py` | Core controller node. Steers, manages obstacle avoidance, handles planning, server communication, and parking. |
| `/object_recognizer` | `detect` | `b3rb_ros_object_recog.py` | Classifies traffic sign boards using a pre-trained Keras model or OpenCV. |
| `/qr_detector` | `qr_detect` | `b3rb_ros_qr_detector.py` | Scans and decodes QR codes representing patients or hospitals. |

---

## 2. Topic & Message Reference

Below are the ROS 2 topics available in the workspace. Use these to communicate between your nodes and interface with the simulation.

| Topic Name | Message Type | Direction (for `runner`) | Purpose |
| :--- | :--- | :--- | :--- |
| `/camera/image_raw/compressed` | `sensor_msgs/msg/CompressedImage` | Subscribed (by `vectors`, `detect`, `qr_detect`) | Raw compressed video feed from the buggy's front-facing camera. |
| `/scan` | `sensor_msgs/msg/LaserScan` | Subscribed (by `runner`) | 360-degree range measurements from the LIDAR scanner. |
| `/edge_vectors` | `synapse_msgs/msg/EdgeVectors` | Subscribed (by `runner`) | Coordinates representing the left and right lane boundaries. |
| `/sign_board_detection` | `std_msgs/msg/String` | Subscribed (by `runner`) | Decoded labels of detected traffic sign boards (e.g. `"STOP_SIGN"`, `"TURN_LEFT"`). |
| `/qr_detection` | `std_msgs/msg/String` | Subscribed (by `runner`) | Decoded string payload of scanned QR codes (e.g. `"PATIENT_3"`, `"HOSPITAL_A"`). |
| `/ServerCommunication` | `synapse_msgs/msg/ServerCommunication` | Bidirectional (Sub & Pub by `runner`) | Interface with the hackathon scoring/routing server. |
| `/cerebri/in/joy` | `sensor_msgs/msg/Joy` | Published (by `runner`) | Manual control inputs (speed and steering) sent to the autopilot. |

### Visual Debugging Topics
*   `/debug_images/thresh_image` (`sensor_msgs/msg/CompressedImage`): View binary thresholded images from the vector publisher.
*   `/debug_images/vector_image` (`sensor_msgs/msg/CompressedImage`): View images drawn with overlay vectors (blue = raw, green = chosen boundary).

---

## 3. Detailed Message Formats

### A. Steering & Driving: `sensor_msgs/msg/Joy`
To steer and drive the buggy, publish to `/cerebri/in/joy` with the following configuration:
*   `msg.axes[1]`: Forward/Reverse speed. Range is `[-1.0, 1.0]` (positive is forward, negative is reverse).
*   `msg.axes[3]`: Steering turn angle. Range is `[-1.0, 1.0]` (positive is left steer, negative is right steer).
*   `msg.buttons`: Must be set to `[1, 0, 0, 0, 0, 0, 0, 1]` to enable manual joystick overrides.

### B. Lane Vectors: `synapse_msgs/msg/EdgeVectors`
*   `image_height` (`uint32`): Camera frame height (default 240).
*   `image_width` (`uint32`): Camera frame width (default 320).
*   `vector_count` (`uint8`): Number of lane boundaries currently visible (0, 1, or 2).
*   `vector_1` (array of `geometry_msgs/Point`): Represents the first boundary (usually Left). Starts at `vector_1[0]` (minimum y) and ends at `vector_1[1]` (maximum y).
*   `vector_2` (array of `geometry_msgs/Point`): Represents the second boundary (usually Right). Starts at `vector_2[0]` and ends at `vector_2[1]`.

### C. Server Communication: `synapse_msgs/msg/ServerCommunication`
Use this to report your progress and receive mission targets:
*   `src` (`uint8`): ID of the sending component (Buggy is `1`, Server is `2`).
*   `dest` (`uint8`): ID of the receiving component (Buggy is `1`, Server is `2`).
*   `uid` (`uint8`): Unique message identifier (incrementing counter).
*   `ack` (`uint8`): Acknowledge status (0 = blank, 1 = acknowledged).
*   `msg` (`string`): The payload content (e.g. `"PATIENT_PICKED"`, `"REACHED_HOSPITAL"`, etc.).

---

## 4. Compilation & Running

Follow these steps to compile and launch the nodes:

1.  **Source ROS 2 and build the workspace**:
    ```bash
    source /opt/ros/humble/setup.bash
    colcon build --symlink-install
    source install/setup.bash
    ```
    *Note: Using `--symlink-install` allows you to modify Python scripts in the `src/` directory and see changes take effect immediately without recompiling.*

2.  **Run individual nodes**:
    *   **Lane Vector Extractor**:
        ```bash
        ros2 run b3rb_ros_line_follower vectors
        ```
    *   **Main Runner/Controller**:
        ```bash
        ros2 run b3rb_ros_line_follower runner
        ```
    *   **Sign Board Classifier**:
        ```bash
        ros2 run b3rb_ros_line_follower detect
        ```
    *   **QR Scanner Node**:
        ```bash
        ros2 run b3rb_ros_line_follower qr_detect
        ```

---

## 5. Development Hints & Tasks

1.  **Lane Following (`b3rb_ros_edge_vectors.py` & `b3rb_ros_line_follower.py`)**:
    *   The vector node provides a default threshold on gray scale. Optimize this by converting to **HSV space** and thresholding for black lane boundary markings.
    *   In the runner node, calculate the offset between the center of the left and right vectors (`midpoint = (left_x + right_x) / 2.0`) and the center of the image (`160.0`). Adjust steering proportional to this offset.
2.  **Obstacle Avoidance (`b3rb_ros_line_follower.py` / `lidar_callback`)**:
    *   LIDAR ranges are indexed by angles (360 degrees total). Map the range array to find which indices point directly in front.
    *   If the minimum front distance drops below `0.8` meters, override lane steering, steer around the obstacle, and merge back once the scan is clear.
3.  **Sign Board Detection (`b3rb_ros_object_recog.py`)**:
    *   Use color (e.g., Red) or shape (e.g., octagons) to isolate sign boards.
    *   If TensorFlow/Keras is available, load the provided `model.h5` model to classify the sign board type and send the label to the runner.
4.  **Patient/Hospital QR & Range Scan (`b3rb_ros_qr_detector.py` & `b3rb_ros_line_follower.py`)**:
    *   Scan for QR codes using `cv2.QRCodeDetector()` or `pyzbar`.
    *   Compare the scanned string payload to the target destination provided by the server.
    *   To load a patient or drop them off, park close to the building (use LIDAR side scans to check range), stop, and trigger the server confirmation message.
5.  **Parking**:
    *   Once the server coordinates inform you that the mission is complete, navigate to the final hospital parking slot. Check scan ranges to align the buggy and park centered inside the lines.
