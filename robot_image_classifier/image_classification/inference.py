#!/usr/bin/env python3
"""
inference.py

ROS2 node that:
 - Subscribes to camera images
 - Runs a fine-tuned EfficientNet-B0 model (3 classes)
 - Performs unique action for each class
"""

# ============ IMPORTS ============

# Multi-threading
import threading

# Python API for ROS2
import rclpy
from rclpy.node import Node

# Handling images
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from PIL import Image as PILImage

# Arm control
from ros_robot_controller_msgs.msg import ServosPosition, ServoPosition

# Handling our model
import torch
from torchvision import transforms, models
import torch.nn.functional as F
from .model_wrapper import EfficientNetB0GroupFineTuner


# Inference node
class InferenceNode(Node):
    """
    Contains code for initializing the model, using it for inference,
    performing unique actions based on unique classes, and
    toggling action for restoring the arm to default position.
    """

    def __init__(self):
        super().__init__("inference_node")

        self.bridge = CvBridge()

        # Publisher for arm movement
        self.arm_pub = self.create_publisher(
            ServosPosition, "/ros_robot_controller/bus_servo/set_position", 10
        )

        # Camera subscriber
        self.create_subscription(
            Image, "/depth_cam/rgb/image_raw", self.image_callback, 1
        )

        # ── Replacing manual load with wrapper ──
        wrapper = EfficientNetB0GroupFineTuner(
            checkpoint_path="/home/ubuntu/47892641/COMP8430_week08/src/image_classification/image_classification/efficientnet_b0_group_best_no_aug.pth",
            num_classes=3,
        )
        self.model = wrapper.model
        self.device = wrapper.device
        self.idx_to_class = wrapper.idx_to_class

        # Preprocessing stays the same
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.threshold = 0.75
        self.lock = threading.Lock()
        self.last_cmd = None
        self.active = False
        self.get_logger().info("Inference node initialized.")

    def image_callback(self, msg: Image):
        """
        This function infers the image's class and
        performs action corresponding to that class.
        """

        # Processing image into tensor
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, 0)
            conf, idx = conf.item(), idx.item()

        if conf >= self.threshold:
            # Arm movements based on detected classes
            if idx == 0:  # Autumn Royal
                arm_msg_1 = ServosPosition(
                    duration=1.5,
                    position=[
                        ServoPosition(id=1, position=900),  # left
                        ServoPosition(id=2, position=650),  # mid-high
                        ServoPosition(id=3, position=50),  # curled
                        ServoPosition(id=4, position=300),  # low
                        ServoPosition(id=10, position=100),  # open
                    ],
                )
                self.arm_pub.publish(arm_msg_1)
                self.get_logger().info("Arm command sent.")
            elif idx == 1:  # Crimson Seedless
                arm_msg_2 = ServosPosition(
                    duration=1.5,
                    position=[
                        ServoPosition(id=1, position=500),  # straight
                        ServoPosition(id=2, position=500),  # straight
                        ServoPosition(id=3, position=500),  # straight
                        ServoPosition(id=4, position=500),  # straight
                        ServoPosition(id=10, position=100),  # open
                    ],
                )
                self.arm_pub.publish(arm_msg_2)
                self.get_logger().info("Arm command sent.")
            elif idx == 2:  # Thompson Seedless
                arm_msg_3 = ServosPosition(
                    duration=1.5,
                    position=[
                        ServoPosition(id=1, position=100),  # right
                        ServoPosition(id=2, position=650),  # right
                        ServoPosition(id=3, position=50),  # curled
                        ServoPosition(id=4, position=300),  # low
                        ServoPosition(id=10, position=500),  # semi-closed
                    ],
                )
                self.arm_pub.publish(arm_msg_3)
                self.get_logger().info("Arm command sent.")

            self.get_logger().info(
                f"Detected idx: {idx}, class: {self.idx_to_class[idx]} (conf={conf:.2f}). "
                "Press Enter to toggle action."
            )

    def toggle_action(self):
        """
        Toggle action to bring arm back to default position.
        """

        if not self.active:

            # Bring arm back to default position
            hold_msg = ServosPosition()
            hold_msg.duration = 1.5
            hold_msg.position = [
                # low-middle position, best for trying inference
                ServoPosition(id=1, position=500),
                ServoPosition(id=2, position=350),
                ServoPosition(id=10, position=500),
            ]
            self.arm_pub.publish(hold_msg)

            self.active = False
            self.get_logger().info("STOPPED movement.")
        else:
            self.get_logger().warn("No detected command to execute.")


def listen_for_enter(node: InferenceNode):
    """
    Listening for Enter key to toggle action.
    """

    while rclpy.ok():
        try:
            input()
        except EOFError:
            break
        node.toggle_action()


def main(args=None):
    """
    Bringing all of the above together for execution.
    """

    rclpy.init(args=args)
    node = InferenceNode()
    listener = threading.Thread(target=listen_for_enter, args=(node,), daemon=True)
    listener.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
