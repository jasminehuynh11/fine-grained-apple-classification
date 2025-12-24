import sys
import os
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import argparse
 
class CamCapture(Node):
    def __init__(self, name, save_path, num_max):
        super().__init__(name)
        # Subscribe to the camera image topic
        self.cam_subscription = self.create_subscription(
            Image,
            '/depth_cam/rgb/image_raw',
            self.image_callback,
            1
        )
        self.cv_bridge = CvBridge()
        self.num_max = num_max
        self.save_id = 0
        self.capture_next = False  # <-- only save when this is True

        # Prepare save directory
        self.save_path = os.path.abspath(save_path)
        os.makedirs(self.save_path, exist_ok=True)
        self.get_logger().info(f"Saving images to: {self.save_path}")

        # Start a thread that waits for Enter in the terminal
        thread = threading.Thread(target=self._wait_for_enter, daemon=True)
        thread.start()
 
    def _wait_for_enter(self):
        """Block on input() so when user presses Enter, we capture next frame."""
        while rclpy.ok():
            input()                # wait for Enter
            self.capture_next = True
            self.get_logger().info("Trigger received: will capture next frame")
 
    def image_callback(self, msg):
        # Convert ROS Image to OpenCV
        image_bgr = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        cv2.imshow("RGB Preview", image_rgb)

        # If user pressed Enter in terminal, save this frame
        if self.capture_next:
            filename = os.path.join(
                self.save_path,
                f'image{self.save_id:04d}.jpg'
            )
            cv2.imwrite(filename, image_bgr)
            self.get_logger().info(f"Saved {filename}")

            # advance counter
            self.save_id = (self.save_id + 1) % self.num_max
            self.capture_next = False

        # required for imshow to refresh
        cv2.waitKey(1)
 
def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(description='Save images from ROS2 topic')
    parser.add_argument('-o', '--save_path', type=str,
                        default='captured_images',
                        help='Directory where images will be saved')
    parser.add_argument('-n', '--num_max', type=int,
                        default=100,
                        help='Max number of images (ring buffer)')
    parsed_args, _ = parser.parse_known_args()

    node = CamCapture(
        name='capture_sub',
        save_path=parsed_args.save_path,
        num_max=parsed_args.num_max
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down image capture...')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
 
if __name__ == '__main__':
    main()

