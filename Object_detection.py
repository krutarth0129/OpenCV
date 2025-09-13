import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class FaceDetectionFromBag(Node):
    def __init__(self):
        super().__init__('face_detection_from_bag')
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw', 
            self.listener_callback,
            10
        )
        self.bridge = CvBridge()
        
        
        self.model = YOLO('yolov8n.pt')
        self.get_logger().info('YOLOv8n model loaded')

    def listener_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        
        results = self.model(cv_image)

        
        annotated_img = results[0].plot()

        
        cv2.imshow('YOLOv8n Face Detection', annotated_img)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    face_detection_from_bag = FaceDetectionFromBag()
    rclpy.spin(face_detection_from_bag)
    face_detection_from_bag.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
