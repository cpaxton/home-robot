import rclpy
from rclpy.node import Node

import numpy as np
# import tf
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import trimesh.transformations as tra
from geometry_msgs.msg import PoseStamped

from home_robot.motion.stretch import (
    STRETCH_BASE_FRAME,
    STRETCH_CAMERA_FRAME,
    STRETCH_HEAD_CAMERA_ROTATIONS,
)
from home_robot.utils.pose import to_matrix
from robot_hw_python.ros.utils import matrix_to_pose_msg

class CameraPosePublisher(Node):

    def __init__(self, topic_name: str = "camera_pose"):
        super().__init__("camera_pose_publisher")

        self._pub = self.create_publisher(PoseStamped, topic_name, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._seq = 0

        timer_freq = 10
        self.timer = self.create_timer(1 / timer_freq, self.timer_callback)

    def timer_callback(self):
        try:
            t = self.tf_buffer.lookup_transform(
                STRETCH_BASE_FRAME, STRETCH_CAMERA_FRAME, rclpy.time.Time()
            )
            trans, rot = t.transform.translation, t.transform.rotation
            self.get_logger().info(f"translation matrix {trans}")
            self.get_logger().info(f"rotation matrix {rot}")
            matrix = to_matrix(trans, rot)

            # We rotate by 90 degrees from the frame of realsense hardware since we are also rotating images to be upright
            matrix_rotated = matrix @ tra.euler_matrix(0, 0, -np.pi / 2)

            msg = PoseStamped(pose=matrix_to_pose_msg(matrix_rotated))
            msg.header.stamp = rospy.Time.now()
            msg.header.seq = self._seq
            self._pub.publish(msg)
            self._seq += 1
        except TransformException as ex:
            self.get_logger().info(
                f"Could not tranform the camera pose"
            )

def main(args=None):
    rclpy.init()

    camera_pose_publisher = CameraPosePublisher()
    rclpy.spin(camera_pose_publisher)

    camera_pose_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

