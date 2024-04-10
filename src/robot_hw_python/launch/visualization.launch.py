import os

from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.substitutions import TextSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource


from ament_index_python import get_package_share_directory



def generate_launch_description():
    image_rotation_node = Node(
        package="robot_hw_python",
        executable="rotate_images",
        name="rotate_images_from_stretch_head"
    )

    return LaunchDescription([
        image_rotation_node
    ])