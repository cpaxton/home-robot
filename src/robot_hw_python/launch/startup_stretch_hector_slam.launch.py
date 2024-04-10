import os

from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.substitutions import TextSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource


from ament_index_python import get_package_share_directory



def generate_launch_description():
    start_robot_arg = DeclareLaunchArgument("start_robot", default_value="false")
    rviz_arg = DeclareLaunchArgument("rviz", default_value="false")

    stretch_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
            get_package_share_directory('stretch_core'),
            'launch/stretch_driver.launch.py'))
    )
    
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
            get_package_share_directory('realsense2_camera'),
            'launch/rs_launch.py'))
    )

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
            get_package_share_directory('stretch_core'),
            'launch/rplidar.launch.py'))
    )


    rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("robot_hw_python"),
                'launch/visualization.launch.py'
            )
        )
    )

    camera_pose_publisher = Node(
        package="robot_hw_python",
        executable="camera_pose_publisher",
        name="camera_pose_publisher"
    )

    return LaunchDescription([
        # start_robot_arg,
        stretch_driver_launch,
        realsense_launch,
        # lidar_launch,
        camera_pose_publisher
    ])