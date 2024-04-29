import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import TextSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    stretch_navigation_path = get_package_share_directory("stretch_nav2")
    start_robot_arg = DeclareLaunchArgument("start_robot", default_value="false")
    rviz_arg = DeclareLaunchArgument("rviz", default_value="false")

    declare_use_sim_time_argument = DeclareLaunchArgument(
        "use_sim_time", default_value="false", description="Use simulation/Gazebo clock"
    )

    declare_slam_params_file_cmd = DeclareLaunchArgument(
        "slam_params_file",
        default_value=os.path.join(
            stretch_navigation_path, "config", "mapper_params_online_async.yaml"
        ),
        description="Full path to the ROS2 parameters file to use for the slam_toolbox node",
    )

    stretch_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("stretch_core"), "launch/stretch_driver.launch.py"
            )
        ),
        launch_arguments={"mode": "navigation", "broadcast_odom_tf": "True"}.items(),
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("realsense2_camera"), "launch/rs_launch.py")
        )
    )

    lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("stretch_core"), "launch/rplidar.launch.py")
        )
    )

    offline_mapping_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [get_package_share_directory("slam_toolbox"), "/launch/offline_launch.py"]
        )
    )

    # rviz_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(
    #             get_package_share_directory("robot_hw_python"),
    #             'launch/visualization.launch.py'
    #         )
    #     )
    # )

    # nav2_offline_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(
    #             get_package_share_directory("stretch_nav2"), "launch/offline_mapping.launch.py"
    #         )
    #     )
    # )

    camera_pose_publisher_node = Node(
        package="robot_hw_python", executable="camera_pose_publisher", name="camera_pose_publisher"
    )

    state_estimator_node = Node(
        package="robot_hw_python", executable="state_estimator", name="state_estimator"
    )

    goto_controller_node = Node(
        package="robot_hw_python", executable="goto_controller", name="goto_controller"
    )

    return LaunchDescription(
        [
            # start_robot_arg,
            stretch_driver_launch,
            offline_mapping_launch,
            realsense_launch,
            lidar_launch,
            camera_pose_publisher_node,
            state_estimator_node,
            goto_controller_node,
            declare_use_sim_time_argument,
            declare_slam_params_file_cmd,
        ]
    )