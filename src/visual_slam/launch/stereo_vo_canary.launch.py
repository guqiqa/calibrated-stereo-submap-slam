import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("visual_slam")
    default_params = os.path.join(share, "config", "stereo_vo_canary.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_params),
        Node(
            package="visual_slam",
            executable="stereo_vo_canary",
            name="stereo_vo_canary",
            parameters=[LaunchConfiguration("params_file")],
            output="screen",
            emulate_tty=True,
        ),
    ])
