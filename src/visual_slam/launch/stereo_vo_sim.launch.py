import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("visual_slam")
    default_params = os.path.join(share, "config", "stereo_vo_sim.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument("submap_mode", default_value="joint"),
        DeclareLaunchArgument("map_export_path", default_value=""),
        Node(
            package="visual_slam",
            executable="stereo_vo_sim",
            name="stereo_vo_sim",
            parameters=[
                LaunchConfiguration("params_file"),
                {
                    "use_sim_time": True,
                    "submap_mode": LaunchConfiguration("submap_mode"),
                    "map_export_path": LaunchConfiguration("map_export_path"),
                },
            ],
            output="screen",
            emulate_tty=True,
        ),
    ])
