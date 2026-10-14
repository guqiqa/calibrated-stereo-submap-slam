import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("visual_slam")
    default_params = os.path.join(share, "config", "stereo_vo_sim.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument("submap_mode", default_value="joint"),
        DeclareLaunchArgument("map_export_path", default_value=""),
        DeclareLaunchArgument("initial_gazebo_x", default_value="0.08"),
        DeclareLaunchArgument("initial_gazebo_y", default_value="0.035"),
        DeclareLaunchArgument("initial_gazebo_z", default_value="0.32"),
        DeclareLaunchArgument("initial_gazebo_yaw", default_value="0.0"),
        DeclareLaunchArgument("straight_motion_constraint", default_value="true"),
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
                    "initial_gazebo_x": ParameterValue(
                        LaunchConfiguration("initial_gazebo_x"), value_type=float),
                    "initial_gazebo_y": ParameterValue(
                        LaunchConfiguration("initial_gazebo_y"), value_type=float),
                    "initial_gazebo_z": ParameterValue(
                        LaunchConfiguration("initial_gazebo_z"), value_type=float),
                    "initial_gazebo_yaw": ParameterValue(
                        LaunchConfiguration("initial_gazebo_yaw"), value_type=float),
                    "straight_motion_constraint": ParameterValue(
                        LaunchConfiguration("straight_motion_constraint"), value_type=bool),
                },
            ],
            output="screen",
            emulate_tty=True,
        ),
    ])
