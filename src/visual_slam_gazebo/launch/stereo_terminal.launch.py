"""Launch a minimal indoor Gazebo scene with a stereo camera robot."""

import os

from ament_index_python.packages import get_package_share_directory
from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("visual_slam_gazebo")
    gazebo_share = get_package_share_directory("gazebo_ros")
    gazebo_prefix = get_package_prefix("gazebo_ros")

    world = os.path.join(package_share, "worlds", "terminal_basic.world")
    robot_urdf = os.path.join(package_share, "urdf", "stereo_robot.urdf")
    gazebo_plugin_path = os.path.join(gazebo_prefix, "lib")
    gazebo_model_path = os.path.join(gazebo_prefix, "share", "gazebo-11", "models")
    custom_model_path = os.path.join(package_share, "models")

    gui = LaunchConfiguration("gui")
    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription([
        SetEnvironmentVariable(
            name="GAZEBO_PLUGIN_PATH",
            value=gazebo_plugin_path,
        ),
        SetEnvironmentVariable(
            name="GAZEBO_MODEL_PATH",
            value=os.pathsep.join([custom_model_path, gazebo_model_path]),
        ),
        DeclareLaunchArgument(
            "gui",
            default_value="true",
            description="Start the Gazebo client GUI",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use Gazebo simulation time",
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_share, "launch", "gazebo.launch.py")
            ),
            launch_arguments={
                "world": world,
                "gui": gui,
                "verbose": "true",
                "factory": "false",
            }.items(),
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="stereo_robot_state_publisher",
            parameters=[{
                "robot_description": open(robot_urdf, encoding="utf-8").read(),
                "use_sim_time": use_sim_time,
            }],
            output="screen",
        ),
    ])
