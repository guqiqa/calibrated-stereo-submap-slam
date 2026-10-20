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
        DeclareLaunchArgument("map_frame_x", default_value="0.08"),
        DeclareLaunchArgument("map_frame_y", default_value="0.035"),
        DeclareLaunchArgument("map_frame_z", default_value="0.32"),
        DeclareLaunchArgument("map_frame_yaw", default_value="0.0"),
        DeclareLaunchArgument("straight_motion_constraint", default_value="true"),
        DeclareLaunchArgument("observability_mode", default_value="b0_plain"),
        DeclareLaunchArgument("rotation_metric_pnp_enabled", default_value="false"),
        DeclareLaunchArgument("scene_manifest_path", default_value=""),
        DeclareLaunchArgument("submap_truth_path", default_value=""),
        DeclareLaunchArgument("candidate_label_path", default_value=""),
        DeclareLaunchArgument("online_submap_id_mode", default_value="distance"),
        DeclareLaunchArgument("calibration_model_path", default_value=""),
        DeclareLaunchArgument("map_database_path", default_value=""),
        DeclareLaunchArgument("map_database_mode", default_value="online"),
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
                    "map_frame_x": ParameterValue(
                        LaunchConfiguration("map_frame_x"), value_type=float),
                    "map_frame_y": ParameterValue(
                        LaunchConfiguration("map_frame_y"), value_type=float),
                    "map_frame_z": ParameterValue(
                        LaunchConfiguration("map_frame_z"), value_type=float),
                    "map_frame_yaw": ParameterValue(
                        LaunchConfiguration("map_frame_yaw"), value_type=float),
                    "straight_motion_constraint": ParameterValue(
                        LaunchConfiguration("straight_motion_constraint"), value_type=bool),
                    "observability_mode": LaunchConfiguration("observability_mode"),
                    "rotation_metric_pnp_enabled": ParameterValue(
                        LaunchConfiguration("rotation_metric_pnp_enabled"), value_type=bool),
                    "scene_manifest_path": LaunchConfiguration("scene_manifest_path"),
                    "submap_truth_path": LaunchConfiguration("submap_truth_path"),
                    "candidate_label_path": LaunchConfiguration("candidate_label_path"),
                    "online_submap_id_mode": LaunchConfiguration("online_submap_id_mode"),
                    "calibration_model_path": LaunchConfiguration("calibration_model_path"),
                    "map_database_path": LaunchConfiguration("map_database_path"),
                    "map_database_mode": LaunchConfiguration("map_database_mode"),
                },
            ],
            output="screen",
            emulate_tty=True,
        ),
    ])
