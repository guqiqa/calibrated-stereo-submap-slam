#!/usr/bin/env python3
"""Run repeatable C0/C1/C4/P submap experiments against a running Gazebo instance.

Use run_experiment_profile.py for the complete campaign workflow, including
Gazebo startup and service readiness checks.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
import time

import rclpy
from gazebo_msgs.msg import EntityState, ModelState
from gazebo_msgs.srv import SetEntityState, SetModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String
from std_srvs.srv import Empty

from waypoint_route_controller import WaypointPlan, load_waypoints, yaw_from_quaternion


KEY_PARAMETERS = {
    "image_size": "640x480",
    "image_rate_hz": 15,
    "baseline_m": 0.07,
    "processing_scale": 0.75,
    "stereo_reprojection_error_px": 1.5,
    "submap_length_m": 5.0,
    "submap_overlap_m": 1.5,
    "candidate_count": 3,
    "visual_candidate_count": 4,
    "pose_candidate_count": 3,
    "topology_candidate_count": 2,
    "total_candidate_count": 10,
    "representative_keyframes": 3,
    "local_window_keyframes": 15,
    "keyframe_distance_m": 0.3,
    "keyframe_angle_deg": 5.0,
    "switch_confirm_frames": 3,
    "switch_min_matches": 30,
    "switch_min_inliers": 20,
    "switch_max_reprojection_px": 3.0,
    "smooth_transition_frames": 5,
    "map_fusion_radius_m": 0.10,
}


class ResourceMonitor:
    """Sample aggregate CPU and RSS for the VO launch process group."""

    def __init__(self, process, output_path):
        self.process = process
        self.output_path = Path(output_path)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=3.0)

    def run(self):
        with self.output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["host_time", "rss_kb", "cpu_percent"])
            while not self.stop_event.is_set() and self.process.poll() is None:
                result = subprocess.run(
                    ["ps", "-o", "rss=,%cpu=", "-g", str(self.process.pid), "--no-headers"],
                    check=False, capture_output=True, text=True,
                )
                rss_kb = 0
                cpu_percent = 0.0
                for line in result.stdout.splitlines():
                    fields = line.split()
                    if len(fields) >= 2:
                        try:
                            rss_kb += int(fields[0])
                            cpu_percent += float(fields[1])
                        except ValueError:
                            continue
                writer.writerow([time.time(), rss_kb, cpu_percent])
                handle.flush()
                self.stop_event.wait(1.0)


def stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExperimentRecorder(Node):
    def __init__(self, output_dir, speed, angular, waypoint_route=None, submap_truth=None):
        super().__init__("submap_experiment_recorder")
        self.output_dir = Path(output_dir)
        self.speed = speed
        self.angular = angular
        self.direction = 1.0
        self.active = False
        self.waypoint_route = waypoint_route
        self.waypoint_plan = None
        self.latest_truth_pose = None
        self.submap_regions = self.load_submap_regions(submap_truth)
        self.route_labels = self.load_route_labels(waypoint_route)
        self.expected_mode = None
        self.statuses = []
        self.truth_positions = []
        self.visual_positions = []
        self.left_image_stamps = []
        self.right_image_stamps = []
        self.status_file = None
        self.odom_file = None
        self.truth_file = None
        self.control_file = None
        self.cmd_publisher = self.create_publisher(Twist, "/sim/cmd_vel", 10)
        qos = QoSProfile(depth=20)
        self.create_subscription(String, "/visual_slam/status_sim", self.status_callback, qos)
        self.create_subscription(Odometry, "/visual_slam/odometry_sim", self.odom_callback, qos)
        self.create_subscription(Odometry, "/sim/odom", self.truth_callback, qos)
        self.create_subscription(Image, "/sim/left/image_raw", self.left_image_callback, qos)
        self.create_subscription(Image, "/sim/right/image_raw", self.right_image_callback, qos)
        self.create_timer(0.05, self.publish_command)
        self.reset_client = self.create_client(Empty, "/reset_world")
        self.unpause_client = self.create_client(Empty, "/unpause_physics")
        self.set_entity_client = self.create_client(SetEntityState, "/set_entity_state")
        self.set_entity_gazebo_client = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        self.set_model_client = self.create_client(SetModelState, "/gazebo/set_model_state")

    @staticmethod
    def load_submap_regions(path):
        if not path:
            return []
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        regions = []
        for item in payload.get("submaps", []):
            geometry = item.get("geometry", {})
            if geometry.get("type") != "Polygon" or not geometry.get("coordinates"):
                raise ValueError(f"unsupported submap geometry for {item.get('id')}")
            regions.append((item["id"], geometry["coordinates"][0]))
        if not regions:
            raise ValueError(f"no submap regions in {path}")
        return regions

    @staticmethod
    def load_route_labels(path):
        if not path:
            return []
        with Path(path).open(newline="", encoding="utf-8") as handle:
            return [{**row, "x": float(row["x"]), "y": float(row["y"]),
                     "waypoint": int(row["waypoint"])} for row in csv.DictReader(handle)]

    def query_annotation_at_truth(self):
        if not self.route_labels or self.latest_truth_pose is None:
            return {}
        x, y, _ = self.latest_truth_pose
        nearest = min(self.route_labels,
                      key=lambda row: math.hypot(x - row["x"], y - row["y"]))
        return {"query_route": nearest.get("route"),
                "query_waypoint": nearest.get("waypoint"),
                "query_type": nearest.get("query_type"),
                "frame_role": nearest.get("frame_role"),
                "query_waypoint_distance_m": math.hypot(x - nearest["x"], y - nearest["y"])}

    @staticmethod
    def point_in_polygon(x, y, polygon):
        inside = False
        j = len(polygon) - 1
        for i, point in enumerate(polygon):
            xi, yi = point[:2]
            xj, yj = polygon[j][:2]
            if ((yi > y) != (yj > y) and
                    x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi):
                inside = not inside
            j = i
        return inside

    def legal_submaps_at_truth(self):
        if not self.submap_regions or self.latest_truth_pose is None:
            return None
        x, y, _ = self.latest_truth_pose
        return [submap_id for submap_id, polygon in self.submap_regions
                if self.point_in_polygon(x, y, polygon)]

    def open_files(self):
        self.truth_positions.clear()
        self.visual_positions.clear()
        self.left_image_stamps.clear()
        self.right_image_stamps.clear()
        self.status_file = (self.output_dir / "status.jsonl").open("w", encoding="utf-8")
        self.odom_file = (self.output_dir / "visual_odom.csv").open("w", newline="", encoding="utf-8")
        self.truth_file = (self.output_dir / "sim_odom.csv").open("w", newline="", encoding="utf-8")
        self.control_file = (self.output_dir / "control_diagnostics.csv").open(
            "w", newline="", encoding="utf-8")
        self.odom_writer = csv.writer(self.odom_file)
        self.truth_writer = csv.writer(self.truth_file)
        self.control_writer = csv.writer(self.control_file)
        header = ["host_time", "stamp", "x", "y", "z", "qx", "qy", "qz", "qw"]
        self.odom_writer.writerow(header)
        self.truth_writer.writerow(header)
        self.control_writer.writerow([
            "host_time", "x", "y", "yaw", "target_yaw", "yaw_error",
            "waypoint", "phase", "linear_x", "angular_z"])

    def close_files(self):
        for file_handle in (self.status_file, self.odom_file, self.truth_file, self.control_file):
            if file_handle is not None:
                file_handle.close()
        self.status_file = None
        self.odom_file = None
        self.truth_file = None
        self.control_file = None

    def write_diagnostics(self):
        if not self.statuses or self.output_dir is None:
            return
        fields = [
            "host_time", "state", "motion_source", "stereo_points", "valid_matches",
            "ransac_inliers", "pnp_inliers", "median_disparity_px", "depth_min_m",
            "depth_median_m", "stereo_reprojection_px", "motion_consistency",
            "vo_recovery_state", "rotation_2d_inliers", "rotation_2d_deg",
            "rotation_2d_signed_deg", "temporal_metric_inliers",
            "temporal_metric_yaw_deg", "rotation_anchor_used",
            "rotation_anchor_matches", "rotation_anchor_inliers",
            "rotation_anchor_yaw_deg", "rotation_anchor_gap_frames",
            "rotation_anchor_pending_frames", "rotation_anchor_recovery_count",
            "rotation_descriptor_used", "rotation_descriptor_matches",
            "rotation_descriptor_inliers", "rotation_descriptor_yaw_deg",
            "rotation_estimator", "yaw_update_source",
            "temporal_matches_2d", "temporal_flow_px", "rotation_2d_residual_px",
            "temporal_rotation_ok", "pure_rotation_model", "rotation_decision",
            "rotation_episode_yaw_deg", "rotation_last_episode_yaw_deg",
            "rotation_total_yaw_deg", "rotation_updates_total",
            "rotation_wait_frames_total", "rotation_estimation_failure_frames",
            "rotation_residual_rejection_frames", "rotation_subthreshold_frames",
            "rotation_rebootstrap_count", "rotation_settle_count", "rotation_wait_count",
            "map_updates_enabled", "validation_success_frames",
            "visual_distance_m", "processing_ms", "matching_ms", "active_map_points",
            "map_points_stable", "tracking_failures",
            "retrieval_method", "retrieval_decision", "selected_submap",
            "geometry_verified", "reject_reason", "visual_reliability",
            "pose_reliability", "topology_reliability", "q_top1", "q_cov",
            "candidate_covered", "top1_correct", "coverage_label_available",
            "candidate_coverage_reason", "none_of_the_above",
            "query_route", "query_waypoint", "query_type", "frame_role",
            "query_waypoint_distance_m",
            "observability_mode", "visual_state", "visual_measurement_valid",
            "observability_reason", "observability_verdict",
            "observability_frames_since_valid", "r_pre_frame", "r_post_frame",
            "reacquire_confirm_frames", "unobserved_interval_id", "unobserved_frames",
            "unobserved_total_frames", "unobserved_closed_by", "unobserved_unresolved",
            "unobserved_closure_rotation_deg", "relocalization_required",
            "evidence_features", "evidence_inliers", "evidence_inlier_ratio",
            "evidence_reprojection_px", "evidence_view_change_deg",
            "evidence_grid_coverage", "evidence_condition_number",
        ]
        with (self.output_dir / "frame_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.statuses)
        label_fields = ["host_time", "scene_id", "map_version", "retrieval_method",
                        "candidate_ids", "legal_submaps", "candidate_covered",
                        "top1_correct", "coverage_label_available",
                        "candidate_coverage_reason", "none_of_the_above", "q_top1", "q_cov"]
        label_fields.extend(["query_route", "query_waypoint", "query_type", "frame_role",
                             "query_waypoint_distance_m"])
        with (self.output_dir / "candidate_labels.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=label_fields, extrasaction="ignore")
            writer.writeheader()
            for row in self.statuses:
                writer.writerow({key: json.dumps(row.get(key), ensure_ascii=True)
                                 if isinstance(row.get(key), (list, dict)) else row.get(key)
                                 for key in label_fields})
        geometry_fields = ["host_time", "retrieval_method", "selected_scene_submap",
                           "geometry_verified", "reprojection_error_px", "valid_matches",
                           "pnp_inliers", "reject_reason"]
        with (self.output_dir / "geometry_verification.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=geometry_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.statuses)
        with (self.output_dir / "reject_reasons.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["host_time", "retrieval_method", "reject_reason"])
            writer.writeheader()
            for row in self.statuses:
                if row.get("reject_reason"):
                    writer.writerow({key: row.get(key) for key in writer.fieldnames})

    def publish_command(self):
        message = Twist()
        phase = "idle"
        waypoint = None
        target_yaw = None
        yaw_error = None
        if self.active:
            if self.waypoint_plan is not None and self.latest_truth_pose is not None:
                message.linear.x, message.angular.z = self.waypoint_plan.command(*self.latest_truth_pose)
                phase = self.waypoint_plan.phase
                waypoint = self.waypoint_plan.index
                target_yaw = self.waypoint_plan.last_target_yaw
                yaw_error = self.waypoint_plan.last_yaw_error
            else:
                message.linear.x = self.speed * self.direction
                message.angular.z = self.angular * self.direction
        self.cmd_publisher.publish(message)
        if self.control_file is not None and self.latest_truth_pose is not None:
            x, y, yaw = self.latest_truth_pose
            self.control_writer.writerow([
                time.time(), x, y, yaw, target_yaw, yaw_error,
                waypoint, phase, message.linear.x, message.angular.z])

    def send_stop(self):
        message = Twist()
        for _ in range(5):
            self.cmd_publisher.publish(message)
            time.sleep(0.02)

    def status_callback(self, message):
        try:
            data = json.loads(message.data)
        except json.JSONDecodeError:
            data = {"state": "invalid_json", "raw": message.data}
        if self.expected_mode is not None and data.get("submap_mode") != self.expected_mode:
            return
        legal = self.legal_submaps_at_truth()
        data.update(self.query_annotation_at_truth())
        candidates = data.get("candidate_ids") or []
        data["evaluation_eligible"] = data.get("state") == "tracking" and bool(candidates)
        if legal is None:
            data.update(candidate_covered=None, top1_correct=None,
                        coverage_label_available=False,
                        candidate_coverage_reason="truth_pose_or_submap_regions_unavailable")
        else:
            legal_set = set(legal)
            data["legal_submaps"] = legal
            data["none_of_the_above"] = not legal
            data["candidate_covered"] = bool(legal_set.intersection(candidates)) if legal else False
            data["top1_correct"] = bool(candidates and candidates[0] in legal_set) if legal else False
            data["coverage_label_available"] = True
            data["candidate_coverage_reason"] = (
                "none_of_the_above" if not legal else
                "covered" if data["candidate_covered"] else "candidate_coverage_miss")
        data["host_time"] = time.time()
        self.statuses.append(data)
        if self.status_file is not None:
            self.status_file.write(json.dumps(data, ensure_ascii=True) + "\n")
            self.status_file.flush()

    @staticmethod
    def write_odom(writer, message, host_time):
        pose = message.pose.pose
        writer.writerow([
            host_time,
            stamp_seconds(message.header.stamp),
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ])

    def odom_callback(self, message):
        self.visual_positions.append((
            stamp_seconds(message.header.stamp),
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            message.pose.pose.position.z,
        ))
        if self.odom_file is not None:
            self.write_odom(self.odom_writer, message, time.time())

    def truth_callback(self, message):
        received_at = time.time()
        self.truth_positions.append((
            stamp_seconds(message.header.stamp),
            message.pose.pose.position.x,
            message.pose.pose.position.y,
        ))
        self.latest_truth_pose = (
            message.pose.pose.position.x, message.pose.pose.position.y,
            yaw_from_quaternion(message.pose.pose.orientation),
        )
        if self.truth_file is not None:
            self.write_odom(self.truth_writer, message, received_at)

    def left_image_callback(self, message):
        if self.active:
            self.left_image_stamps.append(stamp_seconds(message.header.stamp))

    def right_image_callback(self, message):
        if self.active:
            self.right_image_stamps.append(stamp_seconds(message.header.stamp))

    def reset_simulation(self):
        self.active = False
        self.send_stop()
        if not self.reset_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("/reset_world is not available; start Gazebo first")
        future = self.reset_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done() or future.exception() is not None:
            raise RuntimeError("Gazebo world reset failed")

        # Gazebo Classic can leave physics paused after a reset. Explicitly
        # unpause so the recorded truth and image streams cover the run.
        if not self.unpause_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("/unpause_physics is not available")
        future = self.unpause_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done() or future.exception() is not None:
            raise RuntimeError("Gazebo unpause failed")
        if self.waypoint_plan is not None:
            x, y, yaw = self.waypoint_plan.waypoints[0]
            needs_pose_reset = math.hypot(x, y) > 1e-3 or abs(math.atan2(
                math.sin(yaw), math.cos(yaw))) > 1e-3
            if needs_pose_reset:
                pose_client = self.set_entity_client
                service_kind = "entity"
                if not pose_client.wait_for_service(timeout_sec=2.0):
                    pose_client = self.set_entity_gazebo_client
                    service_kind = "entity"
                if not pose_client.wait_for_service(timeout_sec=2.0):
                    pose_client = self.set_model_client
                    service_kind = "model"
                if not pose_client.wait_for_service(timeout_sec=5.0):
                    raise RuntimeError(
                        "Gazebo pose reset services are unavailable "
                        "for non-default waypoint initialization")
                if service_kind == "model":
                    request = SetModelState.Request()
                    request.model_state = ModelState()
                    request.model_state.model_name = "stereo_robot"
                    request.model_state.reference_frame = "world"
                    pose = request.model_state.pose
                else:
                    request = SetEntityState.Request()
                    request.state = EntityState()
                    request.state.name = "stereo_robot"
                    request.state.reference_frame = "world"
                    pose = request.state.pose
                pose.position.x = x
                pose.position.y = y
                pose.orientation.z = math.sin(yaw / 2.0)
                pose.orientation.w = math.cos(yaw / 2.0)
                future = pose_client.call_async(request)
                rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
                if (not future.done() or future.exception() is not None or
                        not future.result().success):
                    raise RuntimeError("failed to initialize robot at the first waypoint")
        self.send_stop()


def spin_for(node, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)


def stop_process(process, interrupt_timeout=8.0, kill_timeout=3.0):
    """Stop a process group and report whether a forced kill was required."""
    if process is None or process.poll() is not None:
        return False
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return False
    try:
        process.wait(timeout=interrupt_timeout)
        return False
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=kill_timeout)
        return True


def start_vo(mode, log_path, map_export_path=None, initial_pose=None,
             scene_manifest=None, submap_truth=None, candidate_label=None,
             calibration_model=None, map_database=None, map_database_mode="online",
             rotation_metric_pnp_enabled=False):
    log_handle = log_path.open("w", encoding="utf-8")
    launch_args = ["ros2", "launch", "visual_slam", "stereo_vo_sim.launch.py",
                   f"submap_mode:={mode}",
                   f"rotation_metric_pnp_enabled:={str(rotation_metric_pnp_enabled).lower()}"]
    if map_export_path is not None:
        launch_args.append(f"map_export_path:={map_export_path}")
    if scene_manifest and submap_truth:
        launch_args.extend([
            f"scene_manifest_path:={scene_manifest}",
            f"submap_truth_path:={submap_truth}",
            f"candidate_label_path:={candidate_label or ''}",
            "online_submap_id_mode:=scene",
        ])
    if calibration_model:
        launch_args.append(f"calibration_model_path:={calibration_model}")
    if map_database:
        launch_args.extend([f"map_database_path:={map_database}",
                            f"map_database_mode:={map_database_mode}"])
    if initial_pose is not None:
        base_x, base_y, yaw = initial_pose
        camera_x = base_x + math.cos(yaw) * 0.08 - math.sin(yaw) * 0.035
        camera_y = base_y + math.sin(yaw) * 0.08 + math.cos(yaw) * 0.035
        launch_args.extend([
            f"initial_gazebo_x:={camera_x}",
            f"initial_gazebo_y:={camera_y}",
            "initial_gazebo_z:=0.32",
            f"initial_gazebo_yaw:={yaw}",
            "straight_motion_constraint:=false",
        ])
    process = subprocess.Popen(
        launch_args,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log_handle


def start_rosbag(output_dir):
    # Recording raw stereo images directly to /mnt/<drive> can stall rosbag's
    # SQLite writer for several seconds. Record on WSL's native filesystem,
    # then move the closed bag into the experiment directory.
    staging_root = Path(tempfile.mkdtemp(prefix="visual_slam_rosbag_"))
    staging_bag_dir = staging_root / "rosbag"
    log_path = output_dir / "rosbag.log"
    log_handle = log_path.open("w", encoding="utf-8")
    topics = [
        "/sim/left/image_raw", "/sim/right/image_raw",
        "/sim/left/camera_info", "/sim/right/camera_info",
        "/sim/odom", "/visual_slam/odometry_sim", "/visual_slam/status_sim",
    ]
    process = subprocess.Popen(
        ["ros2", "bag", "record", "-o", str(staging_bag_dir), *topics],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return process, log_handle, staging_root, staging_bag_dir


def finalize_rosbag(output_dir, staging_root, staging_bag_dir):
    if staging_bag_dir is None or not staging_bag_dir.exists():
        return
    target = output_dir / "rosbag"
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(staging_bag_dir), str(target))
    shutil.rmtree(staging_root, ignore_errors=True)


def dump_parameters(output_dir, process=None, attempts=5):
    """Persist the effective VO parameters while the node is still available."""
    last_result = None
    for attempt in range(attempts):
        if process is not None and process.poll() is not None:
            break
        try:
            result = subprocess.run(
                ["ros2", "param", "dump", "/stereo_vo_sim"],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except subprocess.TimeoutExpired as exc:
            last_result = subprocess.CompletedProcess(
                args=exc.cmd, returncode=124, stdout=exc.stdout or "",
                stderr="parameter dump timed out"
            )
            result = last_result
        last_result = result
        if result.returncode == 0 and "ros__parameters:" in result.stdout:
            (output_dir / "resolved_parameters.yaml").write_text(
                result.stdout, encoding="utf-8"
            )
            return True
        if attempt + 1 < attempts:
            time.sleep(0.5)

    if last_result is None:
        error = "stereo_vo_sim exited before parameter dump"
    else:
        error = (
            f"ros2 param dump failed: returncode={last_result.returncode}; "
            f"stderr={last_result.stderr.strip()}"
        )
    (output_dir / "resolved_parameters.yaml").write_text(
        f"# {error}\n", encoding="utf-8"
    )
    (output_dir / "resolved_parameters_error.txt").write_text(
        error + "\n", encoding="utf-8"
    )
    return False


def summarize(node, mode, run, duration, speed, angular):
    states = [item.get("state") for item in node.statuses]
    tracking = sum(state == "tracking" for state in states)
    failures = sum(state in {"tracking_lost", "insufficient_stereo_features"} for state in states)
    last = node.statuses[-1] if node.statuses else {}
    truth_x = [item[1] for item in node.truth_positions]
    truth_y = [item[2] for item in node.truth_positions]
    visual_positions = node.visual_positions
    def path_length(points):
        return sum(
            ((points[i][1] - points[i - 1][1]) ** 2 +
             (points[i][2] - points[i - 1][2]) ** 2 +
             (points[i][3] - points[i - 1][3]) ** 2) ** 0.5
            for i in range(1, len(points))
        )
    truth_distance = 0.0
    if truth_x:
        truth_distance = ((max(truth_x) - min(truth_x)) ** 2 +
                          (max(truth_y) - min(truth_y)) ** 2) ** 0.5
    visual_distance = float(last.get("visual_distance_m", 0.0) or 0.0)
    truth_path_distance = path_length([
        (index, x, y, 0.0) for index, (x, y) in enumerate(zip(truth_x, truth_y))
    ])
    visual_path_distance = path_length(visual_positions)
    visual_closure_error = None
    truth_closure_error = None
    if len(visual_positions) >= 2:
        visual_closure_error = ((visual_positions[-1][1] - visual_positions[0][1]) ** 2 +
                                (visual_positions[-1][2] - visual_positions[0][2]) ** 2 +
                                (visual_positions[-1][3] - visual_positions[0][3]) ** 2) ** 0.5
    if len(node.truth_positions) >= 2:
        truth_closure_error = ((node.truth_positions[-1][1] - node.truth_positions[0][1]) ** 2 +
                               (node.truth_positions[-1][2] - node.truth_positions[0][2]) ** 2) ** 0.5
    eligible = [item for item in node.statuses if item.get("evaluation_eligible")]
    evaluable = [item for item in eligible if item.get("coverage_label_available")]
    def stream_stats(stamps):
        gaps = [right - left for left, right in zip(stamps, stamps[1:]) if right > left]
        elapsed = stamps[-1] - stamps[0] if len(stamps) > 1 else 0.0
        return {"samples": len(stamps), "rate_hz": ((len(stamps) - 1) / elapsed if elapsed > 0 else 0.0),
                "max_gap_s": max(gaps, default=None)}
    left_stats = stream_stats(node.left_image_stamps)
    right_stats = stream_stats(node.right_image_stamps)
    return {
        "mode": mode,
        "run": run,
        "duration_s": duration,
        "speed_mps": speed,
        "angular_radps": angular,
        "status_messages": len(states),
        "tracking_messages": tracking,
        "failure_messages": failures,
        "last_state": last.get("state"),
        "last_active_submap": last.get("active_submap"),
        "last_active_map_points": last.get("active_map_points"),
        "last_matching_ms": last.get("matching_ms"),
        "switch_count": last.get("switch_count", 0),
        "wrong_switch_count": last.get("wrong_switch_count", 0),
        "truth_samples": len(node.truth_positions),
        "truth_start_x": truth_x[0] if truth_x else None,
        "truth_end_x": truth_x[-1] if truth_x else None,
        "truth_distance_m": truth_distance,
        "visual_distance_m": visual_distance,
        "truth_path_distance_m": truth_path_distance,
        "visual_path_distance_m": visual_path_distance,
        "visual_truth_distance_ratio": (visual_path_distance / truth_path_distance
                                         if truth_path_distance > 1e-6 else None),
        "visual_closure_error_m": visual_closure_error,
        "truth_closure_error_m": truth_closure_error,
        "max_consecutive_tracking_failures": max_consecutive_failures(states),
        "loop_closure_events": sum(item.get("loop_closure_accepted", 0) == 1 for item in node.statuses),
        "loop_closure_corrections": last.get("loop_closure_count", 0),
        "coverage_label_available_fraction": len(evaluable) / len(eligible) if eligible else 0.0,
        "candidate_coverage_rate": (sum(bool(item.get("candidate_covered")) for item in evaluable) /
                                    len(evaluable) if evaluable else None),
        "q_top1_valid_fraction": (sum(item.get("q_top1") is not None for item in evaluable) /
                                  len(evaluable) if evaluable else 0.0),
        "q_cov_valid_fraction": (sum(item.get("q_cov") is not None for item in evaluable) /
                                len(evaluable) if evaluable else 0.0),
        "left_image": left_stats,
        "right_image": right_stats,
        "stereo_sample_count_difference": abs(left_stats["samples"] - right_stats["samples"]),
    }


def max_consecutive_failures(states):
    longest = current = 0
    for state in states:
        if state in {"tracking_lost", "insufficient_stereo_features"}:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def run_one(node, mode, run, args, root):
    output_dir = root / mode / f"run_{run:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    node.statuses.clear()
    node.left_image_stamps.clear()
    node.right_image_stamps.clear()
    node.expected_mode = mode
    node.direction = 1.0
    node.waypoint_plan = (WaypointPlan(
        load_waypoints(args.waypoint_route), args.speed,
        max_angular=args.max_turn_angular)
                          if args.waypoint_route else None)
    node.output_dir = output_dir
    node.open_files()
    node.reset_simulation()
    spin_for(node, 1.0)
    if node.truth_positions:
        start_x = node.truth_positions[-1][1]
        start_y = node.truth_positions[-1][2]
        expected_x, expected_y = ((node.waypoint_plan.waypoints[0][0],
                                   node.waypoint_plan.waypoints[0][1])
                                  if node.waypoint_plan else (0.0, 0.0))
        if math.hypot(start_x - expected_x, start_y - expected_y) > 0.5:
            raise RuntimeError(
                f"world reset did not initialize the expected route start: "
                f"expected=({expected_x:.3f},{expected_y:.3f}), "
                f"actual=({start_x:.3f},{start_y:.3f})"
            )
    # Discard any latched sample from before the reset. All run statistics
    # must describe motion after the confirmed origin check.
    node.truth_positions.clear()
    node.visual_positions.clear()
    map_export_path = output_dir / "map_points.csv" if args.export_map else None
    initial_pose = node.waypoint_plan.waypoints[0] if node.waypoint_plan else None
    process, log_handle = start_vo(
        mode, output_dir / "stereo_vo_sim.log", map_export_path, initial_pose,
        args.scene_manifest, args.submap_truth, args.candidate_label,
        args.calibration_model, args.map_database, args.map_database_mode,
        args.rotation_metric_pnp)
    resource_monitor = ResourceMonitor(process, output_dir / "resource_usage.csv")
    resource_monitor.start()
    bag_process = None
    bag_log_handle = None
    bag_staging_root = None
    bag_staging_dir = None
    try:
        spin_for(node, args.warmup)
        if args.record_bag:
            bag_process, bag_log_handle, bag_staging_root, bag_staging_dir = start_rosbag(output_dir)
            spin_for(node, 1.0)
        node.active = True
        if args.waypoint_route:
            deadline = time.monotonic() + args.duration
            while time.monotonic() < deadline and not node.waypoint_plan.completed:
                spin_for(node, min(0.25, deadline - time.monotonic()))
        elif args.reverse_after is None:
            spin_for(node, args.duration)
        else:
            if args.reverse_after <= 0.0 or args.reverse_after >= args.duration:
                raise RuntimeError("reverse-after must be between 0 and duration")
            spin_for(node, args.reverse_after)
            node.direction = -1.0
            spin_for(node, args.duration - args.reverse_after)
        node.active = False
        spin_for(node, 1.0)
        expected_distance = abs(args.speed) * args.duration
        run_summary = summarize(node, mode, run, args.duration, args.speed, args.angular)
        actual_distance = run_summary["truth_path_distance_m"]
        if not args.waypoint_route and expected_distance > 0.0 and actual_distance < expected_distance * 0.5:
            node.get_logger().warning(
                f"truth motion is short: expected about {expected_distance:.2f} m, "
                f"observed {actual_distance:.2f} m"
            )
        dump_parameters(output_dir, process=process)
        metadata = {
            "mode": mode,
            "run": run,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "key_parameters": KEY_PARAMETERS,
            "topics": {
                "left": "/sim/left/image_raw",
                "right": "/sim/right/image_raw",
                "visual_odom": "/visual_slam/odometry_sim",
                "ground_truth": "/sim/odom",
                "status": "/visual_slam/status_sim",
            },
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        run_manifest = {
            "project": os.environ.get("EXPERIMENT_PROJECT"),
            "scene": os.environ.get("EXPERIMENT_SCENE"),
            "route_m": float(os.environ["EXPERIMENT_ROUTE_M"])
            if os.environ.get("EXPERIMENT_ROUTE_M") else None,
            "git_commit": os.environ.get("EXPERIMENT_GIT_COMMIT"),
            "method": mode,
            "run": run,
            "duration_s": args.duration,
            "command_speed_mps": args.speed,
            "max_turn_angular_radps": args.max_turn_angular,
            "camera": {"width": 640, "height": 480, "rate_hz": 15, "baseline_m": 0.07},
            "submap": KEY_PARAMETERS,
            "calibration_version": os.environ.get(
                "EXPERIMENT_CALIBRATION_VERSION",
                node.statuses[-1].get("calibration_version", "unfitted-v0")
                if node.statuses else "unfitted-v0"),
            "scene_manifest": os.environ.get("EXPERIMENT_SCENE_MANIFEST"),
            "scene_map_version": os.environ.get("EXPERIMENT_SCENE_MAP_VERSION"),
            "waypoint_route": str(Path(args.waypoint_route).resolve()) if args.waypoint_route else None,
            "map_database": str(Path(args.map_database).resolve()) if args.map_database else None,
            "map_database_mode": args.map_database_mode,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        (output_dir / "experiment_manifest.json").write_text(
            json.dumps(run_manifest, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        summary = run_summary
        summary["waypoint_route"] = str(args.waypoint_route) if args.waypoint_route else None
        summary["waypoint_completed"] = node.waypoint_plan.completed if node.waypoint_plan else None
        summary["waypoints_reached"] = node.waypoint_plan.index if node.waypoint_plan else None
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=True))
    finally:
        node.active = False
        node.send_stop()
        # Image bags can take tens of seconds to flush on a mounted Windows
        # volume. Killing rosbag during that flush corrupts its SQLite index.
        bag_forced = stop_process(bag_process, interrupt_timeout=120.0, kill_timeout=5.0)
        stop_process(process)
        resource_monitor.stop()
        log_handle.close()
        if bag_log_handle is not None:
            bag_log_handle.close()
        finalize_rosbag(output_dir, bag_staging_root, bag_staging_dir)
        if args.map_database and Path(args.map_database).exists():
            database_manifest = {
                "path": str(Path(args.map_database).resolve()),
                "mode": args.map_database_mode,
                "sha256": file_sha256(args.map_database),
                "size": Path(args.map_database).stat().st_size,
            }
            (output_dir / "map_database_manifest.json").write_text(
                json.dumps(database_manifest, indent=2), encoding="utf-8")
        if bag_forced:
            (output_dir / "rosbag_forced_termination.txt").write_text(
                "rosbag did not finish flushing within 120 seconds and was forcefully terminated.\n",
                encoding="utf-8",
            )
        node.write_diagnostics()
        node.close_files()
        required = ["status.jsonl", "visual_odom.csv", "sim_odom.csv",
                    "frame_diagnostics.csv", "candidate_labels.csv",
                    "geometry_verification.csv", "reject_reasons.csv",
                    "resource_usage.csv", "summary.json", "experiment_manifest.json"]
        missing = [name for name in required
                   if not (output_dir / name).exists() or (output_dir / name).stat().st_size == 0]
        if args.record_bag and not (output_dir / "rosbag" / "metadata.yaml").exists():
            missing.append("rosbag/metadata.yaml")
        quality = {
            "run_state": "complete" if not missing else "incomplete",
            "missing_or_empty": missing,
            "status_messages": len(node.statuses),
            "left_image_samples": len(node.left_image_stamps),
            "right_image_samples": len(node.right_image_stamps),
            "single_gazebo_instance_checked_by_manager": True,
        }
        (output_dir / "data_quality_run.json").write_text(
            json.dumps(quality, indent=2), encoding="utf-8")
        node.expected_mode = None
        truth_path = output_dir / "sim_odom.csv"
        if truth_path.exists():
            shutil.copyfile(truth_path, output_dir / "trajectory_truth.csv")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+",
                        choices=["c0", "c1", "c4", "p", "global", "fixed", "joint"],
                        default=["c0", "c1", "c4", "p"])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--warmup", type=float, default=3.0)
    parser.add_argument("--speed", type=float, default=0.2)
    parser.add_argument("--max-turn-angular", type=float, default=0.22,
                        help="maximum angular velocity for waypoint turns in rad/s")
    parser.add_argument("--angular", type=float, default=0.0)
    parser.add_argument("--output", default="experiments/submap")
    parser.add_argument("--export-map", action="store_true",
                        help="export runtime global/submap 3D points to map_points.csv")
    parser.add_argument("--record-bag", action="store_true",
                        help="record stereo, truth, odometry and status topics for each run")
    parser.add_argument("--reverse-after", type=float, default=None,
                        help="reverse the commanded route after this many seconds")
    parser.add_argument("--waypoint-route", default=None,
                        help="Generated route CSV; /sim/odom is used only by the motion controller")
    parser.add_argument("--scene-manifest")
    parser.add_argument("--submap-truth")
    parser.add_argument("--candidate-label")
    parser.add_argument("--calibration-model")
    parser.add_argument("--map-database")
    parser.add_argument("--map-database-mode", choices=("online", "build", "load"), default="online")
    parser.add_argument("--rotation-metric-pnp", action="store_true",
                        help="Use guarded stereo-PnP yaw during rotation-only recovery")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.runs < 1 or args.duration <= 0.0 or args.warmup < 0.0:
        raise SystemExit("runs must be >= 1, duration must be > 0, warmup must be >= 0")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = ExperimentRecorder(root, args.speed, args.angular, args.waypoint_route,
                              args.submap_truth)
    try:
        for mode in args.modes:
            for run in range(1, args.runs + 1):
                run_one(node, mode, run, args, root)
    finally:
        node.active = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
