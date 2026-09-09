#!/usr/bin/env python3
"""Run repeatable G/S/P submap experiments against a running Gazebo instance.

Use run_experiment_profile.py for the complete campaign workflow, including
Gazebo startup and service readiness checks.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String
from std_srvs.srv import Empty


KEY_PARAMETERS = {
    "image_size": "640x480",
    "image_rate_hz": 15,
    "baseline_m": 0.07,
    "processing_scale": 0.75,
    "stereo_reprojection_error_px": 1.5,
    "submap_length_m": 5.0,
    "submap_overlap_m": 1.5,
    "candidate_count": 3,
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


def stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class ExperimentRecorder(Node):
    def __init__(self, output_dir, speed, angular):
        super().__init__("submap_experiment_recorder")
        self.output_dir = Path(output_dir)
        self.speed = speed
        self.angular = angular
        self.direction = 1.0
        self.active = False
        self.statuses = []
        self.truth_positions = []
        self.visual_positions = []
        self.status_file = None
        self.odom_file = None
        self.truth_file = None
        self.cmd_publisher = self.create_publisher(Twist, "/sim/cmd_vel", 10)
        qos = QoSProfile(depth=20)
        self.create_subscription(String, "/visual_slam/status_sim", self.status_callback, qos)
        self.create_subscription(Odometry, "/visual_slam/odometry_sim", self.odom_callback, qos)
        self.create_subscription(Odometry, "/sim/odom", self.truth_callback, qos)
        self.create_timer(0.05, self.publish_command)
        self.reset_client = self.create_client(Empty, "/reset_world")
        self.unpause_client = self.create_client(Empty, "/unpause_physics")

    def open_files(self):
        self.truth_positions.clear()
        self.visual_positions.clear()
        self.status_file = (self.output_dir / "status.jsonl").open("w", encoding="utf-8")
        self.odom_file = (self.output_dir / "visual_odom.csv").open("w", newline="", encoding="utf-8")
        self.truth_file = (self.output_dir / "sim_odom.csv").open("w", newline="", encoding="utf-8")
        self.odom_writer = csv.writer(self.odom_file)
        self.truth_writer = csv.writer(self.truth_file)
        header = ["host_time", "stamp", "x", "y", "z", "qx", "qy", "qz", "qw"]
        self.odom_writer.writerow(header)
        self.truth_writer.writerow(header)

    def close_files(self):
        for file_handle in (self.status_file, self.odom_file, self.truth_file):
            if file_handle is not None:
                file_handle.close()

    def write_diagnostics(self):
        if not self.statuses or self.output_dir is None:
            return
        fields = [
            "host_time", "state", "motion_source", "stereo_points", "valid_matches",
            "ransac_inliers", "pnp_inliers", "median_disparity_px", "depth_min_m",
            "depth_median_m", "stereo_reprojection_px", "motion_consistency",
            "visual_distance_m", "processing_ms", "matching_ms", "active_map_points",
            "map_points_stable", "tracking_failures",
        ]
        with (self.output_dir / "frame_diagnostics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.statuses)

    def publish_command(self):
        message = Twist()
        if self.active:
            message.linear.x = self.speed * self.direction
            message.angular.z = self.angular * self.direction
        self.cmd_publisher.publish(message)

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
        if self.truth_file is not None:
            self.write_odom(self.truth_writer, message, received_at)

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
        self.send_stop()


def spin_for(node, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def start_vo(mode, log_path, map_export_path=None):
    log_handle = log_path.open("w", encoding="utf-8")
    launch_args = ["ros2", "launch", "visual_slam", "stereo_vo_sim.launch.py", f"submap_mode:={mode}"]
    if map_export_path is not None:
        launch_args.append(f"map_export_path:={map_export_path}")
    process = subprocess.Popen(
        launch_args,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log_handle


def start_rosbag(output_dir):
    bag_dir = output_dir / "rosbag"
    log_path = output_dir / "rosbag.log"
    log_handle = log_path.open("w", encoding="utf-8")
    topics = [
        "/sim/left/image_raw", "/sim/right/image_raw",
        "/sim/left/camera_info", "/sim/right/camera_info",
        "/sim/odom", "/visual_slam/odometry_sim", "/visual_slam/status_sim",
    ]
    process = subprocess.Popen(
        ["ros2", "bag", "record", "-o", str(bag_dir), *topics],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return process, log_handle


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
    node.direction = 1.0
    node.output_dir = output_dir
    node.open_files()
    node.reset_simulation()
    spin_for(node, 1.0)
    if node.truth_positions:
        start_x = node.truth_positions[-1][1]
        start_y = node.truth_positions[-1][2]
        if abs(start_x) > 0.5 or abs(start_y) > 0.5:
            raise RuntimeError(
                f"world reset did not return robot to origin: x={start_x:.3f}, y={start_y:.3f}"
            )
    # Discard any latched sample from before the reset. All run statistics
    # must describe motion after the confirmed origin check.
    node.truth_positions.clear()
    node.visual_positions.clear()
    map_export_path = output_dir / "map_points.csv" if args.export_map else None
    process, log_handle = start_vo(mode, output_dir / "stereo_vo_sim.log", map_export_path)
    bag_process = None
    bag_log_handle = None
    try:
        spin_for(node, args.warmup)
        if args.record_bag:
            bag_process, bag_log_handle = start_rosbag(output_dir)
            spin_for(node, 1.0)
        node.active = True
        if args.reverse_after is None:
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
        if expected_distance > 0.0 and actual_distance < expected_distance * 0.5:
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
        summary = run_summary
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=True))
    finally:
        node.active = False
        node.send_stop()
        stop_process(bag_process)
        stop_process(process)
        log_handle.close()
        if bag_log_handle is not None:
            bag_log_handle.close()
        node.write_diagnostics()
        node.close_files()
        truth_path = output_dir / "sim_odom.csv"
        if truth_path.exists():
            shutil.copyfile(truth_path, output_dir / "trajectory_truth.csv")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=["global", "fixed", "joint"],
                        default=["global", "fixed", "joint"])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--warmup", type=float, default=3.0)
    parser.add_argument("--speed", type=float, default=0.2)
    parser.add_argument("--angular", type=float, default=0.0)
    parser.add_argument("--output", default="experiments/submap")
    parser.add_argument("--export-map", action="store_true",
                        help="export runtime global/submap 3D points to map_points.csv")
    parser.add_argument("--record-bag", action="store_true",
                        help="record stereo, truth, odometry and status topics for each run")
    parser.add_argument("--reverse-after", type=float, default=None,
                        help="reverse the commanded route after this many seconds")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.runs < 1 or args.duration <= 0.0 or args.warmup < 0.0:
        raise SystemExit("runs must be >= 1, duration must be > 0, warmup must be >= 0")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = ExperimentRecorder(root, args.speed, args.angular)
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
