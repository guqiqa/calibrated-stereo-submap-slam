#!/usr/bin/env python3
"""Calibrate /sim/cmd_vel linear.x against measured /sim/odom motion."""

import argparse
import json
from pathlib import Path
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Empty


class SpeedCalibrator(Node):
    def __init__(self):
        super().__init__("sim_speed_calibrator")
        self.publisher = self.create_publisher(Twist, "/sim/cmd_vel", 10)
        self.create_subscription(
            Odometry, "/sim/odom", self.odom_callback, QoSProfile(depth=50)
        )
        self.reset_client = self.create_client(Empty, "/reset_world")
        self.unpause_client = self.create_client(Empty, "/unpause_physics")
        self.command_speed = 0.0
        self.active = False
        self.samples = []
        self.create_timer(0.05, self.publish_command)

    def publish_command(self):
        message = Twist()
        if self.active:
            message.linear.x = self.command_speed
        self.publisher.publish(message)

    def odom_callback(self, message):
        if self.active:
            self.samples.append((
                time.monotonic(),
                float(message.pose.pose.position.x),
                float(message.pose.pose.position.y),
            ))

    def send_stop(self):
        self.active = False
        message = Twist()
        for _ in range(8):
            self.publisher.publish(message)
            time.sleep(0.02)

    def call_service(self, client, name):
        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"{name} is not available; start Gazebo first")
        future = client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done() or future.exception() is not None:
            raise RuntimeError(f"{name} failed")

    def reset(self):
        self.send_stop()
        self.call_service(self.reset_client, "/reset_world")
        self.call_service(self.unpause_client, "/unpause_physics")
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

    def measure(self, command_speed, duration):
        self.reset()
        self.command_speed = command_speed
        self.samples.clear()
        self.active = True
        start = time.monotonic()
        end = start + duration
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.send_stop()
        if len(self.samples) < 2:
            raise RuntimeError(f"not enough odometry samples for command {command_speed}")
        first = self.samples[0]
        last = self.samples[-1]
        elapsed = max(last[0] - first[0], 1e-6)
        distance = ((last[1] - first[1]) ** 2 + (last[2] - first[2]) ** 2) ** 0.5
        return {
            "command_speed_mps": command_speed,
            "actual_speed_mps": distance / elapsed,
            "distance_m": distance,
            "elapsed_s": elapsed,
            "samples": len(self.samples),
            "start_x": first[1],
            "end_x": last[1],
        }


def fit_line(results):
    xs = [item["command_speed_mps"] for item in results]
    ys = [item["actual_speed_mps"] for item in results]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    intercept = y_mean - slope * x_mean
    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    ss_res = sum(value ** 2 for value in residuals)
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return slope, intercept, r_squared


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commands", nargs="+", type=float,
                        default=[0.20, 0.25, 0.275, 0.30])
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--target", type=float, default=0.20)
    parser.add_argument("--output", default="experiments/speed_calibration_20260905.json")
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.commands) < 2 or args.duration <= 0.0:
        raise SystemExit("provide at least two command speeds and a positive duration")
    rclpy.init()
    node = SpeedCalibrator()
    try:
        results = []
        for command_speed in args.commands:
            result = node.measure(command_speed, args.duration)
            results.append(result)
            print(json.dumps(result, ensure_ascii=True), flush=True)
        slope, intercept, r_squared = fit_line(results)
        if slope <= 0.0:
            raise RuntimeError("calibration slope is not positive")
        recommended = (args.target - intercept) / slope
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "target_actual_speed_mps": args.target,
            "recommended_command_speed_mps": recommended,
            "fit_actual_equals_slope_times_command_plus_intercept": {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_squared,
            },
            "results": results,
            "topics": {"command": "/sim/cmd_vel", "measurement": "/sim/odom"},
        }
        output.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=True), flush=True)
    finally:
        node.send_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
