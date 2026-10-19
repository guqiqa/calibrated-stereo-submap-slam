#!/usr/bin/env python3
"""Per-session angular delivery calibration.

Measures the four anchors (+/-0.35 and +/-0.50 rad/s) on the live simulator and
writes a session calibration file. The anchors are commands with their delivered
rates; the file is the only accepted source of compensation values for a formal
run. Sessions drift, so each formal session must produce its own file.

Usage:
  session_angular_calibration.py <output.json> [session_id]
"""
from __future__ import annotations

import json
import math
import statistics
import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

ANCHORS = (0.35, 0.50)
HOLD_SECONDS = 8.0
TRIM_SECONDS = 1.0
MAX_COMMAND = 0.50
ASYMMETRY_LIMIT = 0.05


def stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def yaw_z(qx, qy, qz, qw) -> float:
    return math.degrees(math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))


def wrap(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


class Calibrator(Node):
    def __init__(self) -> None:
        super().__init__("session_angular_calibration")
        self.publisher = self.create_publisher(Twist, "/sim/cmd_vel", 10)
        self.odom: list[tuple[float, float]] = []
        self.create_subscription(Odometry, "/sim/odom", self.on_odom, 400)

    def on_odom(self, message: Odometry) -> None:
        q = message.pose.pose.orientation
        self.odom.append((stamp_seconds(message.header.stamp), yaw_z(q.x, q.y, q.z, q.w)))

    def hold(self, command: float, seconds: float) -> list[tuple[float, float]]:
        before = len(self.odom)
        message = Twist()
        message.angular.z = command
        end = time.time() + seconds
        while time.time() < end:
            self.publisher.publish(message)
            rclpy.spin_once(self, timeout_sec=0.002)
        stop = Twist()
        stop_end = time.time() + 1.5
        while time.time() < stop_end:
            self.publisher.publish(stop)
            rclpy.spin_once(self, timeout_sec=0.01)
        return self.odom[before:]


def delivered_rate(segment, command):
    """Signed delivered rate from the trimmed window, unwrapping the yaw."""
    begin, finish = segment[0][0], segment[-1][0]
    inner = [(t, y) for t, y in segment if begin + TRIM_SECONDS <= t <= finish - TRIM_SECONDS]
    if len(inner) < 10:
        return float("nan"), float("nan"), float("nan")
    unwrapped = []
    offset = 0.0
    previous = None
    for stamp, yaw in inner:
        if previous is not None:
            delta = yaw - previous
            while delta > 180.0:
                delta -= 360.0
            while delta < -180.0:
                delta += 360.0
            offset += delta - (yaw - previous)
        unwrapped.append((stamp, yaw + offset))
        previous = yaw
    n = float(len(unwrapped))
    mean_t = sum(t for t, _ in unwrapped) / n
    mean_y = sum(y for _, y in unwrapped) / n
    numerator = sum((t - mean_t) * (y - mean_y) for t, y in unwrapped)
    denominator = sum((t - mean_t) ** 2 for t, _ in unwrapped)
    slope = numerator / denominator if denominator > 0 else float("nan")
    per = [abs(math.radians(wrap(b[1] - a[1])) / (b[0] - a[0]))
           for a, b in zip(unwrapped, unwrapped[1:]) if b[0] - a[0] > 1e-9]
    ordered = sorted(per)
    p05 = ordered[int(0.05 * (len(ordered) - 1))] if ordered else float("nan")
    p95 = ordered[int(0.95 * (len(ordered) - 1))] if ordered else float("nan")
    return math.radians(slope), p05, p95


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: session_angular_calibration.py <output.json> [session_id]")
        return 2
    output = sys.argv[1]
    session_id = sys.argv[2] if len(sys.argv) > 2 else time.strftime("%Y%m%d-%H%M%S")

    rclpy.init()
    node = Calibrator()
    end = time.time() + 4.0
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    if not node.odom:
        print("no /sim/odom; is Gazebo running?")
        return 1

    measured = {}
    for command in ANCHORS:
        for sign in (1.0, -1.0):
            rate, _, _ = delivered_rate(node.hold(command * sign, HOLD_SECONDS), command)
            measured[(command, sign)] = rate
    node.destroy_node()
    rclpy.shutdown()

    anchors = []
    valid = True
    asymmetry = 0.0
    for command in ANCHORS:
        positive = abs(measured[(command, 1.0)])
        negative = abs(measured[(command, -1.0)])
        average = 0.5 * (positive + negative)
        difference = abs(positive - negative) / average if average > 1e-9 else 1.0
        asymmetry = max(asymmetry, difference)
        if difference > ASYMMETRY_LIMIT:
            valid = False
        anchors.append({
            "command": command,
            "delivered": round(average, 6),
            "delivered_positive": round(positive, 6),
            "delivered_negative": round(negative, 6),
            "direction_difference": round(difference, 6),
        })

    delivered_values = [item["delivered"] for item in anchors]
    payload = {
        "calibration_version": "session_v1",
        "session_id": session_id,
        "source": "sim_odom",
        "anchors": anchors,
        "max_safe_command": min(MAX_COMMAND, max(item["command"] for item in anchors)),
        "max_safe_delivered": max(delivered_values),
        "min_delivered": min(delivered_values),
        "direction_asymmetry": round(asymmetry, 6),
        "valid": bool(valid and all(value > 0.0 for value in delivered_values)),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0 if payload["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
