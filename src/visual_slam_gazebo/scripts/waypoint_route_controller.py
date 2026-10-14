#!/usr/bin/env python3
"""Follow generated CSV waypoints using simulation truth for motion control only."""

import argparse
import csv
import json
import math
from pathlib import Path

try:
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
except ImportError:  # Pure route-plan tests do not require ROS 2.
    rclpy = None
    Twist = Odometry = Node = None


def normalize_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def load_waypoints(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"waypoint route is empty: {path}")
    return [(float(row["x"]), float(row["y"]), float(row.get("yaw", 0.0))) for row in rows]


class WaypointPlan:
    """Pure control state shared by the standalone and experiment controllers."""

    def __init__(self, waypoints, speed, tolerance=0.35, angular_gain=1.8,
                 max_angular=0.8):
        self.waypoints = waypoints
        self.speed = abs(float(speed))
        self.tolerance = float(tolerance)
        self.angular_gain = float(angular_gain)
        self.max_angular = float(max_angular)
        self.index = 0
        self.completed = False

    def command(self, x, y, yaw):
        while self.index < len(self.waypoints):
            tx, ty, _ = self.waypoints[self.index]
            if math.hypot(tx - x, ty - y) > self.tolerance:
                break
            self.index += 1
        if self.index >= len(self.waypoints):
            self.completed = True
            return 0.0, 0.0
        tx, ty, _ = self.waypoints[self.index]
        heading_error = normalize_angle(math.atan2(ty - y, tx - x) - yaw)
        angular = max(-self.max_angular, min(self.max_angular,
                      self.angular_gain * heading_error))
        linear = self.speed * max(0.0, math.cos(heading_error))
        if abs(heading_error) > 0.65:
            linear = 0.0
        return linear, angular


if Node is not None:
    class WaypointController(Node):
        def __init__(self, route, speed, report):
            super().__init__("visual_slam_waypoint_controller")
            self.plan = WaypointPlan(load_waypoints(route), speed)
            self.report = Path(report) if report else None
            self.publisher = self.create_publisher(Twist, "/sim/cmd_vel", 10)
            self.create_subscription(Odometry, "/sim/odom", self.odom_callback, 20)

        def odom_callback(self, message):
            pose = message.pose.pose
            linear, angular = self.plan.command(
                pose.position.x, pose.position.y, yaw_from_quaternion(pose.orientation))
            command = Twist(); command.linear.x = linear; command.angular.z = angular
            self.publisher.publish(command)
            if self.plan.completed:
                if self.report:
                    self.report.write_text(json.dumps({"state": "complete",
                        "waypoints": len(self.plan.waypoints)}, indent=2), encoding="utf-8")
                rclpy.shutdown()
else:
    class WaypointController:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ROS 2 is required to run the waypoint controller")


def main():
    if rclpy is None:
        raise SystemExit("ROS 2 is required to run waypoint_route_controller.py")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route")
    parser.add_argument("--speed", type=float, default=0.254522061)
    parser.add_argument("--report")
    args = parser.parse_args()
    rclpy.init(); node = WaypointController(args.route, args.speed, args.report)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
