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
                 max_angular=0.22, turn_linear_speed=0.08,
                 yaw_tolerance=math.radians(4.0), settle_cycles=8):
        self.waypoints = waypoints
        self.speed = abs(float(speed))
        self.tolerance = float(tolerance)
        self.angular_gain = float(angular_gain)
        self.max_angular = float(max_angular)
        self.turn_linear_speed = abs(float(turn_linear_speed))
        self.yaw_tolerance = float(yaw_tolerance)
        self.settle_cycles = int(settle_cycles)
        self.index = 0
        self.completed = False
        self.phase = "cruise"
        self.settle_remaining = 0
        self.last_target_yaw = 0.0
        self.last_yaw_error = 0.0
        self.last_linear = 0.0
        self.last_angular = 0.0

    def _command(self, linear, angular):
        self.last_linear = float(linear)
        self.last_angular = float(angular)
        return self.last_linear, self.last_angular

    def command(self, x, y, yaw):
        while self.index < len(self.waypoints):
            tx, ty, target_yaw = self.waypoints[self.index]
            distance = math.hypot(tx - x, ty - y)
            if distance > self.tolerance or abs(normalize_angle(target_yaw - yaw)) > self.yaw_tolerance:
                break
            self.index += 1
        if self.index >= len(self.waypoints):
            self.completed = True
            self.phase = "complete"
            return self._command(0.0, 0.0)
        tx, ty, target_yaw = self.waypoints[self.index]
        self.last_target_yaw = target_yaw
        distance_to_target = math.hypot(tx - x, ty - y)
        yaw_error = normalize_angle(target_yaw - yaw)
        self.last_yaw_error = yaw_error

        # A waypoint with a changed yaw is a real turn state, not merely the
        # next positional target. Stop translation and rotate in place.
        if distance_to_target <= self.tolerance and abs(yaw_error) > self.yaw_tolerance:
            self.phase = "rotate"
        if self.phase == "rotate":
            if abs(yaw_error) <= self.yaw_tolerance:
                self.phase = "settle"
                self.settle_remaining = self.settle_cycles
            else:
                angular = max(-self.max_angular, min(self.max_angular,
                              self.angular_gain * yaw_error))
                return self._command(0.0, angular)
        if self.phase == "settle":
            if abs(yaw_error) > self.yaw_tolerance:
                self.phase = "rotate"
                return self._command(0.0, 0.0)
            self.settle_remaining -= 1
            if self.settle_remaining > 0:
                return self._command(0.0, 0.0)
            self.phase = "cruise"
            self.index += 1
            return self._command(0.0, 0.0)
        if distance_to_target <= self.tolerance:
            self.index += 1
            return self._command(0.0, 0.0)

        heading_error = normalize_angle(math.atan2(ty - y, tx - x) - yaw)
        angular = max(-self.max_angular, min(self.max_angular,
                      self.angular_gain * heading_error))
        error_abs = abs(heading_error)
        linear = self.speed * max(0.0, math.cos(heading_error))
        # Anticipate sharp turns from the route geometry. The old controller
        # only slowed after the heading error appeared, which is too late for
        # a stereo tracker approaching a 180-degree turnaround.
        if self.index + 2 < len(self.waypoints):
            nx, ny, _ = self.waypoints[self.index + 1]
            fx, fy, _ = self.waypoints[self.index + 2]
            incoming = math.atan2(ny - ty, nx - tx)
            outgoing = math.atan2(fy - nx, fx - ny)
            turn_angle = abs(normalize_angle(outgoing - incoming))
            distance_to_corner = math.hypot(tx - x, ty - y)
            if turn_angle > 0.75 and distance_to_corner < 2.5:
                proximity = max(0.0, min(1.0, (2.5 - distance_to_corner) / 2.5))
                linear *= 1.0 - 0.65 * proximity
        # Gradually slow before a junction turn; avoid an abrupt stop-and-spin
        # that changes the view faster than the stereo tracker can follow.
        if error_abs > 0.30:
            linear *= max(0.20, 1.0 - (error_abs - 0.30) / 0.70)
        if error_abs > 0.95:
            linear = 0.0
        if error_abs > 0.52:
            linear = min(linear, self.turn_linear_speed)
        return self._command(linear, angular)


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
