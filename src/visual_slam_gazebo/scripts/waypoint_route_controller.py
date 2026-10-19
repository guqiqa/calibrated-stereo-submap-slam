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


# Delivery calibration: maps a TARGET delivered angular rate to the angular.z
# that /sim/cmd_vel has to carry. Measured on 2026-10-18 with cmd_delivery_probe
# (8 s per segment, 20 Hz, simulation time, 1 s trimmed at each end).
#
# The plant has a deadband and a non-linear gain (0.085 -> 0.694 delivered ratio),
# so a single global linear inverse is not valid; the table is interpolated
# piecewise and clamped at the last verified point. 0.60 rad/s was measured to
# invert the sign of the delivered rotation and is deliberately unreachable.
IN_PLACE_V1 = (
    (0.0, 0.0),
    (0.0042, 0.05),
    (0.0195, 0.10),
    (0.0456, 0.15),
    (0.0878, 0.20),
    (0.1740, 0.30),
    (0.2608, 0.40),
    (0.3472, 0.50),
)
CALIBRATION_TABLES = {"in_place_v1": IN_PLACE_V1}
MAX_COMMAND_ANGULAR = 0.50
DEFAULT_MAX_DELIVERED_ANGULAR = 0.3472
DISABLED_CALIBRATION = "disabled"
SESSION_CALIBRATION = "session_v1"
SESSION_ASYMMETRY_LIMIT = 0.05


class SessionCalibrationError(ValueError):
    """Raised when a session calibration file cannot be trusted."""


def load_session_calibration(path):
    """Load and validate a per-session calibration file.

    Returns (table, metadata) where table is an ascending list of
    (delivered, command) pairs. Refuses anything that is not a fresh, valid,
    sim_odom-sourced session file with at least two positive anchors inside the
    safe command range.
    """
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("calibration_version") != SESSION_CALIBRATION:
        raise SessionCalibrationError(
            f"unexpected calibration_version: {payload.get('calibration_version')}")
    if payload.get("source") != "sim_odom":
        raise SessionCalibrationError(f"unexpected calibration source: {payload.get('source')}")
    if not payload.get("valid"):
        raise SessionCalibrationError("session calibration is marked invalid")
    if not str(payload.get("session_id") or "").strip():
        raise SessionCalibrationError("session calibration has no session_id")
    anchors = payload.get("anchors") or []
    if len(anchors) < 2:
        raise SessionCalibrationError("session calibration needs at least two anchors")
    pairs = []
    for anchor in anchors:
        try:
            delivered = float(anchor["delivered"])
            command = float(anchor["command"])
        except (KeyError, TypeError, ValueError) as error:
            raise SessionCalibrationError(f"malformed anchor: {anchor}") from error
        if delivered <= 0.0:
            raise SessionCalibrationError(f"non-positive delivered rate: {delivered}")
        if abs(command) > MAX_COMMAND_ANGULAR + 1e-12:
            raise SessionCalibrationError(f"anchor command above the safe limit: {command}")
        pairs.append((delivered, command))
    pairs.sort()
    for (low_delivered, low_command), (high_delivered, high_command) in zip(pairs, pairs[1:]):
        if high_delivered <= low_delivered or high_command <= low_command:
            raise SessionCalibrationError("anchors are not strictly increasing")
    asymmetry = float(payload.get("direction_asymmetry") or 0.0)
    if asymmetry > SESSION_ASYMMETRY_LIMIT:
        raise SessionCalibrationError(f"direction asymmetry too large: {asymmetry}")
    return pairs, payload


def angular_command_from_delivered(target, table, max_delivered):
    """Map a target delivered angular rate to the command to publish.

    Returns (command, clamped). Sign is symmetric, zero maps to zero, and the
    output never exceeds MAX_COMMAND_ANGULAR.
    """
    magnitude = abs(float(target))
    clamped = False
    if magnitude > max_delivered:
        magnitude = float(max_delivered)
        clamped = True
    command = float(table[-1][1])
    for (low_delivered, low_command), (high_delivered, high_command) in zip(table, table[1:]):
        if magnitude <= high_delivered:
            span = high_delivered - low_delivered
            ratio = 0.0 if span <= 0.0 else (magnitude - low_delivered) / span
            command = low_command + ratio * (high_command - low_command)
            break
    command = min(command, MAX_COMMAND_ANGULAR)
    if target == 0.0:
        return 0.0, clamped
    return math.copysign(command, target), clamped


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
                 yaw_tolerance=math.radians(4.0), settle_cycles=8,
                 angular_calibration=DISABLED_CALIBRATION,
                 max_delivered_angular=DEFAULT_MAX_DELIVERED_ANGULAR,
                 angular_calibration_file=None):
        self.waypoints = waypoints
        self.speed = abs(float(speed))
        self.tolerance = float(tolerance)
        self.angular_gain = float(angular_gain)
        # With calibration enabled this is the TARGET delivered rate; with it
        # disabled it keeps its historical meaning (the published command).
        self.max_angular = float(max_angular)
        self.turn_linear_speed = abs(float(turn_linear_speed))
        self.yaw_tolerance = float(yaw_tolerance)
        self.settle_cycles = int(settle_cycles)
        self.angular_calibration = str(angular_calibration)
        self.angular_session = None
        if self.angular_calibration == DISABLED_CALIBRATION:
            self.angular_table = None
            self.max_delivered_angular = float(max_delivered_angular)
        elif self.angular_calibration == SESSION_CALIBRATION:
            if not angular_calibration_file:
                raise SessionCalibrationError(
                    "session_v1 requires an --angular-calibration-file")
            self.angular_table, self.angular_session = load_session_calibration(
                angular_calibration_file)
            # Never claim delivery above the highest verified anchor.
            self.max_delivered_angular = min(float(max_delivered_angular),
                                             self.angular_table[-1][0])
        else:
            self.angular_table = CALIBRATION_TABLES.get(self.angular_calibration)
            if self.angular_table is None:
                raise ValueError(f"unknown angular calibration: {angular_calibration}")
            self.max_delivered_angular = float(max_delivered_angular)
        self.index = 0
        self.completed = False
        self.phase = "cruise"
        self.settle_remaining = 0
        self.last_target_yaw = 0.0
        self.last_yaw_error = 0.0
        self.last_linear = 0.0
        self.last_angular = 0.0
        self.last_angular_target = 0.0
        self.last_angular_calibration = self.angular_calibration
        self.last_angular_clamped = False

    def _command(self, linear, angular):
        self.last_linear = float(linear)
        self.last_angular_target = float(angular)
        self.last_angular_low_rate = False
        if self.angular_table is None:
            self.last_angular = float(angular)
            self.last_angular_calibration = DISABLED_CALIBRATION
            self.last_angular_clamped = False
        elif abs(float(angular)) < self.angular_table[0][0]:
            # Below the lowest verified anchor the compensation is NOT applied:
            # holding a fixed lowest command fights the controller and prevents it
            # from ever converging, which is what kept the VO rotating. Pass the
            # target through unchanged and flag the cycle as unpromised instead.
            self.last_angular_low_rate = True
            self.last_angular = float(angular)
            self.last_angular_calibration = self.angular_calibration
            self.last_angular_clamped = False
        else:
            command, clamped = angular_command_from_delivered(
                float(angular), self.angular_table, self.max_delivered_angular)
            self.last_angular = command
            self.last_angular_calibration = self.angular_calibration
            self.last_angular_clamped = clamped
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
        def __init__(self, route, speed, report, max_angular=0.22,
                     angular_calibration=DISABLED_CALIBRATION,
                     max_delivered_angular=DEFAULT_MAX_DELIVERED_ANGULAR):
            super().__init__("visual_slam_waypoint_controller")
            self.plan = WaypointPlan(load_waypoints(route), speed,
                                     max_angular=max_angular,
                                     angular_calibration=angular_calibration,
                                     max_delivered_angular=max_delivered_angular)
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
    parser.add_argument("--max-angular", type=float, default=0.22)
    parser.add_argument("--angular-calibration", default=DISABLED_CALIBRATION,
                        choices=[DISABLED_CALIBRATION, *sorted(CALIBRATION_TABLES)],
                        help="Delivery compensation table for angular.z")
    parser.add_argument("--max-delivered-angular", type=float,
                        default=DEFAULT_MAX_DELIVERED_ANGULAR)
    parser.add_argument("--report")
    args = parser.parse_args()
    rclpy.init(); node = WaypointController(args.route, args.speed, args.report,
                                             args.max_angular,
                                             args.angular_calibration,
                                             args.max_delivered_angular)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
