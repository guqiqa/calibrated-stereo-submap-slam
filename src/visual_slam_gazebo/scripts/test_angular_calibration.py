#!/usr/bin/env python3
"""Unit tests for the angular delivery calibration.

These run without ROS: the calibration table and WaypointPlan are pure Python.
"""

import math
import unittest

from waypoint_route_controller import (
    CALIBRATION_TABLES,
    DISABLED_CALIBRATION,
    IN_PLACE_V1,
    MAX_COMMAND_ANGULAR,
    WaypointPlan,
    angular_command_from_delivered,
)


class AngularCalibrationTableTest(unittest.TestCase):
    def test_endpoints_map_exactly(self):
        for delivered, command in IN_PLACE_V1:
            actual, clamped = angular_command_from_delivered(
                delivered, IN_PLACE_V1, 0.3472)
            self.assertAlmostEqual(actual, command, places=9)
            self.assertFalse(clamped)

    def test_delivered_upper_endpoint_maps_to_command_limit(self):
        command, clamped = angular_command_from_delivered(0.3472, IN_PLACE_V1, 0.3472)
        self.assertAlmostEqual(command, 0.50, places=9)
        self.assertFalse(clamped)

    def test_interpolation_inside_a_segment(self):
        # Midway between (0.0878, 0.20) and (0.1740, 0.30).
        midpoint = 0.5 * (0.0878 + 0.1740)
        command, _ = angular_command_from_delivered(midpoint, IN_PLACE_V1, 0.3472)
        self.assertAlmostEqual(command, 0.25, places=9)

    def test_zero_maps_to_zero(self):
        command, clamped = angular_command_from_delivered(0.0, IN_PLACE_V1, 0.3472)
        self.assertEqual(command, 0.0)
        self.assertFalse(clamped)

    def test_sign_symmetry(self):
        for delivered in (0.01, 0.05, 0.1, 0.2, 0.3, 0.3472):
            positive, _ = angular_command_from_delivered(delivered, IN_PLACE_V1, 0.3472)
            negative, _ = angular_command_from_delivered(-delivered, IN_PLACE_V1, 0.3472)
            self.assertAlmostEqual(positive, -negative, places=12)

    def test_above_cap_is_clamped(self):
        command, clamped = angular_command_from_delivered(0.9, IN_PLACE_V1, 0.3472)
        self.assertTrue(clamped)
        self.assertAlmostEqual(abs(command), MAX_COMMAND_ANGULAR, places=9)

    def test_command_never_reaches_the_unstable_zone(self):
        delivered = 0.0
        while delivered <= 2.0:
            command, _ = angular_command_from_delivered(delivered, IN_PLACE_V1, 0.3472)
            self.assertLessEqual(abs(command), MAX_COMMAND_ANGULAR + 1e-12)
            self.assertLess(abs(command), 0.55)
            delivered += 0.01

    def test_unknown_table_is_rejected(self):
        with self.assertRaises(ValueError):
            WaypointPlan([(0.0, 0.0, 0.0)], 0.2, angular_calibration="does_not_exist")

    def test_registered_table_is_reachable(self):
        self.assertIn("in_place_v1", CALIBRATION_TABLES)


class WaypointPlanCalibrationTest(unittest.TestCase):
    def _plan(self, **kwargs):
        # A route whose first waypoint requires a pure in-place rotation.
        return WaypointPlan([(0.0, 0.0, math.pi / 2), (1.0, 0.0, math.pi / 2)], 0.2, **kwargs)

    def test_disabled_matches_legacy_behaviour(self):
        plan = self._plan()
        self.assertEqual(plan.angular_calibration, DISABLED_CALIBRATION)
        _, angular = plan.command(0.0, 0.0, 0.0)
        self.assertAlmostEqual(plan.last_angular_target, angular, places=12)
        self.assertEqual(plan.last_angular_calibration, DISABLED_CALIBRATION)
        self.assertFalse(plan.last_angular_clamped)

    def test_enabled_raises_the_command(self):
        plan = self._plan(angular_calibration="in_place_v1")
        _, angular = plan.command(0.0, 0.0, 0.0)
        self.assertGreater(abs(plan.last_angular), abs(plan.last_angular_target))
        self.assertEqual(plan.last_angular_calibration, "in_place_v1")
        self.assertLessEqual(abs(plan.last_angular), MAX_COMMAND_ANGULAR + 1e-12)

    def test_enabled_keeps_the_rotation_sign(self):
        plan = self._plan(angular_calibration="in_place_v1")
        _, angular = plan.command(0.0, 0.0, 0.0)
        target = plan.last_angular_target
        self.assertNotEqual(target, 0.0)
        self.assertEqual(math.copysign(1.0, angular), math.copysign(1.0, target))

    def test_clamped_flag_surfaces(self):
        plan = self._plan(angular_calibration="in_place_v1", max_angular=1.0,
                          max_delivered_angular=0.05)
        _, angular = plan.command(0.0, 0.0, 0.0)
        self.assertTrue(plan.last_angular_clamped)
        self.assertAlmostEqual(abs(angular), 0.0 + abs(angular))
        self.assertLessEqual(abs(angular), MAX_COMMAND_ANGULAR + 1e-12)


if __name__ == "__main__":
    unittest.main()
