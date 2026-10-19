#!/usr/bin/env python3
"""Unit tests for the per-session angular calibration loading and validation."""

import json
import math
import tempfile
import unittest
from pathlib import Path

from waypoint_route_controller import (
    MAX_COMMAND_ANGULAR,
    SESSION_CALIBRATION,
    SessionCalibrationError,
    WaypointPlan,
    load_session_calibration,
)


def session_payload(**overrides):
    payload = {
        "calibration_version": SESSION_CALIBRATION,
        "session_id": "session-test",
        "source": "sim_odom",
        "anchors": [
            {"command": 0.35, "delivered": 0.2448},
            {"command": 0.50, "delivered": 0.3911},
        ],
        "max_safe_command": 0.50,
        "max_safe_delivered": 0.3911,
        "direction_asymmetry": 0.006,
        "valid": True,
    }
    payload.update(overrides)
    return payload


class SessionCalibrationLoadTest(unittest.TestCase):
    def _write(self, payload):
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(payload, handle)
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return handle.name

    def test_valid_file_produces_ascending_table(self):
        table, metadata = load_session_calibration(self._write(session_payload()))
        self.assertEqual(len(table), 2)
        self.assertAlmostEqual(table[0][0], 0.2448, places=6)
        self.assertAlmostEqual(table[0][1], 0.35, places=6)
        self.assertAlmostEqual(table[1][0], 0.3911, places=6)
        self.assertEqual(metadata["session_id"], "session-test")

    def test_rejects_wrong_version(self):
        with self.assertRaises(SessionCalibrationError):
            load_session_calibration(self._write(session_payload(calibration_version="v9")))

    def test_rejects_invalid_flag(self):
        with self.assertRaises(SessionCalibrationError):
            load_session_calibration(self._write(session_payload(valid=False)))

    def test_rejects_wrong_source(self):
        with self.assertRaises(SessionCalibrationError):
            load_session_calibration(self._write(session_payload(source="wall_clock")))

    def test_rejects_missing_session_id(self):
        with self.assertRaises(SessionCalibrationError):
            load_session_calibration(self._write(session_payload(session_id="")))

    def test_rejects_large_asymmetry(self):
        with self.assertRaises(SessionCalibrationError):
            load_session_calibration(self._write(session_payload(direction_asymmetry=0.2)))

    def test_rejects_single_anchor(self):
        with self.assertRaises(SessionCalibrationError):
            load_session_calibration(self._write(
                session_payload(anchors=[{"command": 0.35, "delivered": 0.2448}])))

    def test_rejects_non_positive_delivered(self):
        with self.assertRaises(SessionCalibrationError):
            load_session_calibration(self._write(session_payload(
                anchors=[{"command": 0.35, "delivered": 0.0},
                         {"command": 0.50, "delivered": 0.3911}])))

    def test_rejects_command_above_safe_limit(self):
        with self.assertRaises(SessionCalibrationError):
            load_session_calibration(self._write(session_payload(
                anchors=[{"command": 0.35, "delivered": 0.2448},
                         {"command": 0.60, "delivered": 0.45}])))


class SessionCalibrationPlanTest(unittest.TestCase):
    def _file(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(session_payload(), handle)
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return handle.name

    def _plan(self, **kwargs):
        return WaypointPlan([(0.0, 0.0, math.pi / 2), (1.0, 0.0, math.pi / 2)], 0.2, **kwargs)

    def test_requires_a_file(self):
        with self.assertRaises(SessionCalibrationError):
            self._plan(angular_calibration=SESSION_CALIBRATION)

    def test_caps_max_delivered_at_the_top_anchor(self):
        plan = self._plan(angular_calibration=SESSION_CALIBRATION,
                          angular_calibration_file=self._file(),
                          max_delivered_angular=0.9)
        self.assertAlmostEqual(plan.max_delivered_angular, 0.3911, places=6)

    def test_below_lowest_anchor_is_flagged_not_extrapolated(self):
        plan = self._plan(angular_calibration=SESSION_CALIBRATION,
                          angular_calibration_file=self._file(),
                          max_angular=0.05)
        plan.command(0.0, 0.0, 0.0)
        self.assertLess(abs(plan.last_angular_target), 0.2448)
        self.assertTrue(plan.last_angular_low_rate)
        self.assertAlmostEqual(abs(plan.last_angular), 0.35, places=6)

    def test_inside_range_maps_through_the_anchors(self):
        plan = self._plan(angular_calibration=SESSION_CALIBRATION,
                          angular_calibration_file=self._file(),
                          max_angular=0.30)
        plan.command(0.0, 0.0, 0.0)
        self.assertGreaterEqual(abs(plan.last_angular_target), 0.2448)
        self.assertLessEqual(abs(plan.last_angular), MAX_COMMAND_ANGULAR + 1e-12)
        self.assertEqual(plan.last_angular_calibration, SESSION_CALIBRATION)
        self.assertEqual(plan.angular_session["session_id"], "session-test")


if __name__ == "__main__":
    unittest.main()
