#!/usr/bin/env python3
import math
import unittest

from calibration_metrics import brier_score, ece, nll, risk_coverage_curve
from fit_calibration_model import fit_logistic, sigmoid
from submap_retrieval_protocol import (Candidate, Evidence, merge_candidate_sources,
                                       run_c1, run_c4, run_p)


def evidence(available=True, score=0.9, reliability=0.9, source="test"):
    return Evidence(available, score, reliability, source)


class ProtocolTests(unittest.TestCase):
    def candidates(self):
        return [Candidate(1, .9, .2, .2, geometry_verified=True, pnp_inliers=30, reprojection_error_px=1),
                Candidate(2, .5, .9, .2, geometry_verified=True, pnp_inliers=30, reprojection_error_px=1)]

    def test_c1_uses_visual_only(self):
        decision = run_c1(self.candidates(), evidence())
        self.assertEqual(decision.selected_submap, 1)
        self.assertFalse(decision.evidence_available["pose"])
        self.assertFalse(decision.evidence_available["topology"])

    def test_c4_fixed_weights(self):
        decision = run_c4(self.candidates(), evidence(), evidence(), evidence())
        self.assertEqual(decision.selected_submap, 2)

    def test_p_drops_unavailable_evidence(self):
        decision = run_p(self.candidates(), evidence(), evidence(False, reliability=0), evidence())
        self.assertEqual(decision.selected_submap, 1)
        self.assertFalse(decision.evidence_available["pose"])

    def test_p_relocalizes_when_all_reliability_is_low(self):
        low = evidence(reliability=.05)
        decision = run_p(self.candidates(), low, low, low)
        self.assertEqual(decision.decision, "relocalize")
        self.assertEqual(decision.reject_reason, "all_evidence_low_reliability")

    def test_geometry_failure_is_distinct(self):
        candidate = Candidate(1, .95, .1, .1, geometry_verified=False, pnp_inliers=2, reprojection_error_px=10)
        decision = run_c1([candidate], evidence())
        self.assertEqual(decision.reject_reason, "geometry_verification_failed")

    def test_candidate_union_is_deterministic_and_deduplicated(self):
        merged = merge_candidate_sources([1, 2, 3, 4], [2, 5, 6], [1, 7], current=8)
        self.assertEqual(merged, [1, 2, 5, 7, 3, 6, 4, 8])
        self.assertEqual(len(merged), len(set(merged)))

    def test_all_evidence_unavailable_is_explicit(self):
        unavailable = evidence(False, reliability=0)
        decision = run_p(self.candidates(), unavailable, unavailable, unavailable)
        self.assertEqual(decision.decision, "relocalize")
        self.assertEqual(decision.reject_reason, "all_evidence_unavailable")

    def test_empty_candidate_set_has_no_nan_probability(self):
        decision = run_c1([], evidence())
        self.assertIsNone(decision.top_probability)
        self.assertEqual(decision.reject_reason, "candidate_set_empty")


class CalibrationTests(unittest.TestCase):
    def test_metrics(self):
        labels, confidence = [1, 0, 1, 0], [.9, .1, .8, .2]
        self.assertAlmostEqual(brier_score(labels, confidence), .025)
        self.assertLess(ece(labels, confidence), .2)
        self.assertLessEqual(ece(labels, confidence), 1.0)
        self.assertGreater(nll(labels, confidence), 0)
        curve = risk_coverage_curve(labels, confidence)
        self.assertEqual(curve["points"][0]["coverage"], 0.0)
        self.assertTrue(math.isfinite(curve["au_rc"]))

    def test_logistic_calibration_separates_binary_samples(self):
        model = fit_logistic([([-2.0], False), ([-1.0], False),
                              ([1.0], True), ([2.0], True)], steps=1000)
        low = sigmoid(model["intercept"] + model["weights"][0] *
                      (-1.5 - model["mean"][0]) / model["scale"][0])
        high = sigmoid(model["intercept"] + model["weights"][0] *
                       (1.5 - model["mean"][0]) / model["scale"][0])
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)


if __name__ == "__main__":
    unittest.main()
