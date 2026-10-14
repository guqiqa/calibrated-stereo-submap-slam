#!/usr/bin/env python3
"""Strict C1/C4/P submap retrieval protocol.

This module is deliberately independent of ROS. It is the protocol and
decision layer used by online adapters and offline replay. Missing evidence is
represented explicitly instead of being replaced with a default score.
"""

from dataclasses import dataclass, field
import math
from typing import Iterable, Optional


METHODS = ("c1", "c4", "p")
WEIGHTS_C4 = {"visual": 0.50, "pose": 0.30, "topology": 0.20}
DEFAULT_BUDGET = {"visual": 4, "pose": 3, "topology": 2, "total": 10}


def merge_candidate_sources(visual, pose, topology, current=None, budget=DEFAULT_BUDGET):
    """Build the frozen round-robin union without changing per-source quotas."""
    sources = [list(visual)[:budget["visual"]], list(pose)[:budget["pose"]],
               list(topology)[:budget["topology"]]]
    result = []
    for rank in range(max((len(source) for source in sources), default=0)):
        for source in sources:
            if rank < len(source) and source[rank] not in result:
                result.append(source[rank])
                if len(result) == budget["total"]:
                    return result
    if current is not None and current not in result and len(result) < budget["total"]:
        result.append(current)
    return result


@dataclass(frozen=True)
class Evidence:
    available: bool
    score: float
    reliability: float
    source: str
    reason: str = ""

    def normalized_score(self) -> float:
        return min(1.0, max(0.0, float(self.score)))

    def normalized_reliability(self) -> float:
        return min(1.0, max(0.0, float(self.reliability)))


@dataclass
class Candidate:
    submap_id: int
    visual_score: float = 0.0
    pose_score: float = 0.0
    topology_score: float = 0.0
    source_flags: list[str] = field(default_factory=list)
    geometry_verified: bool = False
    pnp_inliers: int = 0
    reprojection_error_px: float = math.inf


@dataclass(frozen=True)
class Decision:
    method: str
    decision: str
    selected_submap: Optional[int]
    candidate_ids: list[int]
    candidate_covered: Optional[bool]
    top_probability: Optional[float]
    q_cov: Optional[float]
    reject_reason: str
    geometry_verified: bool
    evidence_available: dict[str, bool]

    def to_status_fields(self) -> dict:
        """Return the stable JSON fields consumed by ROS status adapters."""
        return {
            "retrieval_method": self.method,
            "retrieval_decision": self.decision,
            "selected_submap": self.selected_submap,
            "candidate_ids": self.candidate_ids,
            "candidate_covered": self.candidate_covered,
            "coverage_label_available": self.candidate_covered is not None,
            "q_top1": self.top_probability,
            "q_cov": self.q_cov,
            "geometry_verified": self.geometry_verified,
            "reject_reason": self.reject_reason,
            "evidence_available": self.evidence_available,
            "calibration_version": "unfitted-v0",
        }


def _probabilities(values: Iterable[float]) -> list[float]:
    """Stable softmax for scores already normalized to [0, 1]."""
    values = list(values)
    if not values:
        return []
    peak = max(values)
    exps = [math.exp(8.0 * (value - peak)) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def _geometry_ok(candidate: Candidate, min_inliers: int, max_reprojection_px: float) -> bool:
    return (candidate.geometry_verified or
            (candidate.pnp_inliers >= min_inliers and
             candidate.reprojection_error_px <= max_reprojection_px))


def _decision(method: str, candidates: list[Candidate], probabilities: list[float],
              evidences: dict[str, Evidence], *, candidate_covered: Optional[bool],
              min_probability: float, min_inliers: int,
              max_reprojection_px: float, low_reliability: float) -> Decision:
    available = {key: value.available for key, value in evidences.items()}
    if not candidates:
        return Decision(method, "relocalize", None, [], candidate_covered, None, None,
                        "candidate_set_empty", False, available)
    top = max(range(len(candidates)), key=lambda index: probabilities[index])
    candidate = candidates[top]
    geometry_ok = _geometry_ok(candidate, min_inliers, max_reprojection_px)
    top_probability = probabilities[top]
    q_cov = top_probability
    if method == "p" and all(e.reliability < low_reliability for e in evidences.values() if e.available):
        return Decision(method, "relocalize", None, [c.submap_id for c in candidates],
                        candidate_covered, top_probability, q_cov,
                        "all_evidence_low_reliability", geometry_ok, available)
    if not geometry_ok:
        return Decision(method, "reject", None, [c.submap_id for c in candidates],
                        candidate_covered, top_probability, q_cov,
                        "geometry_verification_failed", False, available)
    if top_probability < min_probability:
        return Decision(method, "reject", None, [c.submap_id for c in candidates],
                        candidate_covered, top_probability, q_cov,
                        "low_calibrated_confidence", True, available)
    return Decision(method, "accept", candidate.submap_id, [c.submap_id for c in candidates],
                    candidate_covered, top_probability, q_cov, "", True, available)


def _evidence_for(candidate: Candidate, visual: Evidence, pose: Evidence,
                  topology: Evidence) -> dict[str, Evidence]:
    return {
        "visual": Evidence(visual.available, candidate.visual_score, visual.reliability,
                            visual.source, visual.reason),
        "pose": Evidence(pose.available, candidate.pose_score, pose.reliability,
                          pose.source, pose.reason),
        "topology": Evidence(topology.available, candidate.topology_score,
                              topology.reliability, topology.source, topology.reason),
    }


def run_c1(candidates: list[Candidate], visual: Evidence, *, candidate_covered=None,
           min_probability=0.50, min_inliers=20, max_reprojection_px=3.0) -> Decision:
    """C1: visual retrieval plus geometric verification only."""
    if not visual.available:
        return Decision("c1", "relocalize", None, [c.submap_id for c in candidates],
                        candidate_covered, None, None, "visual_evidence_unavailable", False,
                        {"visual": False, "pose": False, "topology": False})
    probabilities = _probabilities([c.visual_score for c in candidates])
    unavailable = Evidence(False, 0.0, 0.0, "disabled", "method_c1")
    return _decision("c1", candidates, probabilities,
                     _evidence_for(candidates[0], visual, unavailable, unavailable)
                     if candidates else {"visual": visual, "pose": unavailable, "topology": unavailable},
                     candidate_covered=candidate_covered, min_probability=min_probability,
                     min_inliers=min_inliers, max_reprojection_px=max_reprojection_px,
                     low_reliability=0.0)


def run_c4(candidates: list[Candidate], visual: Evidence, pose: Evidence,
           topology: Evidence, *, candidate_covered=None, min_probability=0.50,
           min_inliers=20, max_reprojection_px=3.0) -> Decision:
    """C4: fixed-weight fusion; unavailable evidence is omitted and renormalized."""
    if not candidates:
        return _decision("c4", [], [], {"visual": visual, "pose": pose, "topology": topology},
                         candidate_covered=candidate_covered, min_probability=min_probability,
                         min_inliers=min_inliers, max_reprojection_px=max_reprojection_px,
                         low_reliability=0.0)
    evidences = {"visual": visual, "pose": pose, "topology": topology}
    weights = {key: WEIGHTS_C4[key] for key, value in evidences.items() if value.available}
    if not weights:
        return _decision("c4", candidates, [1.0 / len(candidates)] * len(candidates), evidences,
                         candidate_covered=candidate_covered, min_probability=min_probability,
                         min_inliers=min_inliers, max_reprojection_px=max_reprojection_px,
                         low_reliability=0.0)
    total = sum(weights.values())
    scores = []
    for candidate in candidates:
        scores.append(sum(weights[key] * getattr(candidate, key + "_score") for key in weights) / total)
    return _decision("c4", candidates, _probabilities(scores),
                     {key: value for key, value in evidences.items()},
                     candidate_covered=candidate_covered, min_probability=min_probability,
                     min_inliers=min_inliers, max_reprojection_px=max_reprojection_px,
                     low_reliability=0.0)


def run_p(candidates: list[Candidate], visual: Evidence, pose: Evidence,
          topology: Evidence, *, candidate_covered=None, min_probability=0.50,
          min_inliers=20, max_reprojection_px=3.0, low_reliability=0.20) -> Decision:
    """P: reliability-gated log-likelihood fusion with explicit rejection."""
    evidences = {"visual": visual, "pose": pose, "topology": topology}
    available = {key: value for key, value in evidences.items() if value.available}
    if not available:
        return Decision("p", "relocalize", None, [c.submap_id for c in candidates],
                        candidate_covered, None, None, "all_evidence_unavailable", False,
                        {key: False for key in evidences})
    if all(value.reliability < low_reliability for value in available.values()):
        return Decision("p", "relocalize", None, [c.submap_id for c in candidates],
                        candidate_covered, None, None, "all_evidence_low_reliability", False,
                        {key: value.available for key, value in evidences.items()})
    scores = []
    for candidate in candidates:
        likelihood = 0.0
        for key, evidence in available.items():
            score = min(1.0 - 1e-6, max(1e-6, getattr(candidate, key + "_score")))
            likelihood += evidence.reliability * math.log(score)
        scores.append(likelihood)
    return _decision("p", candidates, _probabilities(scores), evidences,
                     candidate_covered=candidate_covered, min_probability=min_probability,
                     min_inliers=min_inliers, max_reprojection_px=max_reprojection_px,
                     low_reliability=low_reliability)


def run(method: str, candidates: list[Candidate], evidences: dict[str, Evidence], **kwargs) -> Decision:
    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"unknown retrieval method: {method}; expected one of {METHODS}")
    if method == "c1":
        return run_c1(candidates, evidences["visual"], **kwargs)
    if method == "c4":
        return run_c4(candidates, evidences["visual"], evidences["pose"], evidences["topology"], **kwargs)
    return run_p(candidates, evidences["visual"], evidences["pose"], evidences["topology"], **kwargs)
