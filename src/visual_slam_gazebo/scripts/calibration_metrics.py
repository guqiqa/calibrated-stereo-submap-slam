#!/usr/bin/env python3
"""Dependency-free probability calibration and risk-coverage metrics."""

import math


def _clean(y_true, confidence):
    pairs = [(int(bool(y)), min(1.0 - 1e-6, max(1e-6, float(c))))
             for y, c in zip(y_true, confidence)]
    if not pairs:
        raise ValueError("at least one labeled prediction is required")
    return pairs


def ece(y_true, confidence, bins=10):
    pairs = _clean(y_true, confidence)
    if bins < 1:
        raise ValueError("bins must be positive")
    total = len(pairs)
    result = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        bucket = [(y, c) for y, c in pairs if lower <= c < upper or
                  (index == bins - 1 and c <= upper)]
        if bucket:
            result += len(bucket) / total * abs(sum(y for y, _ in bucket) / len(bucket) -
                                                sum(c for _, c in bucket) / len(bucket))
    return result


def brier_score(y_true, confidence):
    pairs = _clean(y_true, confidence)
    return sum((c - y) ** 2 for y, c in pairs) / len(pairs)


def nll(y_true, confidence, epsilon=1e-6):
    pairs = _clean(y_true, confidence)
    return -sum(y * math.log(max(epsilon, c)) + (1 - y) * math.log(max(epsilon, 1 - c))
                for y, c in pairs) / len(pairs)


def risk_coverage_curve(y_true, confidence, steps=101):
    pairs = sorted(_clean(y_true, confidence), key=lambda item: item[1], reverse=True)
    if steps < 2:
        raise ValueError("steps must be at least 2")
    curve = []
    for index in range(steps):
        coverage = index / (steps - 1)
        count = min(len(pairs), int(round(coverage * len(pairs))))
        accepted = pairs[:count]
        risk = (1.0 - sum(y for y, _ in accepted) / count) if accepted else 0.0
        curve.append({"coverage": coverage, "risk": risk, "count": count})
    area = sum((curve[i - 1]["risk"] + curve[i]["risk"]) / 2 *
               (curve[i]["coverage"] - curve[i - 1]["coverage"])
               for i in range(1, len(curve)))
    return {"points": curve, "au_rc": area}
