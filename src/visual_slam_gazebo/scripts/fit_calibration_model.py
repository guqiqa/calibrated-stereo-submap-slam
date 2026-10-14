#!/usr/bin/env python3
"""Fit frozen reliability and confidence calibration models from labeled pilot logs."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value))))


def load_rows(root):
    rows = []
    for path in sorted(Path(root).rglob("status.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("coverage_label_available") and row.get("evaluation_eligible", row.get("state") == "tracking"):
                rows.append(row)
    return rows


def fit_logistic(samples, l2=0.01, steps=2500, rate=0.05):
    if not samples:
        raise ValueError("no calibration samples")
    labels = [int(bool(label)) for _, label in samples]
    if len(set(labels)) != 2:
        raise ValueError("calibration labels must contain both classes")
    width = len(samples[0][0])
    mean = [sum(x[index] for x, _ in samples) / len(samples) for index in range(width)]
    scale = []
    for index in range(width):
        variance = sum((x[index] - mean[index]) ** 2 for x, _ in samples) / len(samples)
        scale.append(max(1e-6, math.sqrt(variance)))
    normalized = [([(x[i] - mean[i]) / scale[i] for i in range(width)], y)
                  for x, y in samples]
    weights = [0.0] * width
    intercept = 0.0
    for _ in range(steps):
        gradient = [0.0] * width
        intercept_gradient = 0.0
        for features, label in normalized:
            prediction = sigmoid(intercept + sum(w * x for w, x in zip(weights, features)))
            error = prediction - label
            intercept_gradient += error
            for index, value in enumerate(features):
                gradient[index] += error * value
        count = len(normalized)
        intercept -= rate * intercept_gradient / count
        for index in range(width):
            weights[index] -= rate * (gradient[index] / count + l2 * weights[index])
    return {"weights": weights, "intercept": intercept, "mean": mean, "scale": scale}


def evidence_samples(rows, name):
    result = []
    for row in rows:
        features = row.get("reliability_features", {}).get(name)
        source = row.get("candidate_sources", {}).get(name, [])
        legal = set(row.get("legal_submaps", []))
        if isinstance(features, list) and features and source and legal:
            result.append(([float(value) for value in features], source[0] in legal))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--stage", choices=("reliability", "confidence"),
                        default="reliability")
    parser.add_argument("--base-model",
                        help="stage-1 model used for the confidence calibration rerun")
    args = parser.parse_args()
    rows = load_rows(args.status_root)
    if not rows:
        raise SystemExit("no independently labeled pilot rows found")
    models, counts = {}, {}
    if args.stage == "reliability":
        for name in ("visual", "pose", "topology"):
            samples = evidence_samples(rows, name)
            counts[name] = len(samples)
            models[name] = fit_logistic(samples, l2=args.l2)
    else:
        if not args.base_model:
            raise SystemExit("confidence stage requires --base-model")
        base = json.loads(Path(args.base_model).read_text(encoding="utf-8"))
        models = base["reliability_models"]
        counts = base.get("sample_counts", {})
        base_version = base.get("version")
        if any(row.get("calibration_version") != base_version for row in rows):
            raise SystemExit("confidence rows were not produced with the supplied stage-1 model")
    top1 = [(row.get("candidate_fusion_scores"), row.get("top1_correct")) for row in rows
            if row.get("candidate_fusion_scores") and row.get("top1_correct") is not None]
    coverage = [(row.get("q_top1"), row.get("candidate_covered")) for row in rows
                if row.get("q_top1") is not None and row.get("candidate_covered") is not None]
    temperature = 1.0
    coverage_coefficients = [1.0, 0.0]
    if args.stage == "confidence" and top1:
        def loss(value):
            total = 0.0
            for scores, label in top1:
                peak = max(scores)
                denominator = sum(math.exp(8.0 * (score - peak) / value) for score in scores)
                calibrated = min(1 - 1e-6, max(1e-6, 1.0 / denominator))
                total -= int(bool(label)) * math.log(calibrated) + (1 - int(bool(label))) * math.log(1 - calibrated)
            return total / len(top1)
        temperature = min((0.25 + index * 0.05 for index in range(76)), key=loss)
        coverage_samples = []
        for probability, label in coverage:
            probability = min(1 - 1e-6, max(1e-6, float(probability)))
            coverage_samples.append(([math.log(probability / (1 - probability))], label))
        coverage_model = fit_logistic(coverage_samples, l2=args.l2)
        coverage_coefficients = [coverage_model["weights"][0] / coverage_model["scale"][0],
                                 coverage_model["intercept"] -
                                 coverage_model["weights"][0] * coverage_model["mean"][0] /
                                 coverage_model["scale"][0]]
    digest = hashlib.sha256()
    for path in sorted(Path(args.status_root).rglob("status.jsonl")):
        digest.update(path.read_bytes())
    model = {
        "version": args.version,
        "schema_version": 1,
        "stage": args.stage,
        "reliability_models": models,
        "temperature": temperature,
        "coverage_logistic": coverage_coefficients,
        "l2": args.l2,
        "source_sha256": digest.hexdigest(),
        "source_root": str(Path(args.status_root).resolve()),
        "sample_counts": counts,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "calibration_model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    (output / "calibration_version.txt").write_text(args.version + "\n", encoding="utf-8")
    report = {"status": "passed", "version": args.version, "stage": args.stage, "rows": len(rows),
              "sample_counts": counts, "temperature": temperature,
              "source_sha256": digest.hexdigest()}
    (output / "calibration_fit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
