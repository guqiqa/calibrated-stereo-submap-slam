#!/usr/bin/env python3
"""Freeze dataset and calibration identities after Gates A-C pass."""

import argparse
import hashlib
import json
from pathlib import Path
import time


PROTOCOL_VERSION = "c1-c4-p-calibrated-v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--preflight-report", required=True)
    parser.add_argument("--calibration-model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    dataset_path = Path(args.dataset_manifest).resolve()
    preflight_path = Path(args.preflight_report).resolve()
    calibration_path = Path(args.calibration_model).resolve()
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    gates = preflight.get("gate_status", {})
    if not all(gates.get(name) for name in
               ("A_code_protocol", "B_pilot_runtime", "C_labels_calibration")):
        raise SystemExit("Gates A-C must pass before freezing the formal protocol")
    if dataset.get("status") != "passed":
        raise SystemExit("dataset validation did not pass")
    test_layouts = [item for item in dataset.get("layouts", []) if item.get("split") == "test"]
    if len(dataset.get("layouts", [])) < 10 or not test_layouts:
        raise SystemExit("formal dataset requires at least 10 independent layouts and a test split")
    if calibration.get("stage") != "confidence":
        raise SystemExit("only a stage=confidence calibration model can be frozen")
    payload = {
        "frozen": True,
        "protocol_version": PROTOCOL_VERSION,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset_manifest": str(dataset_path),
        "dataset_manifest_sha256": sha256(dataset_path),
        "calibration_model": str(calibration_path),
        "calibration_model_sha256": sha256(calibration_path),
        "calibration_version": calibration.get("version"),
        "preflight_report": str(preflight_path),
        "preflight_report_sha256": sha256(preflight_path),
        "layout_count": len(dataset.get("layouts", [])),
        "test_layouts": [{"family": item.get("family"), "seed": item.get("seed"),
                          "output": item.get("output")} for item in test_layouts],
        "candidate_budget": {"visual": 4, "pose": 3, "topology": 2, "total": 10},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
