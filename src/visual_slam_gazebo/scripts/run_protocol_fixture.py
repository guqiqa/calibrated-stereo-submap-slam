#!/usr/bin/env python3
"""Generate a small labeled C1/C4/P fixture for end-to-end protocol checks."""

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from submap_retrieval_protocol import Candidate, Evidence, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=("c1", "c4", "p"), default="p")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(12):
        correct = 3 if index % 3 else 4
        candidates = [
            Candidate(correct, .92, .88, .91, ["visual", "pose", "topology"], True, 35, 1.2),
            Candidate(8, .61, .74, .60, ["visual", "pose"], True, 28, 1.8),
            Candidate(11, .55, .42, .58, ["visual", "topology"], index == 9, 12, 4.5),
        ]
        evidence = {
            "visual": Evidence(True, .9, .9, "fixture_visual"),
            "pose": Evidence(index not in (4, 8), .8, .75, "fixture_pose"),
            "topology": Evidence(index not in (5, 9), .85, .8, "fixture_topology"),
        }
        decision = run(args.method, candidates, evidence, candidate_covered=correct in decision_ids(candidates),
                       min_probability=.34)
        row = decision.to_status_fields()
        row["ground_truth_submap"] = correct
        row["candidate_correct"] = correct in row["candidate_ids"]
        row["q_top1"] = 1.0 if row["selected_submap"] == correct else 0.0
        rows.append(row)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def decision_ids(candidates):
    return [candidate.submap_id for candidate in candidates]


if __name__ == "__main__":
    main()
