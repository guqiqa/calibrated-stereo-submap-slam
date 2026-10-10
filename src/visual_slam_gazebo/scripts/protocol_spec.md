# C1/C4/P Protocol Contract

The protocol is implemented in `submap_retrieval_protocol.py` and is ROS-independent.
An online ROS adapter may call `run(method, candidates, evidences)` once per query
frame and merge `Decision.to_status_fields()` into `/visual_slam/status_sim`.

## Methods

- `C1` (`c1`): visual candidate ranking followed by geometric verification. Pose and
  topology are disabled and must be reported as unavailable.
- `C4` (`c4`): fixed weights `0.50/0.30/0.20`; unavailable evidence is removed and
  the remaining weights are renormalized.
- `P` (`p`): reliability-weighted log likelihood. An unavailable evidence term is
  removed. If every available term is below the reliability threshold, the result is
  `relocalize`, not an accepted switch.

## Required per-query fields

`candidate_ids`, `candidate_covered`, `coverage_label_available`,
`geometry_verified`, `reject_reason`, `q_top1`, `q_cov`, and
`evidence_available`. `candidate_covered` is `null` unless an independent mapping /
query ground-truth label exists. `/sim/odom` is not an online label source.

## Offline metrics

`evaluate_retrieval_protocol.py` reads status logs and writes
`retrieval_events.jsonl`, `retrieval_summary.json`, `calibration_metrics.json`, and
`risk_coverage.csv`. Legacy logs are replayed for diagnostics, but report ECE,
Brier, NLL, and coverage as unavailable when labels or calibrated probabilities are
missing.

## Online mode mapping

- `c0`: global-map baseline; it does not claim submap retrieval.
- `c1`: visual Top-4 candidates and geometric verification only.
- `c4`: union of visual Top-4, pose Top-3, and topology Top-2 candidates,
  followed by fixed `0.50/0.30/0.20` fusion.
- `p`: the same candidate budget followed by reliability-gated log-likelihood fusion.
- Legacy aliases remain readable: `global -> c0`, `joint -> c4`; `fixed` remains
  `legacy_fixed` and is not silently renamed to C1.

The online node always publishes `candidate_covered: null` and
`coverage_label_available: false`. Independent mapping/query ground truth is joined
offline. `/sim/odom` must never feed the online selector.

## Rejection vocabulary

The stable reasons are `candidate_set_empty`, `visual_evidence_unavailable`,
`geometry_verification_failed`, `low_calibrated_confidence`, and
`all_evidence_low_reliability`. Empty rejection text means no rejection occurred.

The labeled fixture can be used to validate the complete file contract:

```bash
python run_protocol_fixture.py --method p --output /tmp/protocol_status.jsonl
python evaluate_retrieval_protocol.py --status-root /tmp --output /tmp/protocol_analysis
```
