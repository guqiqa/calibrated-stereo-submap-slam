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

The online node publishes scene-space candidate IDs but always leaves
`candidate_covered: null` and `coverage_label_available: false`. The experiment
recorder joins `/sim/odom` with `submaps.json` in an evaluation side channel and
writes `legal_submaps`, `candidate_covered`, `top1_correct`, and a coverage reason.
These annotations must never feed the online selector.

Candidate support is the deterministic round-robin union of `Cv(4)`, `Cp(3)`,
`Cg(2)`, and the current submap, capped at `K=10`. C1 ranks this common support
using visual evidence only; C4 uses fixed weights; P uses calibrated reliability
gates and log-likelihood fusion.

Calibration metrics are separated by meaning: `q_top1` is evaluated against
`top1_correct`, while `q_cov` is evaluated against `candidate_covered`. Using the
coverage label as the Top-1 label is prohibited.

`q_cov` uses the frozen `candidate-coverage-v1` logistic model. Its 12 ordered
inputs are candidate count; visual, pose, and topology source counts; maximum
fusion score and Top-1/Top-2 fusion margin; maximum visual, pose, and topology
scores; and three evidence-unavailable flags. A stage-1 reliability run records
these features but publishes `q_cov: null`. Only a stage-2 `confidence` model
with the exact feature names and width may publish a calibrated `q_cov`.

Formal query runs must use `map_database_mode=load` with a database produced by
an independent `mapping_route` run in `map_database_mode=build`. Query runs keep
that database frozen; online keyframes may support VO but cannot mutate the loaded
retrieval map.

## Typed query protocol

Mapping and query are separate captures at least 5 s apart, with no image,
timestamp, feature-cache, or mutable-map reuse. The protocol does not impose a
single spatial/viewpoint conjunction on every frame. `mapped_same_view` has no
2 m or 30 degree gate; `mapped_reverse_view` requires at least 120 degrees only
on preregistered retrieval-opportunity frames; `mapped_crossing` requires at
least 30 degrees only on those opportunity frames. `cold_start` is evaluated on
relocalization frames and `none_of_the_above` requires an empty legal-map set.
All frames remain in trajectory evaluation and retain nearest-mapping distance
and viewpoint diagnostics.

Scene submaps are heterogeneous: 5 m corridor maps with 1.5 m overlap, 7 m hall
grid maps with 1.5 m overlap, and 3 m-radius junction maps. Semantic scene IDs
are mapped explicitly to runtime IDs; integer equality is never assumed.

## Rejection vocabulary

The stable reasons are `candidate_set_empty`, `visual_evidence_unavailable`,
`geometry_verification_failed`, `low_calibrated_confidence`, and
`all_evidence_low_reliability`. Smoke runs without a frozen confidence model use
`low_retrieval_confidence` instead of claiming calibrated confidence. Empty rejection
text means no rejection occurred.

The labeled fixture can be used to validate the complete file contract:

```bash
python run_protocol_fixture.py --method p --output /tmp/protocol_status.jsonl
python evaluate_retrieval_protocol.py --status-root /tmp --output /tmp/protocol_analysis
```
