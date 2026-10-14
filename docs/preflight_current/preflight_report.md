# Formal Experiment Preflight

Formal test ready: **False**

## A_code_protocol

- [x] scene ID interface: node exposes explicit runtime-to-scene ID alignment
- [x] fixed candidate quotas: Kv=4, Kp=3, Kg=2 and deterministic merge are present
- [x] no online truth subscription: visual node source must not subscribe to simulation truth

## B_pilot_runtime

- [x] C1/C4/P observed: observed methods: ['c1', 'c4', 'p']
- [ ] six pilot layouts: observed 1 scene/map versions
- [ ] five query protocols: observed routes: ['query_normal_01']
- [ ] obstacle variant: observed obstacle versions: []
- [ ] independent frozen mapping: query manifests using map_database_mode=load: 0/3
- [ ] mapping/query separation: status=failed
- [x] tracking continuity: rows=272, max consecutive failures=0

## C_labels_calibration

- [x] coverage labels complete: 263/263 evaluable rows labeled
- [x] common ID space: candidate and legal IDs must both be scene strings
- [ ] calibration frozen: no calibration model supplied
- [ ] q_top1/q_cov available: both probabilities are required on every evaluable frame

## D_formal_test_freeze

- [ ] full test dataset frozen: no full dataset manifest supplied

mapping/query 5 s, 2 m and 30 degree separation is a conjunctive preregistered gate; failure blocks formal calibration/test
