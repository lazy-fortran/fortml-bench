# LightGBM-style multiclass classification

This release lane independently replays the bounded two-leaf logistic
OVR updates for arbitrary labels `[-8, 2, 11]`. It checks weighted
normalized staged probabilities, common-prefix validation metadata,
the fixed-tree input JVP/VJP boundary, typed CUDA refusal, and
transactional unknown-label rejection.

| metric | observed | independent oracle |
|---|---:|---:|
| best iteration | 1 | 1 |
| requested estimators | 4 | 4 |
| retained estimators | 1 | 1 |
| weighted validation log-loss | 9.4309471214668528e-01 | 9.4309471214668528e-01 |
| staged probability max error | 0.0000000000000000e+00 | <= 3e-12 |
| probability simplex max error | 2.2204460492503131e-16 | <= 3e-12 |
| input JVP max error | 0.0000000000000000e+00 | <= 3e-12 |
| input VJP max error | 0.0000000000000000e+00 | <= 3e-12 |

All CPU rows agree with the independent oracle. CUDA is recorded as
`unavailable` with the typed resident-histogram refusal. No host
fallback is included in the device result.

Reproduce:

```bash
python -B scripts/bench_lightgbm_multiclass.py \
  --fortml ../fortml --output results/lightgbm_multiclass.csv \
  --report results/LIGHTGBM_MULTICLASS.md
```

FortML revision: `afc17bce530716ceb6552576871a3a73ae758056`
Benchmark revision: `e12f249c158e5459dcef8b51020d45713117de15`
Python 3.14.6, NumPy 2.5.1
