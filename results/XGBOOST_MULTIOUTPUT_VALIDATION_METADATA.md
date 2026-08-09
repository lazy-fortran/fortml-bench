# Multi-output tree validation metadata

This lane compares the multi-output XGBoost-style and LightGBM-style adapters
with an independent NumPy two-leaf Newton-stump oracle. The fixture has eight
rows, one feature, and two regression targets. The inverse validation targets
make the first fitted round the best round for both outputs. The analytic
validation losses are `[40.5, 6.48]`, and two-round patience sets both
`early_stopped` flags.

The release app records each output's best iteration, validation loss, and
early-stop flag. The CUDA row is a typed refusal because resident
multi-output tree state is not linked.

Reproduce:

```bash
.venv/bin/python -B scripts/bench_xgboost_multioutput_validation_metadata.py \
  --fortml ../fortml --output results/xgboost_multioutput_validation_metadata.csv \
  --report results/XGBOOST_MULTIOUTPUT_VALIDATION_METADATA.md
```

Raw data: [`xgboost_multioutput_validation_metadata.csv`](xgboost_multioutput_validation_metadata.csv).
