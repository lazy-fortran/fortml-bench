# XGBoost text persistence

This release-fixture lane fits a four-tree squared-loss XGBoost-style model
with validation diagnostics, learned missing routing, feature subsampling, and
a monotone constraint. It saves the fitted ensemble to the versioned
`FORTML_XGBOOST_TEXT` schema and loads it into a fresh model. The harness checks
predictions, raw margins, staged outputs, estimator metadata, and constraint
metadata before and after the round trip.

For an independent semantic oracle, the Python harness parses the text schema
itself and walks every serialized node using the query fixture. Thus the
round-trip check does not merely compare two FortML calls. All CPU outputs
agree exactly at the recorded precision. CUDA is a typed refusal because the
tree predictor has no resident device kernel in this release.

| workload | device | status | max absolute error | timing |
|---|---|---:|---:|---:|
| save/load round trip | CPU | pass | see CSV | see CSV |
| independent serialized-tree walk | CPU | pass | see CSV | see CSV |
| resident tree prediction | CUDA | unavailable | — | — |

Run:

```bash
python -B scripts/bench_xgboost_serialization.py \
  --fortml ../fortml --output results/xgboost_serialization.csv
```

Raw data: [`xgboost_serialization.csv`](xgboost_serialization.csv).
