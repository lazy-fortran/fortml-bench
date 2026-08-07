# Multilabel classification metrics

`bench_multilabel_metrics.py` checks precision, recall, and F1 for a dense
binary indicator matrix against an independent NumPy TP/FP/FN oracle.  The
fixture records micro, macro, and samples averages, plus probability
thresholding at `0.5` with the documented `>=` rule.  Zero-support terms use
the explicit zero-division policy (`0`), so the report does not hide an
undefined score behind a NaN or warning.

Run the correctness-gated CPU lane with:

```bash
python -B scripts/bench_multilabel_metrics.py \
  --fortml ../fortml --output results/multilabel_metrics.csv
```

The CSV stores all twelve CPU metric values and a typed CUDA capability row.
The latter is `unavailable` until resident multilabel reduction kernels are
linked; the FortML device entry points return a typed refusal and the harness
never substitutes a host timing.  ROC-AUC is intentionally a separate lane
(`bench_roc_auc.py`) so its binary and one-vs-rest fixtures retain accurate
sample/class dimensions and independent provenance.
