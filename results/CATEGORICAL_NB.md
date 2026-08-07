# Categorical Naive Bayes workload

`scripts/bench_categorical_nb.py` uses a 12-row, two-feature integer-category
fixture and an independent NumPy reconstruction of per-feature category counts,
Laplace smoothing, class priors, and stable normalization. The FortML release
app writes every query probability and predicted label; unknown-category and
discrete-JVP refusal behavior are covered by the FortML test suite.

The CSV is regenerated with:

```bash
python scripts/bench_categorical_nb.py --fortml ../fortml \
  --output results/categorical_naive_bayes.csv
```

Rows are correctness-gated before timing is recorded. The workload is CPU-only;
resident GPU category tables and differentiable category lookup remain explicit
roadmap work.
