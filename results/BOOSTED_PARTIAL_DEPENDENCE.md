# Boosted-tree partial dependence

This lane checks weighted one-feature partial dependence and individual
conditional expectation values for the XGBoost-style and LightGBM-style tree
estimators. The four-row fixture fits one Newton stump. Its leaf predictions
are `5/3` and `25/3`. Weights `[1, 1, 1, 3]` give a partial-dependence value of
`55/9` when the intervention changes an unused feature.

NumPy constructs the full expected PDP and ICE arrays from those analytic leaf
values and weights. A separate check compares transformed binary predictions
with the logistic transform of raw margins. CUDA rows record the typed refusal
because no resident boosted-tree PDP kernel is linked.

Reproduce:

```bash
.venv/bin/python -B scripts/bench_boosted_partial_dependence.py   --fortml ../fortml --output results/boosted_partial_dependence.csv   --report results/BOOSTED_PARTIAL_DEPENDENCE.md
```

Raw data: [`boosted_partial_dependence.csv`](boosted_partial_dependence.csv).
