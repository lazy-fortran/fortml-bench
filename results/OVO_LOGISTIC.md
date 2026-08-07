# One-vs-one logistic benchmark

The release workload uses 256 rows, six features, four arbitrary integer
classes, and six lexicographically ordered pair estimators. FortML aggregates
the positive and negative pair probabilities as votes and divides by the six
pair models. This is the declared FortML policy. scikit-learn's pairwise
coupling probabilities are reported only as contextual results.

Run it from `fortml-bench` with:

```text
.venv/bin/python -B scripts/bench_ovo_logistic.py \
  --fortml ../fortml --output results/ovo_logistic.csv
```

The NumPy lane independently solves each regularized binary logistic problem
with Newton updates, checks the complete FortML output arrays, and verifies
probability normalization and labels. The current run has six passing CPU rows
plus six explicit CUDA `unavailable` refusal rows:

| Backend | Fit seconds | Predict seconds | Accuracy | Maximum oracle error |
| --- | ---: | ---: | ---: | ---: |
| NumPy oracle | 2.885669e-3 | 2.756152e-3 | 0.43359375 | 0 |
| FortML | 6.820250e-4 | 1.570930e-5 | 0.43359375 | 7.573e-8 |
| scikit-learn context | 3.197886e-2 | 1.408835e-3 | 0.4375 | not reported |

The CSV records compiler, package, repository, and benchmark revisions. The
current FortML rows use revision `877a27b`; the benchmark revision is recorded
per run in the CSV. CUDA rows are capability refusals, not timings.
