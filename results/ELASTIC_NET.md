# Weighted elastic-net regression

This release lane uses 96 rows, six features, three outputs, nonnegative
sample weights, an unregularized intercept, `alpha=0.21`, and `l1_ratio=0.43`.
The NumPy implementation independently solves each output with weighted
coordinate descent.  Before timing, it checks every coefficient, matrix and
vector prediction, packed-parameter JVP, parameter VJP, and input VJP against
the complete array emitted by the FortML release app.  Derivative products are
for the fixed fitted state; the nonsmooth active-set and convergence decisions
of the fit are not differentiated.

Run it from `fortml-bench` with:

```text
.venv/bin/python -B scripts/bench_elastic_net.py \
  --fortml ../fortml --output results/elastic_net.csv
```

The protocol rejects checksum-only output and requires every element in all
seven arrays. The current run has 14 passing CPU rows (seven NumPy oracle and
seven FortML) plus seven explicit CUDA `unavailable` refusal rows:

| Backend | Fit matrix s | Fit vector s | Predict matrix s | Predict vector s | JVP s | VJP parameter s | VJP input s | Maximum oracle error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NumPy oracle | 1.501329e-3 | 3.392496e-4 | 2.640375e-6 | 1.345417e-6 | 4.488458e-6 | 3.650207e-6 | 2.745207e-6 | 0 |
| FortML | 2.966258e-5 | 7.775917e-6 | 8.190417e-7 | 5.618750e-7 | 2.078500e-6 | 1.830125e-6 | 1.830125e-6 | 7.105e-15 |

The CSV records compiler flags and both repository revisions. The current
FortML rows use revision `877a27b`; the benchmark revision is recorded per run
in the CSV. CUDA rows are capability refusals, not timings.
