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
seven arrays.  The current run has 14 passing rows (seven NumPy oracle and
seven FortML):

| Backend | Fit matrix s | Fit vector s | Predict matrix s | Predict vector s | JVP s | VJP parameter s | VJP input s | Maximum oracle error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NumPy oracle | 1.452070e-3 | 3.413210e-4 | 2.591543e-6 | 1.337125e-6 | 4.430874e-6 | 3.620541e-6 | 2.365707e-6 | 0 |
| FortML | 3.296338e-5 | 9.701167e-6 | 9.104583e-7 | 4.654583e-7 | 1.960375e-6 | 1.785833e-6 | 1.785833e-6 | 7.105e-15 |

The CSV records compiler flags and both repository revisions. The current
FortML rows use revision `263b6b4`; the benchmark revision is recorded per run
in the CSV.
