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
| NumPy oracle | 1.438168e-3 | 3.490831e-4 | 2.492210e-6 | 1.337083e-6 | 4.399124e-6 | 3.623502e-6 | 2.458374e-6 | 0 |
| FortML | 2.896292e-5 | 7.588042e-6 | 6.829583e-7 | 4.375000e-7 | 1.774167e-6 | 1.672750e-6 | 1.672750e-6 | 7.105e-15 |

The CSV records compiler flags and both repository revisions.  This first
shared-workspace measurement carries `+dirty` provenance while other release
lanes are being finalized; rerun after those commits for a clean revision
stamp.
