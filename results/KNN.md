# Deterministic k-nearest-neighbor benchmark

The kNN lane uses 96 training rows, 48 query rows, three features, three
arbitrary integer classes, and `k=7`. The NumPy oracle independently computes
squared Euclidean distances, stable lexicographic neighbor order (distance
then original training-row index), sorted class columns, and both uniform and
inverse-distance votes. It checks accuracy, the complete probability squared
norm, the prediction-label checksum, and the probability-mass sum before any
FortML timing is retained.

The release target is `fortml_bench_knn`. Its output is parsed into separate
uniform and distance rows. kNN neighbor selection is discrete, so FortML's
input JVP/VJP operations intentionally return a structured refusal; the
benchmark does not treat a fabricated zero derivative as evidence.

Run:

```bash
.venv/bin/python -B scripts/bench_knn.py \
  --fortml ../fortml --output results/knn.csv
```

Rows include compiler flags, source revisions, and independent oracle errors.
GPU/device rows are explicit refusals until a device-resident neighbor-search
backend exists.
