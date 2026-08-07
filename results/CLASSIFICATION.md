# Binary classification workload

`scripts/bench_classification.py` compares the FortML binary logistic slice
with scikit-learn on one deterministic row-major fixture. Labels are `-3` and
`7`, generated from a known linear score. The NumPy-generated labels are the
behavioral oracle. The workload checks training accuracy and probability
normalization before recording fit and prediction timings.

Run it from this repository with:

```bash
.venv/bin/python -B scripts/bench_classification.py \
  --fortml ../fortml --output results/classification_workloads.csv
```

The CSV records status and an explicit refusal or unavailable row when a
reference package or compiler is missing. Timings are single-process CPU
measurements. They are not comparable across machines until two release
baselines have been recorded with the same compiler, flags, and dependency
revisions.
