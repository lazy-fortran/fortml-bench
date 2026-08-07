# Binary classification workload

`scripts/bench_classification.py` compares the FortML binary logistic slice
with scikit-learn on one deterministic row-major fixture. Labels are `-3` and
`7`, generated from a known linear score. The NumPy-generated labels are the
behavioral oracle. The workload checks training accuracy and probability
normalization before recording fit and prediction timings. The FortML driver
also writes every target, predicted label, and class probability. The Python
harness computes accuracy, mean log loss, and the complete confusion matrix
from those arrays with NumPy, then requires agreement with `sklearn.metrics`
before it writes a `metrics` row.

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

The recorded NumPy/scikit checks pass. With class order `[-3, 7]`, scikit-learn
has accuracy `1.0`, log loss `0.0028016188190255276`, and confusion matrix
`[[481, 0], [0, 543]]`. FortML has accuracy `0.982421875`, log loss
`0.15186232408879013`, and confusion matrix `[[468, 13], [5, 538]]`.
Metric rows carry no timing, and the four existing fit/predict timings were
left unchanged when these checks were added.

These quality numbers validate each backend's full predictions. They are not
a matched estimator-quality comparison: the current FortML driver uses
`l2=0.1`, while the scikit lane sets `C=1/L2` with `L2=0.001`. Matching the
regularization objective is separate work and requires a new timing record.
