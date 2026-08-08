# Random-forest OOB decision probabilities

The raw record is [`random_forest_oob.csv`](random_forest_oob.csv), generated
by [`scripts/bench_random_forest_oob.py`](../scripts/bench_random_forest_oob.py).
The release executable fits a 64-tree seeded CART forest on a deterministic
three-class fixture, then computes OOB decision probabilities and OOB accuracy
from the stored bootstrap-inclusion matrix.

The Python gate independently reconstructs the class labels from the feature
thresholds, requires every row to have at least one OOB tree, and checks the
probability simplex and score. The source test also fits a one-tree/two-row
fixture where every row is in-bag; the OOB method returns
`RANDOM_FOREST_OOB_INSUFFICIENT` and leaves its sentinel output unchanged.
CUDA is an explicit `unavailable` row: no resident OOB tree kernel is linked,
and the typed refusal does not copy or predict on the host.

Run:

```bash
python3 -B scripts/bench_random_forest_oob.py \
  --fortml ../fortml --output results/random_forest_oob.csv
```
