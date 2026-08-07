# SGD/Nesterov training and differentiable imputation

`scripts/bench_training.py` produces
[`training_imputer.csv`](training_imputer.csv).  It covers two production
MLP-training configurations and all three `simple_imputer_t` strategies:

- `sgd` uses full-batch momentum SGD with learning rate `0.01`, momentum
  `0.8`, 24 epochs, and an L2 coefficient of `1e-4`.
- `nesterov` uses the same fixture and hyperparameters with the exact Nesterov
  look-ahead update.
- `mean`, `median`, and `constant` fit per-feature statistics on a 12-by-4
  matrix containing NaNs.  Transform, input JVP, and input VJP are checked
  entry by entry; missing entries have zero derivative and observed entries are
  identity.

The NumPy implementation is independent of FortML.  It reproduces the MLP
initializer, tanh forward pass, MSE gradient, momentum recurrence, and L2
penalty, then separately computes the imputer statistics and piecewise
derivative rules.  The release app must write complete arrays before a timing
row is accepted.  The current CPU record has 32 rows (six NumPy MLP oracle
rows, two FortML MLP rows, and four rows per imputer strategy per backend), with
FortML maximum oracle error below `4e-15`.

Run the workload serially:

```bash
.venv/bin/python -B scripts/bench_training.py \
    --fortml ../fortml --output results/training_imputer.csv
```

The harness invokes `fortml_bench_training` after `fo build --flag -O3` and
records compiler flags, source revisions, package versions, repetitions, and
correctness error.  CUDA is not claimed: these trainer and imputer paths are
host-bound in the current FortML release and a device row is not inferred from
the CPU measurement.
