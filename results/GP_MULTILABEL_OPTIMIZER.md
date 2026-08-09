# Shared multilabel Laplace-GP hyperparameter optimization

This release lane checks the common-kernel outer objective for
`gp_multilabel_classification_t`. Two independent binary Laplace heads are
fitted on the weighted indicator fixture. The NumPy oracle keeps both fitted
Newton modes fixed, solves each candidate prior system, and evaluates the
negative summed mode posterior with analytic RBF log-parameter contractions.
The FortML probe must agree on the value, gradient, directional JVP, VJP, and
central finite-difference objective check before the bounded FortOpt
L-BFGS-B result is accepted.

The model uses the shared vector `[log(1.3), log(0.75)]`, jitter `1e-7`,
weights `[1,.9,1.1,1,.8,1.2,1,1.1,.9,1]`, and uniform optimizer bounds
`[-1,1]`. `set_shared_parameters` is tested transactionally through the
source unit oracle; malformed vectors leave all label heads unchanged.

CUDA is recorded as an explicit unavailable capability (`device_supported` is
false). No host fallback or GPU timing is reported because resident Laplace
factorizations and the FortOpt objective graph are not linked.

Run from the benchmark repository with:

```bash
python -B scripts/bench_gp_multilabel_optimizer.py \
  --fortml ../fortml --output results/gp_multilabel_optimizer.csv
```

Raw rows are in [`gp_multilabel_optimizer.csv`](gp_multilabel_optimizer.csv).
