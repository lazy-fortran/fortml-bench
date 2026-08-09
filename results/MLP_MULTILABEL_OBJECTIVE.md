# Weighted multilabel MLP objective

This lane checks the shared multilabel MLP objective against an independent
NumPy implementation. The fixture uses six rows, two features, two indicator
columns, positive per-label class factors, and nonnegative sample weights. The
Fortran probe emits the fitted packed state, weighted BCE products, direct-L2
products, positive log-L2 products, and bounded FortOpt results. NumPy rebuilds
the logits, weighted reductions, gradients, and mixed HVPs from that packed
state. It does not call FortML internals.

| phase | device | status | error or result |
|---|---|---|---:|
| direct L2 products | CPU | pass | see CSV |
| positive log-L2 products | CPU | pass | see CSV |
| bounded L-BFGS-B | CPU | pass | direct and log coordinates converged |
| resident objective graph | CUDA | unavailable | typed status 3 |

The CUDA row is `FORTNUM_NOT_IMPLEMENTED`. The classifier has no resident
multilayer CUDA graph, so the benchmark does not report host computation as
GPU execution.

Run the release lane from this repository:

```bash
FO_SCAN_FALLBACK=regex python -B scripts/bench_mlp_multilabel_objective.py \
  --fortml ../fortml --output results/mlp_multilabel_objective.csv
```

The script builds a dedicated release probe, records clean source and
benchmark revisions, and writes four provenance rows. The raw data is in
[`mlp_multilabel_objective.csv`](mlp_multilabel_objective.csv).
