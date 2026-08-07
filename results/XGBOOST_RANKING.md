# XGBoost pairwise-ranking benchmark

`bench_xgboost_ranking.py` evaluates the `rank:pairwise` logistic loss and
positive diagonal Hessian with an independent NumPy oracle.  A row in a
different query is included to verify query isolation.  The FortML test covers
the public pairwise loss/derivative API, a depth-one fit whose prediction
margins order a two-item query, and singleton-query refusal.

Run:

```bash
python3 -B scripts/bench_xgboost_ranking.py \
  --fortml ../fortml --output results/xgboost_ranking.csv
```

This is a CPU correctness lane.  Native CUDA ranking trees and pairwise
reductions remain a typed `unavailable` row; no host fallback is hidden.
