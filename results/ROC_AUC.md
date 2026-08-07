# ROC-AUC classification metrics

`bench_roc_auc.py` gates binary and one-vs-rest multiclass ROC area against an
independent NumPy pairwise oracle.  Scores above a negative score receive one
concordance point and exact ties receive one half; binary labels use arbitrary
integers and the multiclass lane reports the unweighted macro average.  The
fixture includes an intentional binary tie and records per-class OVR values.

The FortML CPU lane reports correctness and per-call timings for
`classification_roc_auc` and `classification_roc_auc_ovr`.  Degenerate class
support, nonfinite scores, malformed weights, and more-than-two binary labels
are typed domain errors.  The CUDA row is explicitly `unavailable` until a
resident ranking/reduction kernel is linked; no host fallback is hidden.

Run:

```bash
python scripts/bench_roc_auc.py --fortml ../fortml
```

`results/roc_auc.csv` records source/benchmark revisions, compiler metadata,
oracle provenance, and the device-contract result.
