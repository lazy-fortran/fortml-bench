# OVO RBF SVM

This correctness-gated lane compares FortML's deterministic sorted-label
one-vs-one finite-basis RBF-SVM wrapper with an independent NumPy/SciPy
replay. SciPy solves each class pair separately with L-BFGS-B on the same
squared-hinge RKHS objective, then evaluates pair margins and the documented
normalized pairwise-vote probability map. The gate checks labels, sorted
class metadata, pair decisions, simplex probabilities, predictions, and
packed pair-specific parameters. Packed-coordinate drift is recorded because
the dense kernel basis can be mildly ill-conditioned; behavioral errors are
gated at `2e-5` for margins and `2e-6` for probabilities.

The 36-row, three-class workload completes with 100% label accuracy. The
recorded FortML fit margin error is `1.281e-05`, probability error is
`1.038e-06`, and prediction throughput is `1.280e-05` seconds per call on the
captured CPU run. CUDA is represented by a typed `FORTNUM_NOT_IMPLEMENTED`
row; no host fallback is counted as GPU execution.

Raw data: [`ovo_rbf_svm.csv`](ovo_rbf_svm.csv). Reproduce with:

```bash
python scripts/bench_ovo_rbf_svm.py --fortml ../fortml
```
