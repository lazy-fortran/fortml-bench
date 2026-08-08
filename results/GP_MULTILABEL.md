# Multilabel Laplace-GP benchmark

This lane covers two independent binary Laplace-GP heads over a dense
10-by-2 indicator fixture.  The NumPy oracle reconstructs the weighted Newton
mode, jittered RBF covariance, posterior variance, MacKay logistic predictive
map, and a central finite difference in the five query coordinates.  The
FortML release probe additionally exercises packed per-label products and
thresholded indicator prediction before recording timing rows.

Run it with:

```bash
python -B scripts/bench_gp_multilabel.py \
  --fortml ../fortml --output results/gp_multilabel.csv
```

The source app and independent Fortran behavioral gate are pinned in the CSV;
the benchmark revision is pinned separately.  A clean run records four rows:
the independent NumPy oracle, FortML fit, FortML prediction/derivative, and a
typed CUDA-unavailable contract row.  The CUDA row is not a CPU timing claim:
the resident binary Laplace states, solves, and multilabel reduction are not
linked, so the app must return `FORTNUM_NOT_IMPLEMENTED` without host staging.

The probability and input-JVP error threshold is `4e-6`; predictions must be
complete binary indicators.  The fixture's positive probabilities are not
simplex-normalized across labels, matching multilabel semantics rather than
multiclass OVR coupling.
