# Dense PCA benchmark

This lane measures centered dense PCA on the deterministic `512 x 16` fixture
used by `fortml_bench_pca`, retaining the first eight components. NumPy's
float64 thin SVD is the independent oracle. It centers rows, applies the same
largest-loading-positive sign convention, checks orthonormality and sorted
explained variance, and times fit and projection separately. scikit-learn's
`PCA(svd_solver="full", whiten=False)` is a contextual reference; component
rows are sign-aligned before comparison with a `1e-6` tolerance because this
fixture has a clustered/ill-conditioned trailing subspace.

The FortML release app runs `fit` and `transform`, checks the component
orthonormality guard, and reports the four-fit average. The current app does
not export its fitted component array, so the FortML row is explicitly a
release-app guard rather than a claim of complete cross-engine array
comparison. A future app revision should write components, mean, explained
variance, and projected rows through an oracle file; the benchmark script will
then tighten the gate to compare those arrays directly.

Run from this repository:

```bash
python3 scripts/bench_pca.py --fortml ../fortml --output results/pca.csv
```

The raw CSV records Python/NumPy/scikit-learn versions, FortML and benchmark
revisions, compiler and flags, status, timing, oracle description, and this
boundary. `unavailable` is retained when the release app or an optional
scikit-learn dependency cannot be built or imported.
