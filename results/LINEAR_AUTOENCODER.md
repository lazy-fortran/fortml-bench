# PCA-initialized linear autoencoder benchmark

This lane measures the tied, centered linear autoencoder initialized from a
rank-eight PCA on the deterministic `512 x 16` fixture. The encoder is the
PCA loading matrix and the decoder is its transpose. NumPy's float64 centered
thin SVD is the independent behavioral oracle; reconstruction RMSE must agree
within `2e-10` before a FortML timing is retained.

The current checked run reports matching RMSE values:

```text
backend         reconstruction_rmse       seconds/operation
numpy_oracle    0.4092698664533236        3.0713e-4
fortml          0.4092698664533234        3.9430e-5
```

The FortML module exposes exact input JVPs for the fixed PCA state. It is
CPU-only today; CUDA is an explicit refusal rather than a host fallback claim
until a resident matrix-product kernel is linked.

Run from this repository:

```bash
python3 scripts/bench_linear_autoencoder.py \
  --fortml ../fortml --output results/linear_autoencoder.csv
```

The CSV records toolchain versions, revisions, timing, oracle, and refusal
boundary. The FortML revision is marked dirty when the source worktree has
uncommitted changes, so release runs should execute after committing the
corresponding source and app.
