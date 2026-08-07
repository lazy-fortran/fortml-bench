# Named MLP parameter layout

This lane checks the stable metadata returned by FortML's
`mlp_t%parameter_layout()` and `parameter_range()` APIs for a dense network
with dimensions `3 -> 4 -> 2`.  The independent NumPy oracle computes each
matrix/vector size with `numpy.prod`, then checks that the one-based packed
ranges are contiguous, non-overlapping, and cover all 26 trainable values.

The probe is linked against the built FortML archive.  It emits every block's
name, kind, one-based range, shape, trainable flag, and buffer flag; the script
fails before writing a passing row if any field differs from the NumPy oracle.
The checked-in CSV therefore records both runtime metadata and the independent
expected metadata, including the named `layer_2.weight` range lookup.

CUDA is a typed `unavailable` row.  Parameter metadata has no resident CUDA
path yet, so the benchmark makes no CPU-to-GPU inference and does not relabel
host metadata as device support.

Run:

```bash
python3 scripts/bench_mlp_parameter_layout.py \
  --fortml ../fortml --output results/mlp_parameter_layout.csv
```

Raw data: [`mlp_parameter_layout.csv`](mlp_parameter_layout.csv).
