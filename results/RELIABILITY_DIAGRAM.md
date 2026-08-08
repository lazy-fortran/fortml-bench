# Reliability diagram benchmark

This lane checks `classification_reliability_diagram` on a deterministic
512-row, three-class fixture with ten equal-width confidence bins and positive
sample weights.  The reference computes normalized row confidence, first-class
ties, weighted bin masses, and weighted mean confidence/accuracy independently
with NumPy.  Every bin is compared before the timing row is retained; empty
bins must return zero means and zero mass.

The recorded CPU run used gfortran `-O3` and reports a FortML curve time of
`9.75e-06` seconds per 512-row operation with maximum absolute oracle error
`0.0`.  The NumPy oracle row is retained for provenance, and the CUDA row is
an explicit `unavailable` capability record because no resident metric kernel
is linked.

Reproduce it with:

```bash
python3 -B scripts/bench_reliability_diagram.py \
    --fortml ../fortml --output results/reliability_diagram.csv
```

The machine-readable rows are in
[`reliability_diagram.csv`](reliability_diagram.csv).
