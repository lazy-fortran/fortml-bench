# K-means benchmark

This lane compares FortML's deterministic dense `kmeans_t` baseline with an
independent NumPy Lloyd implementation on the same 240-by-2 float64 fixture.
Both use three clusters, cyclic seeded initialization (`seed=7`), lowest-index
assignment ties, 100 iterations, and tolerance `1e-8`. The harness checks the
final inertia before retaining FortML fit and transform timings.

The CPU rows are correctness-gated. The CUDA row records the typed
`FORTNUM_NOT_IMPLEMENTED` device contract because no resident clustering kernel
is linked. Empty-cluster and zero-distance derivative boundaries are covered by
the FortML unit test. Reproduce with:

```sh
python3 scripts/bench_kmeans.py --fortml ../fortml \
    --output results/kmeans.csv
```

Raw rows are in [`kmeans.csv`](kmeans.csv). The release application is
`fortml_bench_kmeans` in the FortML checkout.
