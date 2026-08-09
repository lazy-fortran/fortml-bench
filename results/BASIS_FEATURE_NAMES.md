# Semantic basis feature labels

This lane checks the optional semantic output-label contract for `basis_map_t`
and its horizontal, sequential, and column-selecting pipeline consumers. The
independent NumPy fixture assembles polynomial and Fourier values directly and
compares the expected qualified names. The Fortran test additionally checks
duplicate-name refusal is transactional and that metadata does not alter
values.

Run from this repository:

```bash
python -B scripts/bench_basis_feature_names.py \
  --fortml ../fortml --output results/basis_feature_names.csv \
  --report results/BASIS_FEATURE_NAMES.md
```

## Results

| Phase | Backend/device | Result | Evidence |
| --- | --- | --- | --- |
| Independent labels and values | NumPy | Pass | 6 qualified labels, direct trigonometric/polynomial construction, and maximum error `0.000e+00` |
| Fortran semantic-label gate | FortML / CPU | Pass | `test_basis_feature_names`, transactional duplicate refusal, and composition metadata |

FortML revision: `218ec64052881924faefcc90e045946a445f988c`. Benchmark revision: `b1757f38dc40af922770e3a0303f524f21e068b9`. Python 3.14.6, NumPy 2.5.1, GNU Fortran `-O3`.

This is a metadata and correctness lane. It does not claim resident GPU
transform throughput. Structural pipeline persistence, sparse feature views,
and device-resident transforms remain explicit roadmap boundaries.
