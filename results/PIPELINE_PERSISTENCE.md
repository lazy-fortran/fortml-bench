# Basis-pipeline persistence benchmark

This release lane exercises the versioned host state dictionary for a fitted
horizontal `basis_pipeline_t` with a 64-sample, one-input fixture. The Fortran
release app saves a two-stage polynomial/Fourier pipeline, loads it onto an
equivalent fitted topology, and compares transformed features. The source test
also checks value/JVP/VJP/HVP equivalence, names, one-based offsets,
malformed-input transactionality, and the typed CUDA refusal.

The Python harness independently reconstructs
`[x, x**2, sin(0.8*x), cos(0.8*x)]` with NumPy and checks the emitted feature
and metadata counts before recording timing. It does not use FortML internals
as an oracle.

| Phase | Device | Status | Max error | Time (s) |
| --- | --- | --- | ---: | ---: |
| Save/load round trip | CPU | pass | 0.0 | 1.35746e-4 |
| Independent feature oracle | CPU | pass | 2.22045e-16 | 1.35746e-4 |
| Resident serialization | CUDA | unavailable | n/a | n/a |

Raw rows are in [`pipeline_persistence.csv`](pipeline_persistence.csv). The
rows were generated with FortML `ce2ff195ddc0edbd2c498ccbf24370eb730f9b4e`
and benchmark harness `69036e7886fe1733586f43a49ed60f17cf2396ed`, using GNU
Fortran `-O2`, Python 3.14.6, and NumPy 2.5.1:

```sh
python -B scripts/bench_pipeline_persistence.py \
  --fortml ../fortml \
  --output results/pipeline_persistence.csv
```

The CUDA row is a typed `FORTNUM_NOT_IMPLEMENTED` capability result. It is not
an end-to-end GPU persistence claim and does not hide a host transfer.
