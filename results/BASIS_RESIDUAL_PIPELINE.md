# Named residual-sum basis pipeline

The release lane exercises `basis_residual_pipeline_t`, a bounded two-branch
DAG.  Both named sequential branches consume the same dense input and must
produce the same feature shape.  The forward output is the elementwise sum of
the main and residual branches; parameter coordinates are packed in main,
residual order.  CPU value, JVP, VJP, HVP, metadata, and transactional CUDA
refusal behavior are covered by `test_basis_residual_pipeline`.

The benchmark's NumPy oracle independently constructs a degree-two polynomial
main branch and a two-frequency Fourier residual branch.  It checks the
residual sum and a mixed input/log-frequency directional derivative against a
central finite difference before invoking the Fortran gate.  CUDA remains an
explicit `FORTNUM_NOT_IMPLEMENTED` row: the device API leaves all output
buffers untouched because no resident residual executor is linked yet.

Raw results are in [`basis_residual_pipeline.csv`](basis_residual_pipeline.csv)
and are generated with:

```sh
python3 scripts/bench_basis_residual_pipeline.py \
  --fortml ../fortml \
  --output results/basis_residual_pipeline.csv
```
