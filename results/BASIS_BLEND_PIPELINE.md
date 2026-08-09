# Learned basis fan-in

This release lane exercises `basis_blend_pipeline_t` with two named branches.
A degree-two polynomial branch and a Fourier branch both produce four features.
The graph combines them with two learned scalar weights. The packed parameter
vector contains each weight followed by that branch's basis parameters.

The NumPy oracle constructs the polynomial and Fourier values directly. It
derives a joint directional derivative over both mixing weights, both Fourier
log-frequencies, and every input coordinate. An analytic reverse product must
satisfy the JVP/VJP adjoint identity. The Fortran test separately checks the
weight gradient, HVPs against central differences, stable metadata, and rollback
after invalid append or parameter updates.

The timed row measures a 4,096-row CPU transform after the value and derivative
gates pass. CUDA and OpenACC appear as unavailable rows. Their device calls
return `FORTNUM_NOT_IMPLEMENTED` and preserve caller-owned output buffers. No
host fallback is included in either accelerator row.

Raw results are in [`basis_blend_pipeline.csv`](basis_blend_pipeline.csv) and
are generated with:

```sh
python3 scripts/bench_basis_blend_pipeline.py \
  --fortml ../fortml \
  --output results/basis_blend_pipeline.csv
```
