# Conditional basis feature union

This release lane exercises `conditional_basis_pipeline_t`, a parallel
interval-routed feature union over the existing selected-column basis
primitive. Two named Fourier branches consume the same signal column: `left`
is active on `[-2,0)` and `right` on `[0,2)`. Features are concatenated in
branch order and inactive rows are zero. The source-side
`test_conditional_pipeline` gate covers value, central finite-difference JVP,
VJP adjoint, mixed HVP, metadata offsets, transactional append/parameter/schema
updates, endpoint derivative refusal, and output-preserving device dispatch.

The companion NumPy fixture is independent of FortML. It reconstructs the
interval masks and Fourier formulas, checks the mixed JVP against a central
finite difference, verifies the VJP inner-product identity, and compares the
analytic HVP to a finite difference of the independent reverse product. The
largest independent error in the checked run was `2.10e-6` (HVP finite
difference, `h=2e-6`); JVP and adjoint errors were `5.74e-11` and `4.55e-12`.
The CPU release workload processed 2,048 rows and four features in
`4.05e-5 s` per transform operation. These are correctness-gated timings, not
cross-framework throughput claims.

CUDA is recorded as `unavailable`: the device API returns
`FORTNUM_NOT_IMPLEMENTED` and preserves every caller-owned output buffer until
a resident route-mask executor is linked. Route endpoints are value-defined by
the half-open rule but return `FORTNUM_DOMAIN_ERROR` for derivative products.

Raw records are in [`conditional_pipeline.csv`](conditional_pipeline.csv).
Reproduce the run with:

```sh
python3 -B scripts/bench_conditional_pipeline.py \
  --fortml ../fortml \
  --output results/conditional_pipeline.csv
```

The checked source revision is recorded in the CSV's `fortml_revision` field;
the benchmark revision is recorded in `benchmark_revision`.
