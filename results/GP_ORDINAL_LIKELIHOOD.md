# Native ordinal GP likelihood products

This release lane covers the backend-independent ordered-logit and
ordered-probit likelihood primitive in `fortml_gp_ordinal_classification`.
The fixture has five latent scores, three ordered classes, two cut points,
and 100,000 repetitions per timed product. The NumPy oracle computes the
category CDF differences and density derivatives independently. It checks
value, joint JVP, VJP norm, and the full directional HVP norm before retaining
the release rows.

| Likelihood | Value | JVP | VJP norm | HVP norm |
| --- | ---: | ---: | ---: | ---: |
| Ordered logistic | -5.003477779364477 | -0.0819943130044955 | 2.61355192343308 | 0.417725623724659 |
| Ordered probit | -4.672378236044239 | -0.00320358478011153 | 3.51616553575807 | 0.691927168466301 |

All release checks passed. Maximum checksum errors were below `3e-10` for
the value, JVP, VJP, and HVP rows. The CPU timings are approximately
`0.18--0.35 microseconds` per operation for this small fixture. The focused
behavioral oracle `test_gp_ordinal_likelihood` passed for both likelihoods,
including central-difference JVP/HVP checks, VJP/JVP adjoint duality,
transactional malformed-threshold refusal, and device capability checks.

CUDA is recorded as an explicit unsupported capability. The value, JVP, VJP,
and HVP device entry points return `FORTNUM_NOT_IMPLEMENTED` and preserve
zeroed outputs for a selected CUDA context. No host fallback is counted as GPU
execution until a resident ordinal likelihood reduction kernel is linked.

The machine-readable rows are in
[`gp_ordinal_likelihood.csv`](gp_ordinal_likelihood.csv). Reproduce with:

```text
python scripts/bench_gp_ordinal_likelihood.py \
  --fortml ../fortml --output results/gp_ordinal_likelihood.csv
```

Pinned source revision: `93da9b549fee98aad5af977e486cd64b16507a40`.
Pinned benchmark revision: `25515c70397a10554b0ee70e8cde54e1a12b46e5`.
