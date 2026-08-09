# ICM multi-output GP likelihood products

This lane measures the exact intrinsic-coregionalization likelihood products
added to `fortml_multi_output_gp`. The release fixture has 40 samples, two
features, three outputs, two latent ICM columns, and five repetitions. An
independent NumPy dense solve assembles
`A = (W W^T + diag(independent)) (x) K + sigma^2 I`. Central differences of
the scalar likelihood provide gradient and directional-HVP oracles. The
focused Fortran oracle additionally checks every coordinate, transactional
rollback, scalar JVP/VJP composition, shape metadata, and CUDA refusal.

| phase | device | status | seconds/op | metric | value | max error |
| --- | --- | --- | ---: | --- | ---: | ---: |
| likelihood JVP | CPU | pass | - | JVP value | -1.066681959088936 | 6.20e-10 |
| hyperparameter gradient | CPU | pass | 1.4728292e-3 | gradient sum | -56.534692322053864 | 2.20e-8 |
| hyperparameter HVP | CPU | pass | 1.7706084e-3 | HVP sum | -0.4786974599017657 | 3.17e-6 |
| hyperparameter products | CUDA | unavailable | - | typed refusal | - | - |

The CSV records Python 3.14.6, NumPy 2.5.1, gfortran, FortML revision
`124a9b430085eeb5fbf2343aa36bbc0a0a08adb5`, and benchmark revision
`ca1bbf1ea967458a6f4cb71c21bfed7a507eff82`. Reproduce it with:

```bash
FO_FC=gfortran FO_SCAN_FALLBACK=regex \
  python3 -B scripts/bench_multi_output_gp_hypergradients.py \
  --fortml ../fortml --output results/multi_output_gp_hypergradients.csv \
  --report results/MULTI_OUTPUT_GP_HYPERGRADIENTS.md
```

CUDA is intentionally `unavailable` (`FORTNUM_NOT_IMPLEMENTED`) until
resident ICM covariance and factorization kernels are linked. The CPU path is
never relabelled as an accelerator measurement.
