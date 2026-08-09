# Ordinal-GP evidence gradients and HVPs

This lane measures the exact latent-Gaussian ordinal GP evidence products and
the bounded FortOpt L-BFGS-B adapter. The Fortran model fits rank targets for
three sorted integer labels, then exposes the packed coordinate block
`[log(signal variance), log(length scale), log(noise variance)]`. The
independent NumPy oracle assembles the dense RBF covariance and Cholesky solve
directly; it checks the analytic evidence gradient against coordinate-wise
central differences and obtains the directional HVP by central-differencing
that independently assembled gradient.

## Protocol

- Source revision: `ef13cfbacadfffc2dbf4ba56abc14baf11cb722d`
- Benchmark revision: `4ad685249e9a488c9b3e4fdc19e837a537e8b903`
- Compiler: GNU Fortran, `-O3`
- Precision: IEEE float64
- Fixture: 18 points on `[-1.7,1.7]`; labels `[-4,7,19]` by two ordered cuts;
  RBF variance `1.35`, length scale `0.79`, noise variance `0.05`, jitter
  `1e-8`; direction `[0.07,-0.04,0.03]`
- Timing: 32 resident gradient/HVP calls after fit and warmup
- Training: bounded FortOpt L-BFGS-B, bounds `[-8,8]`, gradient tolerance
  `2e-5`, maximum 120 iterations

The independent oracle gives initial log marginal likelihood
`-1.3085550253342490e+01`, gradient norm `3.564467471703633`, and HVP norm
`5.7132941498536105e-01`. The coordinate-wise likelihood-gradient error is
`3.76e-09`, and the scalar directional finite-difference check is below
`2e-08`.

## Results

| Product | NumPy norm/check | Fortran result | Max absolute error | Fortran seconds/call |
| --- | ---: | ---: | ---: | ---: |
| Evidence gradient | `3.564467471703633` | `3.5644674717035922` | `4.09e-14` | `1.45834e-05` |
| Directional HVP | `5.7132941498536105e-01` | `5.7132941618935000e-01` | `1.20e-09` | `3.75320e-05` |

FortOpt converged in 16 iterations and 18 objective evaluations. The final
negative log marginal likelihood was `8.9340880731858778`, with gradient norm
`7.658010707165161e-07`. The Fortran behavioral gate
`test_gp_ordinal_classification_hyperparameters` passed its independent
finite-difference, HVP, transactionality, convergence, and device-boundary
checks.

CUDA is recorded as a typed `FORTNUM_NOT_IMPLEMENTED` refusal (status code
`3`) for both evidence products and the optimizer. Exact factorization and
the control-plane optimizer are not resident CUDA implementations, so this
lane makes no GPU timing claim. The raw six-row record is
[`gp_ordinal_hyperparameters.csv`](gp_ordinal_hyperparameters.csv).

Run the lane with:

```bash
FO_SCAN_FALLBACK=regex python3 -B scripts/bench_gp_ordinal_hyperparameters.py \
  --fortml ../fortml --output results/gp_ordinal_hyperparameters.csv
```
