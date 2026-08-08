# Binary Laplace-GP hyperparameter HVP

This workload checks the implicit-mode kernel hyperparameter HVP for the
binary Laplace GP classifier. The Fortran method differentiates the converged
Newton mode through the resident posterior factorization. The independent
NumPy path refits perturbed RBF kernels and central-differences the envelope
gradient, so it checks the mode tangent and kernel second products together.

## Protocol

- Source: FortML `db0d42a81e2d21d8e56bc3584e73fb1bede16679`
- Benchmark harness: fortml-bench `5e80fee8bd711675f08c2e6c554b14e369213dc7`
- Compiler: GNU Fortran, `-O3`
- Precision: IEEE float64
- Fixture: 24 points on `[-1.5,1.5]`, labels `[-7,11]`, RBF log parameters
  `log([1.35,0.79])`, direction `[0.07,-0.04]`, jitter `1e-7`
- Repetitions: 32 resident HVP calls after one fit and warmup

The behavioral gate `test_gp_classification_hvp` passed. It covers logistic
and probit fits, independent refit finite differences, transactional invalid
parameter updates, and the selected-CUDA boundary.

## Results

| Likelihood | NumPy HVP checksum | Fortran checksum | Max absolute checksum error | Fortran seconds/call |
| --- | ---: | ---: | ---: | ---: |
| Logistic | 0.17509019980516127 | 0.17509019978731244 | `1.78e-11` | `5.13e-05` |
| Probit | 0.13030205829545060 | 0.13030205831162820 | `2.58e-11` | `5.58e-05` |

The checksum is the sum of the two packed HVP coordinates. The NumPy oracle
also reports HVP norms `1.9209442171713242e-01` (logistic) and
`1.9461063506134180e-01` (probit). The full records are in
[`gp_classification_hvp.csv`](gp_classification_hvp.csv).

CUDA is recorded as a typed `FORTNUM_NOT_IMPLEMENTED` refusal (status code
`3`). The binary Laplace covariance and derivative state are not resident on
CUDA, so the benchmark makes no GPU timing claim.

Run the lane with:

```bash
python -B scripts/bench_gp_classification_hvp.py \
  --fortml ../fortml --output results/gp_classification_hvp.csv
```
