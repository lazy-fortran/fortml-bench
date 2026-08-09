# Ordinal-GP cut-point calibration

This lane checks the fixed-latent cut-point optimizer in
`fortml_gp_ordinal_cutpoint_training`. A dense RBF GP is fitted to rank targets
for 24 observations and three ordered integer labels. The cut-point objective
uses the GP posterior mean, an ordered-probit likelihood, and nonuniform sample
weights. FortOpt L-BFGS-B works in one location coordinate and one logarithmic
gap coordinate.

## Protocol

- FortML source: `090103e37a9ea0d916d167533fa929a2f6b3a794`
- Compiler: GNU Fortran with `-O3`
- Precision: IEEE float64
- RBF kernel: variance `1.3`, length scale `0.81`
- Observation noise: `0.08`, with `1e-8` jitter
- Initial thresholds: `[1.18, 2.78]`
- Product direction: `[0.16, -0.11]`
- Timing: 64 calls after warmup
- Optimizer bounds: first threshold `[-1,4]`, log gap `[-4,2]`
- Optimizer tolerance: `2e-7`, maximum 160 iterations

The Python oracle assembles the dense RBF posterior mean with NumPy. It then
codes the weighted ordered-probit reduction and threshold gradient separately
with SciPy's normal CDF. Coordinate central differences check the gradient. A
directional central difference of that independent gradient supplies the HVP
oracle. SciPy L-BFGS-B solves the transformed bounded objective independently.

## Results

| Quantity | Independent oracle | FortML | Absolute error | Seconds/call |
| --- | ---: | ---: | ---: | ---: |
| Initial NLL | `0.6029667498348230` | `0.6029667498348232` | `2.22e-16` | `1.33e-5` |
| Gradient norm | `0.1700267597582023` | `0.1700267597582021` | `2.22e-16` | `1.33e-5` |
| Directional HVP norm | `0.0899315324922422` | `0.0899315324776850` | `1.46e-11` | `1.35e-5` |
| Final NLL | `0.5750542355400928` | `0.5750542355401022` | `9.44e-15` | `6.11e-5` total |

FortOpt converged in seven iterations and six line-search evaluations. Its
final threshold-gradient norm was `9.34e-8`. FortML returned thresholds
`[1.3904253331, 2.5502926281]`. The maximum difference from the independent
SciPy solution was `1.97e-7`, within the declared optimizer tolerance. The
coordinate-wise finite-difference error in the Python gradient was `1.29e-11`.

`test_gp_ordinal_cutpoint_training` also checks probability and log-probability
threshold JVP/VJP products, adjoint identities, transformed-gap ordering,
weighted convergence, malformed-input rollback, and nonconvergence rollback.
CUDA objective and training calls return `FORTNUM_NOT_IMPLEMENTED` (status
code 3). The implementation makes no GPU timing claim.

The raw six-row record is
[`gp_ordinal_cutpoints.csv`](gp_ordinal_cutpoints.csv). Reproduce it with:

```bash
FO_SCAN_FALLBACK=regex python3 -B scripts/bench_gp_ordinal_cutpoints.py \
  --fortml ../fortml --output results/gp_ordinal_cutpoints.csv
```
