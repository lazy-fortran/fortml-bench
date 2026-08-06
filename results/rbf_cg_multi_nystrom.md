# Nystrom-preconditioned multi-RHS CG scaling

This record evaluates the experimental KeOps-style path in FortML. A rank-32
Nystrom/Woodbury factor is built from uniformly spaced landmarks. Each CG
iteration applies the RBF operator matrix-free and applies the low-rank
preconditioner to all four right-hand sides in one fused operation.

The matched workload uses 4 right-hand sides, 8 features, float64 arithmetic,
RBF variance 1.4, lengthscale 0.7, diagonal shift 0.08, relative tolerance
`1e-8`, and a 500-iteration cap. Sample counts are 256, 512, 1024, and 2048.
The CPU lane uses 16 nvfortran OpenMP threads. The GPU lane uses an RTX 5060
Ti with nvfortran 26.5 and resident inputs. Each timing is the mean of two
solves after one correctness warm-up.

The `seconds_per_solve` column is steady-state solve time. The
`setup_seconds` column covers Nystrom construction, factorization, and the
correctness warm-up after operator data has entered the device. The first-solve
plots add those two columns. Reference setup times are recorded by their own
drivers under the same workload.

At 2048 samples, the endpoint measurements are:

| lane | backend | iterations | steady solve (s) | setup + solve (s) |
| --- | --- | ---: | ---: | ---: |
| CPU, default OpenACC build | FortML Nystrom | 58 | 0.254 | 0.527 |
| CPU, default OpenACC build | dense PyTorch | 199 | 0.129 | 0.364 |
| CPU, default OpenACC build | KeOps | 195 | 0.866 | 1.830 |
| CPU, default OpenACC build | GPyTorch-KeOps | 193 | 0.730 | 1.614 |
| CUDA, OpenACC | FortML Nystrom | 59 | 0.196 | 0.397 |
| CUDA, OpenACC | dense PyTorch | 198 | 0.361 | 0.930 |
| CUDA, OpenACC | KeOps | 192 | 0.962 | 1.927 |
| CUDA, OpenACC | GPyTorch-KeOps | 196 | 0.850 | 1.873 |
| CPU, native-CUDA build | FortML Nystrom | 58 | 0.261 | 0.538 |
| CPU, native-CUDA build | dense PyTorch | 199 | 0.130 | 0.390 |
| CPU, native-CUDA build | KeOps | 195 | 0.851 | 1.796 |
| CPU, native-CUDA build | GPyTorch-KeOps | 193 | 0.716 | 1.589 |
| CUDA, native CUDA | FortML Nystrom | 58 | 0.112 | 0.233 |
| CUDA, native CUDA | dense PyTorch | 198 | 0.355 | 0.922 |
| CUDA, native CUDA | KeOps | 192 | 0.961 | 1.928 |
| CUDA, native CUDA | GPyTorch-KeOps | 196 | 0.847 | 1.866 |

The native CUDA FortML lane is below all three reference curves at every
tested GPU size in steady-state time. Its 2048-point first solve is also below
all three references. The default OpenACC lane is below the reference GPU
curves in both steady-state and first-solve time, except that dense PyTorch
remains the faster CPU steady-state reference. The Nystrom path is therefore
an effective iteration-reduction path, while the CPU and default GPU plots
still expose room for lower per-operator overhead.

Plots:

- [default CPU, steady state](https://box.sloppy.at/0fbe3.png)
- [default CUDA, steady state](https://box.sloppy.at/b6181.png)
- [default CPU, first solve](https://box.sloppy.at/aac6c.png)
- [default CUDA, first solve](https://box.sloppy.at/eb42b.png)
- [native-CUDA build CPU, steady state](https://box.sloppy.at/c9432.png)
- [native CUDA, steady state](https://box.sloppy.at/dbb7e.png)
- [native-CUDA build CPU, first solve](https://box.sloppy.at/57cb8.png)
- [native CUDA, first solve](https://box.sloppy.at/e8774.png)

Every row passed the blocked NumPy matmat residual oracle. The FortML rows
also passed the independent dense multi-RHS solve in the `fortml` test suite.
The Nystrom implementation is covered by a rank-2 multi-RHS test against the
dense solve, so the benchmark result is not validated only by repository-state
checks.

The raw records are [default OpenACC](nystrom_setup/rbf_cg_multi_scaling.csv)
and [native CUDA](nystrom_native_setup/rbf_cg_multi_scaling.csv). The source
revisions are FortML `77d1f71` and FortNum `5bce667`. The native CUDA run uses
`FORTML_NATIVE_CUDA=1` and the shared neighbor-tile kernel. Nsight Compute
counters remain unavailable under the cluster's `ERR_NVGPUCTRPERM` restriction,
so this record makes no occupancy or memory-counter claim.

The Nystrom path changes the algorithm relative to the unpreconditioned
reference rows. The comparison keeps the matrix, precision, stopping rule,
right-hand sides, and input residency matched, and reports both solve-only and
setup-inclusive costs so the trade-off is explicit.
