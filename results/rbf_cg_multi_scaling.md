# Multi-RHS matrix-free CG scaling

The matched workload uses 4 right-hand sides, 8 features, float64 arithmetic,
RBF variance 1.4, lengthscale 0.7, diagonal shift 0.08, relative tolerance
`1e-8`, and a 500-iteration cap. Sample counts are 256, 512, 1024, and 2048.
The CPU lane uses 16 nvfortran OpenMP threads. The GPU lane uses an RTX 5060
Ti with nvfortran 26.5 and resident inputs. Each timing is the mean of two
solves after one correctness warm-up.

Every row passed the blocked NumPy matmat residual oracle. The FortML rows also
passed the independent dense multi-RHS solve in `fortml`'s test suite. The
comparison rows passed the same NumPy residual and dense solve checks where the
matrix fit in the oracle limit.

At 2048 samples, the default OpenACC lane records:

| device | FortML | dense PyTorch | KeOps | GPyTorch-KeOps |
| --- | ---: | ---: | ---: | ---: |
| CPU, seconds | 0.834 | 0.158 | 0.867 | 0.718 |
| CUDA, seconds | 0.576 | 0.357 | 0.962 | 0.846 |

The refreshed native CUDA lane records 0.328 seconds at 2048 samples. It is
faster than dense PyTorch, KeOps, and GPyTorch-KeOps at that endpoint. Across
all four tested GPU sizes it remains within the 30 percent GPyTorch target. The
native CUDA GPU plot is published at https://box.sloppy.at/01752.png. The
default OpenACC GPU plot is published at https://box.sloppy.at/7976b.png. CPU
plots are https://box.sloppy.at/802a2.png and https://box.sloppy.at/10733.png
for the native and default lanes.

The raw records are `rbf_cg_multi_scaling.csv` and
`native_cuda/rbf_cg_multi_scaling.csv`. The source revision is `fortml`
`e0c9a85` with `fortnum` `5bce667`. The native CUDA run uses
`FORTML_NATIVE_CUDA=1` and the shared neighbor-tile matmat kernel. Nsight
Compute counters remain unavailable under the cluster's `ERR_NVGPUCTRPERM`
restriction, so this record makes no occupancy or memory-counter claim.

The native GPU curve still has a higher local asymptotic slope than the
GPyTorch-KeOps curve. Block or Nystrom preconditioning and a persistent
Krylov-workspace API remain the next scaling work.
