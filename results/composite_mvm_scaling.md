# KeOps-style composite MVM scaling

This record compares a static matrix-free RBF-plus-constant operator:

\[
 y_i = 0.08v_i + 0.2\sum_j v_j + 1.4
 \sum_j \exp\left(-\frac{\lVert x_i-x_j\rVert^2}{2(0.7)^2}\right)v_j.
\]

The eight-dimensional points and vector are deterministic and shared by all
lanes. Every timed result first passes an independent blocked NumPy pairwise
oracle in float64. FortML uses nvfortran: `-O3 -mp=multicore` on the CPU and
`-O3 -acc` on the RTX 5060 Ti. KeOps uses its LazyTensor reduction. GPyTorch
uses its KeOps RBF operator plus the same explicit constant rank-one term.
Dense PyTorch materializes the reference matrix.

## Representative resident timings

Milliseconds per MVM; lower is better. The high-N rows use two repetitions.

| N | FortML CPU | KeOps CPU | GPyTorch CPU | FortML GPU | KeOps GPU | GPyTorch GPU |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 1.199 | 3.854 | 3.340 | 0.993 | 3.828 | 3.636 |
| 16,384 | 54.537 | 80.695 | 62.110 | 53.833 | 80.319 | 61.544 |

FortML is faster than the GPyTorch-KeOps lane at both endpoints and remains
within the requested 30% runtime envelope. The GPU dense PyTorch reference is
faster at small sizes, but it materializes O(N²) storage and is out of memory
at N=4096 and above in this run; it is therefore a capacity reference rather
than a scalable matrix-free competitor.

The GPU FortML doubling slope settles near quadratic at the high end (1.94
and 1.97 for 4096→8192 and 8192→16384), as expected for an exact dense-kernel
sum. KeOps and GPyTorch show lower local endpoint slopes in this particular
run, while their absolute times remain higher. CPU slopes are noisier because
the full-core host lane crosses parallel scheduling and cache regimes; the
N=16384 endpoint is still below both matrix-free competitors.

The raw data are in
[`composite_mvm_scaling_extended.csv`](composite_mvm_scaling_extended.csv),
with plots for [CPU](composite_mvm_scaling_extended_cpu.png) and
[CUDA](composite_mvm_scaling_extended_cuda.png). The shorter five-repetition
record through N=4096 is retained in
[`composite_mvm_scaling.csv`](composite_mvm_scaling.csv).

The optional native CUDA path is enabled with `FORTML_NATIVE_CUDA=1` and was
independently oracle-checked at N=2048. Nsight Compute counter collection
remains permission-gated on this cluster (`ERR_NVGPUCTRPERM`); the benchmark
does not present unavailable hardware counters as evidence.
