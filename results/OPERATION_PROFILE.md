# Operation-level RBF MVM profile

Run date: 2026-08-05. The matched workload is 1024 samples, 8 features,
float64, resident inputs, and the same variance, lengthscale, points, input
vector, and independent NumPy oracle as the timing suite. Python operations
were collected with torch.profiler after two warm-up calls. The Fortran CPU
lane used nvfortran -O3 -mp and perf stat over 300 MVMs. The GPU lane used
nvfortran -O3 -acc, Nsight Systems, and NV_ACC_TIME over 12 MVMs.

## What each implementation executes

| Implementation | Pairwise operation | Storage and launch behavior |
| --- | --- | --- |
| Dense PyTorch | sub -> pow -> sum -> scale -> exp -> addmv | Materializes an N x N x D difference tensor and N x N intermediates, then launches separate elementwise, reduction, exponential, and matrix-vector work. |
| KeOps | A generated GenredAutograd map-reduce, exposed on CUDA as GpuConv1DOnDevice | Fuses distance, exponential, multiply, and reduction into a custom tiled kernel without an N x N result. |
| GPyTorch + KeOps | GPyTorch KernelLinearOperator.matmul around the same KeOps map-reduce | Keeps the matrix-free KeOps kernel, but adds operator dispatch, parameter transforms, copies, and stream synchronization. |
| fortml | One tiled reduction per output row with eight-feature distance unrolling, one exponential, and an OpenMP/OpenACC reduction over neighbors | Does not materialize pairwise storage. CPU uses a static outer-row schedule and SIMD neighbor reduction. nvfortran launches one 128-thread kernel per MVM with sample-major point storage. |

For this workload, each exact MVM evaluates 1,048,576 pairs. The Fortran
hot loop therefore performs eight coordinate differences and eight squares per
pair, followed by the distance accumulation, one exponential, one kernel
scale, and one vector accumulation. The operation count is the same
mathematically for KeOps and GPyTorch-KeOps, but their generated map-reduce
keeps those values in registers or local tile storage instead of exposing
each intermediate tensor to the framework.

## Measured operation traces

The Python CUDA trace reported these dominant self-device operations:

| Backend | Dominant measured operations |
| --- | --- |
| Dense PyTorch | sum 367.1 us, pow 350.8 us, sub 136.6 us, exp 97.5 us, and addmv_ 11.6 us, plus multiple CUDA launches. |
| KeOps | One GpuConv1DOnDevice operation at 1731.2 us and one GenredAutograd wrapper at 1731.4 us. |
| GPyTorch-KeOps | One GpuConv1DOnDevice operation at 1296.2 us, with a GenredAutograd wrapper at 1296.4 us and a separate Matmul framework operation. |

The Fortran Nsight Systems trace reported 13 launches for the correctness
call plus 12 timed resident calls. The kernel used grid [1024], block [128],
and averaged 276.0 us per launch. NV_ACC_TIME measured 48 us of input copies
and 16 us of output copyout for the whole profiled run. The Nsight Compute
attempt was blocked by ERR_NVGPUCTRPERM, so occupancy, register count,
warp-stall, and memory throughput counters still require GPU
performance-counter permission on the cluster.

The nvfortran CPU perf stat run measured, amortized per MVM, approximately
15.0 million cycles, 42.0 million instructions, 42 thousand cache misses, and
3.7 thousand branch misses. The regular reduction loop remains branch-light.
The remaining CPU profiling question is vector math throughput for the
exponential and reuse across neighboring output rows, which needs a
symbol-level profile on an otherwise idle node.

## Inefficiencies found and fixed

The original Fortran operator stored points with the feature index leading.
The neighbor loop consequently loaded each feature with an eight-sample
stride for this workload. The operator now stores samples contiguously and
the inner neighbor index is unit stride. The eight-feature path also uses
explicit multiplies instead of scalar power expressions. These changes are in
fortml commit a205898.

The remaining structural differences are deliberate optimization targets:

- Dense PyTorch pays for pairwise tensors and reaches the recorded GPU OOM
  boundary at 4096.
- KeOps already has the right fused map-reduce structure. Its visible
  overhead is mainly generated-kernel dispatch and synchronization.
- GPyTorch-KeOps retains that fused kernel but adds the KernelLinearOperator
  layer. This is why its operation trace contains Matmul, copies, and more
  stream synchronization than direct KeOps.
- fortml still needs feature-specialized kernels beyond eight features,
  multi-right-hand-side fusion for matmat, and a CPU vector-math profile.
  The current GPU kernel also reloads neighbor points for each output row.
  two-dimensional point tiling is the next GPU optimization candidate.

All reported traces passed the independent direct pairwise oracle. The Python
KeOps and GPyTorch results are float64 but are not bitwise identical to the
Fortran reduction order. Their observed relative errors were about 2e-15 and
1e-8, respectively, against the same oracle.

Raw Python operation tables are
[operation_profile_cpu.csv](operation_profile_cpu.csv) and
[operation_profile_cuda.csv](operation_profile_cuda.csv). The reproducible
commands are scripts/profile_python_ops.py and scripts/profile_rbf_mvm.sh.
