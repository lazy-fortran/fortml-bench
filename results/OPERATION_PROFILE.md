# Operation-level RBF MVM profile

Run date: 2026-08-06. The matched workload is 1024 samples, 8 features,
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
| fortml | One fused reduction for each output row, with two output rows assigned to worker lanes in the eight-feature path, one exponential, and an OpenMP/OpenACC reduction over neighbors | Does not materialize pairwise storage. CPU uses a static outer-row schedule and SIMD neighbor reduction. nvfortran launches one kernel per MVM with sample-major point storage. Neighbor points are still reloaded for each output row. |

For this workload, each exact MVM evaluates 1,048,576 pairs. The Fortran
hot loop therefore performs eight coordinate differences and eight squares per
pair, followed by the distance accumulation, one exponential, one kernel
scale, and one vector accumulation. The operation count is the same
mathematically for KeOps and GPyTorch-KeOps, but their generated map-reduce
keeps those values in registers or local tile storage instead of exposing
each intermediate tensor to the framework.

## Measured operation traces

The refreshed Python CUDA trace reported these dominant self-device operations:

| Backend | Dominant measured operations |
| --- | --- |
| Dense PyTorch | sum 358.8 us, pow 353.0 us, sub 137.2 us, exp 95.3 us, and addmv_ 11.1 us, plus multiple CUDA launches. |
| KeOps | One GpuConv1DOnDevice operation at 1685.4 us and one GenredAutograd wrapper at 1685.7 us. |
| GPyTorch-KeOps | One GpuConv1DOnDevice operation at 1261.9 us, with a GenredAutograd wrapper at 1262.1 us and a separate Matmul framework operation. |

The refreshed FortML Nsight Systems trace reported 13 launches for the
correctness call plus 12 timed resident calls. The specialized OpenACC kernel
averaged 922.1 us per launch at 2,048 samples. The benchmark application
measured 946.4 us per resident MVM across the same 12 calls. NV_ACC_TIME
reported 45 us of input copies and 15 us of output copyout for the operator
call. The optional native CUDA bridge produced one four-warp kernel per MVM,
averaging 915.6 us in Nsight Systems and 940.7 us in the application across
eight timed calls. Its shared-memory neighbor tile therefore matches the
OpenACC envelope on this GPU, but does not yet justify changing the default.
Both paths use the same direct pairwise oracle. The Nsight Compute attempt
returned `ERR_NVGPUCTRPERM`, so occupancy, register count, warp-stall, and
memory-throughput counters still require GPU performance-counter permission on
the cluster.

The refreshed nvfortran CPU perf stat run at 4,096 samples measured, amortized
per MVM, approximately 197.1 million cycles, 478.2 million instructions, 3.67
million cache misses, and 11.8 thousand branch misses. The regular reduction
loop remains branch-light relative to the arithmetic work. The remaining CPU
profiling question is vector math throughput for the exponential and reuse
across neighboring output rows, which needs a symbol-level profile on an
otherwise idle node.

## Inefficiencies found and fixed

The original Fortran operator stored points with the feature index leading.
The neighbor loop consequently loaded each feature with an eight-sample
stride for this workload. The operator now stores samples contiguously and
the inner neighbor index is unit stride. The eight-feature path also uses
explicit multiplies instead of scalar power expressions. The current source
is fortml commit `81b0655`. The optional native bridge adds a linked CUDA
kernel with shared neighbor storage while leaving the OpenACC path as the
default comparison backend.

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
  The default OpenACC GPU kernel assigns two output rows to worker lanes and
  reloads neighbor points for each output row. The optional native path removes
  that reload with a shared neighbor tile, but its launch and data-region
  overhead currently erase the kernel-time gain.

All reported traces passed the independent direct pairwise oracle. The Python
KeOps and GPyTorch results are float64 but are not bitwise identical to the
Fortran reduction order. Their observed relative errors were about 2e-15 and
1e-8, respectively, against the same oracle.

Raw Python operation tables are
[operation_profile_cpu.csv](operation_profile_cpu.csv) and
[operation_profile_cuda.csv](operation_profile_cuda.csv). The reproducible
commands are scripts/profile_python_ops.py and scripts/profile_rbf_mvm.sh.
