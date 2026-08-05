#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
fortml=${FORTML_DIR:-$root/../fortml}
fortnum=${FORTNUM_DIR:-$root/../fortnum}
output=${PROFILE_OUT:-$root/results/profiles}
cpu_n=${PROFILE_CPU_N:-4096}
gpu_n=${PROFILE_GPU_N:-2048}
features=${PROFILE_FEATURES:-8}
cpu_repetitions=${PROFILE_CPU_REPETITIONS:-200}
gpu_repetitions=${PROFILE_GPU_REPETITIONS:-12}
cpu_threads=${CPU_THREADS:-$(lscpu -p=CORE 2>/dev/null | awk '!/^#/ {print $1}' | sort -u | wc -l)}
test "$cpu_threads" -gt 0 || cpu_threads=1
mkdir -p "$output"
build=$(mktemp -d)

if [[ -n "${PROFILE_CPU_FC:-}" ]]; then
    cpu_fc=$PROFILE_CPU_FC
elif command -v nvfortran >/dev/null 2>&1; then
    cpu_fc=nvfortran
else
    cpu_fc=gfortran
fi
if [[ "$cpu_fc" == "nvfortran" ]]; then
    cpu_flags=${PROFILE_CPU_FLAGS:--O3 -mp}
    cpu_module_flag=(-module "$build")
else
    cpu_flags=${PROFILE_CPU_FLAGS:--O3 -march=native -fopenmp -fno-math-errno -flto -fwhole-program}
    cpu_module_flag=(-J "$build")
fi
$cpu_fc $cpu_flags "${cpu_module_flag[@]}" \
    -o "$build/rbf_cpu" \
    "$fortnum/src/fortnum_kinds.f90" \
    "$fortnum/src/fortnum_status.f90" \
    "$fortnum/src/linalg/fortnum_krylov.f90" \
    "$fortml/src/gp/fortml_kernels.f90" \
    "$fortml/src/gp/fortml_linear_operator.f90" \
    "$fortml/src/gp/fortml_kernel_operator.f90" \
    "$fortml/app/fortml_bench_rbf_operator.f90"
OMP_NUM_THREADS="$cpu_threads" perf stat -r 3 \
    -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
    "$build/rbf_cpu" resident "$cpu_n" "$features" "$cpu_repetitions" \
    >"$output/perf_cpu_output.txt" 2>"$output/perf_cpu.txt"

if command -v nvfortran >/dev/null 2>&1 && command -v nsys >/dev/null 2>&1; then
    nvfortran -O3 -acc -module "$build" -o "$build/rbf_gpu" \
        "$fortnum/src/fortnum_kinds.f90" \
        "$fortnum/src/fortnum_status.f90" \
        "$fortnum/src/linalg/fortnum_krylov.f90" \
        "$fortml/src/gp/fortml_kernels.f90" \
        "$fortml/src/gp/fortml_linear_operator.f90" \
        "$fortml/src/gp/fortml_kernel_operator.f90" \
        "$fortml/app/fortml_bench_rbf_operator.f90"
    gpu_index=${GPU_INDEX:-1}
    CUDA_VISIBLE_DEVICES="$gpu_index" NV_ACC_TIME=1 nsys profile \
        --trace=cuda,nvtx,osrt --stats=true --force-overwrite=true \
        -o "$output/nsys_gpu" "$build/rbf_gpu" resident "$gpu_n" "$features" \
        "$gpu_repetitions" >"$output/nsys_gpu_output.txt" \
        2>"$output/nsys_gpu_profile.txt"
    nsys stats --report cuda_gpu_kern_sum --force-export=true \
        "$output/nsys_gpu.nsys-rep" >"$output/nsys_gpu_kernels.txt" 2>&1
    if command -v ncu >/dev/null 2>&1; then
        set +e
        CUDA_VISIBLE_DEVICES="$gpu_index" ncu --set basic --launch-skip 1 \
            --launch-count 1 --csv --log-file "$output/ncu_gpu.csv" \
            "$build/rbf_gpu" resident "$gpu_n" "$features" "$gpu_repetitions" \
            >"$output/ncu_gpu_output.txt" 2>&1
        ncu_status=$?
        set -e
        printf 'ncu_exit_status=%s\n' "$ncu_status" >"$output/ncu_gpu_status.txt"
    fi
else
    printf 'nvfortran or nsys unavailable\n' >"$output/gpu_profile_unavailable.txt"
fi
