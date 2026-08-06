#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
fortml=${FORTML_DIR:-$root/../fortml}
python=${PYTHON:-$root/.venv/bin/python}
cpu_threads=${CPU_THREADS:-$(lscpu -p=CORE 2>/dev/null | awk '!/^#/ {print $1}' | sort -u | wc -l)}
test "$cpu_threads" -gt 0 || cpu_threads=1
repetitions=${REPETITIONS:-12}
features=${N_FEATURES:-8}
values=${N_VALUES:-"256 512 1024 2048 4096"}
if [[ -n "${CPU_FC:-}" ]]; then
    cpu_compiler=$CPU_FC
elif command -v nvfortran >/dev/null 2>&1; then
    cpu_compiler=nvfortran
else
    cpu_compiler=gfortran
fi
if [[ "$cpu_compiler" == "nvfortran" ]]; then
    cpu_flags=${CPU_FFLAGS:--O3 -mp=multicore}
else
    cpu_flags=${CPU_FFLAGS:--O3 -march=native -fopenmp -fno-math-errno}
fi
gpu_flags=${GPU_FFLAGS:--O3 -acc}
results_dir=${RESULTS_DIR:-$root/results}
run_dir=$(mktemp -d)
trap 'rm -rf "$run_dir"' EXIT

inputs=()
for n in $values; do
    OMP_NUM_THREADS=$cpu_threads FORTML_FC="$cpu_compiler" \
        FFLAGS="$cpu_flags" "$python" "$root/scripts/run_fortran_composite.py" \
        --fortml "$fortml" --device cpu --n "$n" --d "$features" \
        --repetitions "$repetitions" --output "$run_dir/fortml_cpu_$n.csv"
    OMP_NUM_THREADS=$cpu_threads "$python" "$root/scripts/bench_composite_mvm.py" \
        --device cpu --n "$n" --d "$features" --threads "$cpu_threads" \
        --repetitions "$repetitions" --output "$run_dir/python_cpu_$n.csv"
    inputs+=("$run_dir/fortml_cpu_$n.csv" "$run_dir/python_cpu_$n.csv")

    FFLAGS="$gpu_flags" "$python" \
        "$root/scripts/run_fortran_composite.py" --fortml "$fortml" \
        --device cuda --n "$n" --d "$features" --repetitions "$repetitions" \
        --output "$run_dir/fortml_cuda_$n.csv"
    "$python" "$root/scripts/bench_composite_mvm.py" --device cuda --n "$n" \
        --d "$features" --repetitions "$repetitions" \
        --output "$run_dir/python_cuda_$n.csv"
    inputs+=("$run_dir/fortml_cuda_$n.csv" "$run_dir/python_cuda_$n.csv")
done

mkdir -p "$results_dir"
"$python" "$root/scripts/merge_results.py" \
    "$results_dir/composite_mvm_scaling.csv" "${inputs[@]}"
"$python" "$root/scripts/plot_composite_scaling.py" \
    "$results_dir/composite_mvm_scaling.csv" "$results_dir/composite_mvm_scaling"
