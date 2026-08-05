#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
python=${PYTHON:-$root/.venv/bin/python}
fortml=${FORTML_DIR:-$root/../fortml}
sizes=${SIZES:-"256 512 1024 2048 4096"}
repetitions=${SCALING_REPETITIONS:-12}
cpu_threads=${CPU_THREADS:-$(lscpu -p=CORE 2>/dev/null | awk '!/^#/ {print $1}' | sort -u | wc -l)}
test "$cpu_threads" -gt 0 || cpu_threads=1
cpu_flags=${CPU_FFLAGS:--O3 -march=native -fopenmp -fno-math-errno -flto -fwhole-program}
gpu_flags=${GPU_FFLAGS:--O3 -acc}
mkdir -p "$root/results"
inputs=()

for n in $sizes; do
    OMP_NUM_THREADS="$cpu_threads" FFLAGS="$cpu_flags" \
        "$python" "$root/scripts/run_fortran_rbf.py" \
        --fortml "$fortml" --device cpu --n "$n" --d 8 \
        --repetitions "$repetitions" \
        --output "$root/results/scaling_fortran_cpu_$n.csv"
    OMP_NUM_THREADS="$cpu_threads" "$python" "$root/scripts/bench_rbf_mvm.py" \
        --device cpu --n "$n" --d 8 --threads "$cpu_threads" \
        --repetitions "$repetitions" \
        --output "$root/results/scaling_python_cpu_$n.csv"
    inputs+=("$root/results/scaling_fortran_cpu_$n.csv")
    inputs+=("$root/results/scaling_python_cpu_$n.csv")
done

if "$python" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    for n in $sizes; do
        FFLAGS="$gpu_flags" "$python" "$root/scripts/run_fortran_rbf.py" \
            --fortml "$fortml" --device cuda --n "$n" --d 8 \
            --repetitions "$repetitions" \
            --output "$root/results/scaling_fortran_cuda_$n.csv"
        "$python" "$root/scripts/bench_rbf_mvm.py" \
            --device cuda --n "$n" --d 8 --repetitions "$repetitions" \
            --output "$root/results/scaling_python_cuda_$n.csv"
        inputs+=("$root/results/scaling_fortran_cuda_$n.csv")
        inputs+=("$root/results/scaling_python_cuda_$n.csv")
    done
fi

"$python" "$root/scripts/merge_results.py" \
    "$root/results/rbf_mvm_scaling.csv" "${inputs[@]}"
"$python" "$root/scripts/plot_scaling.py" \
    "$root/results/rbf_mvm_scaling.csv" "$root/results/rbf_mvm_scaling"
