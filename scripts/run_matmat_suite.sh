#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
python=${PYTHON:-$root/.venv/bin/python}
fortml=${FORTML_DIR:-$root/../fortml}
results_dir=${RESULTS_DIR:-$root/results}
n=${N_SAMPLES:-2048}
d=${N_FEATURES:-8}
repetitions=${REPETITIONS:-5}
rhs_values=${MATMAT_RHS:-"1 2 4 8"}
mkdir -p "$results_dir"

for rhs in $rhs_values; do
    OMP_NUM_THREADS=${CPU_THREADS:-1} "$python" \
        "$root/scripts/run_fortran_rbf_matmat.py" \
        --fortml "$fortml" --device cpu --backend openacc --n "$n" --d "$d" \
        --rhs "$rhs" --repetitions "$repetitions" \
        --output "$results_dir/rbf_matmat_cpu_${rhs}.csv"
    GPU_FFLAGS=${GPU_FFLAGS:--O3 -acc} "$python" \
        "$root/scripts/run_fortran_rbf_matmat.py" \
        --fortml "$fortml" --device cuda --backend openacc --n "$n" --d "$d" \
        --rhs "$rhs" --repetitions "$repetitions" \
        --output "$results_dir/rbf_matmat_openacc_${rhs}.csv"
    GPU_FFLAGS=${GPU_FFLAGS:--O3 -acc} "$python" \
        "$root/scripts/run_fortran_rbf_matmat.py" \
        --fortml "$fortml" --device cuda --backend native --n "$n" --d "$d" \
        --rhs "$rhs" --repetitions "$repetitions" \
        --output "$results_dir/rbf_matmat_native_${rhs}.csv"
done

inputs=("$results_dir"/rbf_matmat_cpu_*.csv)
inputs+=("$results_dir"/rbf_matmat_openacc_*.csv)
inputs+=("$results_dir"/rbf_matmat_native_*.csv)
"$python" "$root/scripts/merge_results.py" \
    "$results_dir/rbf_matmat.csv" "${inputs[@]}"
"$python" "$root/scripts/plot_matmat.py" \
    "$results_dir/rbf_matmat.csv" "$results_dir/rbf_matmat_rhs"
