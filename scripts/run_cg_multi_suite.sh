#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python=${PYTHON:-$root/.venv/bin/python}
fortml=${FORTML_DIR:-$root/../fortml}
results_dir=${RESULTS_DIR:-$root/results}
n=${N_SAMPLES:-2048}
d=${N_FEATURES:-8}
rhs=${N_RHS:-4}
repetitions=${REPETITIONS:-3}
tolerance=${CG_TOLERANCE:-1e-8}
max_iterations=${CG_MAX_ITERATIONS:-500}
cpu_threads=${CPU_THREADS:-$(lscpu -p=CORE 2>/dev/null | awk '!/^#/ {print $1}' | sort -u | wc -l)}
test "$cpu_threads" -gt 0 || cpu_threads=1
if [[ -n "${CPU_FC:-}" ]]; then
    cpu_compiler=$CPU_FC
elif command -v nvfortran >/dev/null 2>&1; then
    cpu_compiler=nvfortran
else
    cpu_compiler=gfortran
fi
if [[ "$cpu_compiler" == "nvfortran" ]]; then
    cpu_flags=${CPU_FFLAGS:--O3 -mp}
else
    cpu_flags=${CPU_FFLAGS:--O3 -march=native -fopenmp -fno-math-errno}
fi
gpu_flags=${GPU_FFLAGS:--O3 -acc}
mkdir -p "$results_dir"
preconditioner_args=()
if [[ -n "${NYSTROM_RANK:-}" ]]; then
    preconditioner_args=(--nystrom-rank "$NYSTROM_RANK")
fi

OMP_NUM_THREADS="$cpu_threads" FORTML_FC="$cpu_compiler" FFLAGS="$cpu_flags" \
    "$python" "$root/scripts/run_fortran_rbf_cg_multi.py" \
    --fortml "$fortml" --device cpu --n "$n" --d "$d" --rhs "$rhs" \
    --repetitions "$repetitions" --tolerance "$tolerance" \
    --max-iterations "$max_iterations" "${preconditioner_args[@]}" \
    --output "$results_dir/rbf_cg_multi_fortran_cpu.csv"
OMP_NUM_THREADS="$cpu_threads" "$python" "$root/scripts/bench_rbf_cg_multi.py" \
    --device cpu --n "$n" --d "$d" --rhs "$rhs" --threads "$cpu_threads" \
    --repetitions "$repetitions" --tolerance "$tolerance" \
    --max-iterations "$max_iterations" --output "$results_dir/rbf_cg_multi_python_cpu.csv"
if "$python" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    FFLAGS="$gpu_flags" "$python" "$root/scripts/run_fortran_rbf_cg_multi.py" \
        --fortml "$fortml" --device cuda --n "$n" --d "$d" --rhs "$rhs" \
        --repetitions "$repetitions" --tolerance "$tolerance" \
        --max-iterations "$max_iterations" "${preconditioner_args[@]}" \
        --output "$results_dir/rbf_cg_multi_fortran_cuda.csv"
    "$python" "$root/scripts/bench_rbf_cg_multi.py" \
        --device cuda --n "$n" --d "$d" --rhs "$rhs" \
        --repetitions "$repetitions" --tolerance "$tolerance" \
        --max-iterations "$max_iterations" --output "$results_dir/rbf_cg_multi_python_cuda.csv"
fi
"$python" "$root/scripts/merge_results.py" \
    "$results_dir/rbf_cg_multi.csv" \
    "$results_dir/rbf_cg_multi_fortran_cpu.csv" \
    "$results_dir/rbf_cg_multi_python_cpu.csv" \
    "$results_dir/rbf_cg_multi_fortran_cuda.csv" \
    "$results_dir/rbf_cg_multi_python_cuda.csv"
