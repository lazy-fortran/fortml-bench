#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python=${PYTHON:-$root/.venv/bin/python}
fortml=${FORTML_DIR:-$root/../fortml}
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
    cpu_flags=${CPU_FFLAGS:--O3 -march=native -fopenmp -fno-math-errno -flto -fwhole-program}
fi
gpu_flags=${GPU_FFLAGS:--O3 -acc}
mkdir -p "$root/results"

OMP_NUM_THREADS="$cpu_threads" FORTML_FC="$cpu_compiler" FFLAGS="$cpu_flags" \
    "$python" "$root/scripts/run_fortran_rbf.py" \
    --fortml "$fortml" --device cpu --output "$root/results/rbf_fortran_cpu.csv"
OMP_NUM_THREADS="$cpu_threads" "$python" "$root/scripts/bench_rbf_mvm.py" \
    --device cpu --threads "$cpu_threads" \
    --output "$root/results/rbf_python_cpu.csv"
if "$python" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    FFLAGS="$gpu_flags" "$python" "$root/scripts/run_fortran_rbf.py" \
        --fortml "$fortml" --device cuda --output "$root/results/rbf_fortran_cuda.csv"
    "$python" "$root/scripts/bench_rbf_mvm.py" \
        --device cuda --output "$root/results/rbf_python_cuda.csv"
fi
"$python" "$root/scripts/merge_results.py" \
    "$root/results/rbf_mvm.csv" \
    "$root/results/rbf_fortran_cpu.csv" \
    "$root/results/rbf_python_cpu.csv" \
    "$root/results/rbf_fortran_cuda.csv" \
    "$root/results/rbf_python_cuda.csv"
"$python" "$root/scripts/plot_rbf_mvm.py" \
    "$root/results/rbf_mvm.csv" "$root/results/rbf_mvm.png"
