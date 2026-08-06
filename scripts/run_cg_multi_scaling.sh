#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python=${PYTHON:-$root/.venv/bin/python}
results_dir=${RESULTS_DIR:-$root/results}
sizes=${CG_MULTI_SIZES:-"256 512 1024 2048"}
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
inputs=()

for n in $sizes; do
    run_dir="$scratch/$n"
    mkdir -p "$run_dir"
    RESULTS_DIR="$run_dir" N_SAMPLES="$n" \
        "$root/scripts/run_cg_multi_suite.sh"
    inputs+=("$run_dir/rbf_cg_multi.csv")
done

mkdir -p "$results_dir"
"$python" "$root/scripts/merge_results.py" \
    "$results_dir/rbf_cg_multi_scaling.csv" "${inputs[@]}"
"$python" "$root/scripts/plot_cg_multi.py" \
    "$results_dir/rbf_cg_multi_scaling.csv" \
    "$results_dir/rbf_cg_multi_scaling"
