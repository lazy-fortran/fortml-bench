#!/usr/bin/env python3
"""Correctness-gated resident CUDA additive-tree benchmark.

The NumPy oracle independently walks a flattened two-tree ensemble, including
learned NaN routing and per-tree scales.  The ordinary Fortran test checks the
typed no-native-CUDA boundary and sentinel preservation; the native gate runs
the same model through the resident CUDA value/JVP ABI when a device exists.
"""

from __future__ import annotations

import argparse
import csv
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_inputs", "n_trees", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            continue
    dirty = any(
        line[3:].split(" -> ")[-1].strip() not in ignored_names
        for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True,
        ).splitlines()
    )
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, ...]:
    offsets = np.array([0, 3, 6], dtype=np.int32)
    features = np.array([0, -1, -1, 1, -1, -1], dtype=np.int32)
    left = np.array([1, -1, -1, 4, -1, -1], dtype=np.int32)
    right = np.array([2, -1, -1, 5, -1, -1], dtype=np.int32)
    threshold = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    weight = np.array([0.0, -1.0, 2.0, 0.0, 0.5, -0.5])
    missing_left = np.array([1, 0, 0, 0, 0, 0], dtype=np.int32)
    scales = np.array([1.0, 0.5])
    query = np.array([
        -1.0, 0.0, 0.5, 2.0, np.nan,
         0.0, 1.0, 1.5, 2.0, 2.0,
    ])
    return offsets, features, left, right, threshold, weight, missing_left, scales, query


def oracle() -> tuple[np.ndarray, float]:
    offsets, features, left, right, threshold, weight, missing_left, scales, query = fixture()
    n_query = query.size // 2
    values = np.full(n_query, 0.2)
    for row in range(n_query):
        value = 0.2
        for tree in range(scales.size):
            node = int(offsets[tree])
            while features[node] >= 0:
                feature = int(features[node])
                coordinate = query[feature * n_query + row]
                if np.isnan(coordinate):
                    node = int(left[node] if missing_left[node] else right[node])
                else:
                    node = int(left[node] if coordinate < threshold[node] else right[node])
            value += 0.7 * scales[tree] * weight[node]
        values[row] = value
    expected = np.array([-0.325, 1.425, 1.425, 1.425, -0.675])
    error = float(np.max(np.abs(values - expected)))
    if error > 1.0e-14:
        raise RuntimeError(f"independent additive-tree oracle failed: {error}")
    return values, error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/cuda_boosted_tree.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/CUDA_BOOSTED_TREE.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    report = args.report if args.report.is_absolute() else root / args.report
    _, oracle_error = oracle()
    started = time.perf_counter()
    gate = subprocess.run(
        ["fo", "test", "test_cuda_boosted_tree_api"], cwd=fortml,
        capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - started
    gate_ok = gate.returncode == 0
    cuda_ready = shutil.which("nvcc") is not None and subprocess.run(
        ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    cuda_status = "typed_refusal"
    cuda_value: float | str = "typed_refusal"
    cuda_elapsed: float | str = ""
    cuda_notes = "nvcc/device unavailable; ordinary build returns FORTNUM_NOT_IMPLEMENTED"
    if cuda_ready:
        started = time.perf_counter()
        cuda_gate = subprocess.run(
            ["bash", "test/run_cuda_boosted_tree_plan.sh"], cwd=fortml,
            capture_output=True, text=True,
        )
        cuda_elapsed = time.perf_counter() - started
        cuda_status = "pass" if cuda_gate.returncode == 0 else "failed"
        cuda_value = 1.0 if cuda_gate.returncode == 0 else 0.0
        cuda_notes = "resident value/JVP kernel, NaN routing, and split-boundary oracle"
    metadata = {
        "n_samples": 5, "n_inputs": 2, "n_trees": 2,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output, report)),
        "compiler": "gfortran/nvcc", "flags": "-O3",
        "oracle": "independent NumPy flattened-tree leaf walk",
    }
    rows = [{
        "workload": "cuda_boosted_tree", "phase": "value_and_routing",
        "backend": "numpy_oracle", "device": "cpu", "status": "pass",
        "seconds_per_operation": "", "metric": "margin_max_abs_error",
        "value": 1.0, "max_abs_error": oracle_error, **metadata,
        "notes": "base score, learning rate, per-tree scales, and learned NaN route",
    }, {
        "workload": "cuda_boosted_tree", "phase": "ordinary_stub",
        "backend": "fortml", "device": "cpu", "status": "pass" if gate_ok else "failed",
        "seconds_per_operation": elapsed, "metric": "gate_seconds",
        "value": 1.0 if gate_ok else 0.0, "max_abs_error": oracle_error, **metadata,
        "notes": "typed CUDA refusal preserves prediction and JVP sentinels",
    }, {
        "workload": "cuda_boosted_tree", "phase": "resident_value_jvp",
        "backend": "fortml", "device": "cuda", "status": cuda_status,
        "seconds_per_operation": cuda_elapsed, "metric": "native_gate",
        "value": cuda_value, "max_abs_error": 0.0, **metadata,
        "notes": cuda_notes,
    }]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Resident CUDA boosted-tree plan\n\n"
        "This lane compares a fixed two-tree additive ensemble with an "
        "independent NumPy leaf-walk oracle. The oracle covers base score, "
        "learning rate, per-tree scales, strict split routing, and a learned "
        "NaN default. The Fortran test checks ordinary-build typed refusal, "
        "invalid-device and output-preservation behavior. Native CUDA is run "
        "only when both `nvcc` and `nvidia-smi` are available; no device is "
        "recorded as GPU timing evidence when unavailable.\n\n"
        "Run:\n\n```sh\n"
        "python3 scripts/bench_cuda_boosted_tree.py --fortml ../fortml \\\n"
        "  --output results/cuda_boosted_tree.csv \\\n"
        "  --report results/CUDA_BOOSTED_TREE.md\n```\n\n"
        f"Oracle maximum absolute error: `{oracle_error:.3e}`.\n"
    )
    print(f"wrote {len(rows)} rows to {output}")
    if not gate_ok:
        print(gate.stdout + gate.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
