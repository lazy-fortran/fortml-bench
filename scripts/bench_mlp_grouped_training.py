#!/usr/bin/env python3
"""Correctness-gated grouped MLP regularization benchmark.

The NumPy fixture is the independent scalar linear-ridge oracle.  FortML's
packed objective is accepted only when its value, gradient norm, HVP norm, and
JVP agree with that oracle; CUDA remains an explicit refusal until a resident
MLP derivative graph exists.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
from pathlib import Path

import numpy as np


N_SAMPLES, REPETITIONS = 128, 128
PARAMETERS = np.array([0.4, -0.3, -1.0, -2.0], dtype=np.float64)
DIRECTION = np.array([0.17, -0.23, 0.31, -0.27], dtype=np.float64)
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_parameters", "repetitions", "seconds_per_operation", "metric",
    "value", "max_abs_error", "oracle", "python_version", "numpy_version",
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
            pass
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        if line[3:].strip() not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, N_SAMPLES)
    target = 0.8 * x + 0.2
    return x, target


def oracle() -> dict[str, float]:
    x, target = fixture()
    prediction = PARAMETERS[0] * x + PARAMETERS[1]
    residual = prediction - target
    value = 0.5 * float(np.mean(residual * residual))
    gradient = np.array([
        float(np.mean(x * residual)), float(np.mean(residual)),
        0.5 * np.exp(PARAMETERS[2]) * PARAMETERS[0] ** 2,
        0.5 * np.exp(PARAMETERS[3]) * PARAMETERS[1] ** 2,
    ])
    gradient[:2] += np.array([
        np.exp(PARAMETERS[2]) * PARAMETERS[0],
        np.exp(PARAMETERS[3]) * PARAMETERS[1],
    ])
    value += 0.5 * np.exp(PARAMETERS[2]) * PARAMETERS[0] ** 2
    value += 0.5 * np.exp(PARAMETERS[3]) * PARAMETERS[1] ** 2
    data_hessian = np.array([[np.mean(x * x), np.mean(x)], [np.mean(x), 1.0]])
    hessian_direction = data_hessian @ DIRECTION[:2]
    hvp = np.zeros(4)
    hvp[:2] = hessian_direction
    for first, log_index in ((0, 2), (1, 3)):
        lam = np.exp(PARAMETERS[log_index])
        theta = PARAMETERS[first]
        hvp[first] += lam * (DIRECTION[first] + theta * DIRECTION[log_index])
        hvp[log_index] = lam * theta * DIRECTION[first] + 0.5 * lam * theta * theta * DIRECTION[log_index]
    return {
        "value": value,
        "gradient_norm": float(np.linalg.norm(gradient)),
        "hvp_norm": float(np.linalg.norm(hvp)),
        "jvp": float(np.dot(gradient, DIRECTION)),
    }


def parse(output: str) -> dict[str, str]:
    records = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0].startswith("mlp_grouped_"):
            records[fields[0]] = fields[1]
    required = {
        "mlp_grouped_value_gradient_seconds", "mlp_grouped_value",
        "mlp_grouped_gradient_norm", "mlp_grouped_hvp_norm", "mlp_grouped_jvp",
        "mlp_grouped_cuda",
    }
    missing = required.difference(records)
    if missing:
        raise RuntimeError(f"FortML omitted grouped benchmark fields: {sorted(missing)}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/mlp_grouped_training.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    expected = oracle()
    metadata = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }

    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, check=True)
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_mlp_grouped_training"],
        cwd=fortml, capture_output=True, text=True, check=True,
    )
    records = parse(completed.stdout)
    tolerances = {"value": 2e-11, "gradient_norm": 2e-11, "hvp_norm": 2e-10, "jvp": 2e-10}
    rows = []
    for metric, tolerance in tolerances.items():
        observed = float(records[f"mlp_grouped_{metric}"])
        error = abs(observed - expected[metric])
        if error > tolerance:
            raise RuntimeError(f"grouped {metric} mismatch: {error:.3e}")
        rows.append({
            **metadata, "workload": "mlp_grouped_training", "phase": metric,
            "backend": "fortml", "device": "cpu", "status": "pass",
            "n_samples": N_SAMPLES, "n_parameters": 4, "repetitions": REPETITIONS,
            "seconds_per_operation": records["mlp_grouped_value_gradient_seconds"] if metric == "value" else "",
            "metric": metric, "value": observed, "max_abs_error": error,
            "oracle": "independent NumPy linear ridge value/derivative oracle",
            "notes": f"tolerance={tolerance:.1e}; named log-L2 groups",
        })
    if records["mlp_grouped_cuda"] != "unavailable":
        raise RuntimeError("grouped MLP CUDA refusal changed unexpectedly")
    rows.append({
        **metadata, "workload": "mlp_grouped_training", "phase": "device",
        "backend": "fortml", "device": "cuda", "status": "unavailable",
        "n_samples": N_SAMPLES, "n_parameters": 4, "repetitions": REPETITIONS,
        "seconds_per_operation": "", "metric": "", "value": "",
        "max_abs_error": "", "oracle": "typed_device_contract",
        "notes": "FORTNUM_NOT_IMPLEMENTED; resident MLP derivative graph is open",
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
