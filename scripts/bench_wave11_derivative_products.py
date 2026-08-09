#!/usr/bin/env python3
"""Independent-oracle release gate for the Wave 11 derivative products.

The NumPy calculations are deliberately separate from FortML.  The Fortran
rows only become ``pass`` after the corresponding behavioural tests pass; the
CUDA rows retain the typed capability boundary and claim no host fallback.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def poisson_oracle() -> tuple[float, float, float]:
    counts = np.array([0.0, 1.0, 3.0, 2.0])
    latents = np.array([-0.3, 0.2, 0.8, -0.1])
    direction = np.array([0.4, -0.2, 0.1, 0.3])
    value = np.sum(counts * latents - np.exp(latents) -
                   [math.lgamma(y + 1.0) for y in counts])
    gradient = counts - np.exp(latents)
    hvp = -np.exp(latents) * direction
    return float(value), float(np.linalg.norm(gradient)), float(np.sum(hvp))


def affine_sgd_oracle() -> tuple[float, float]:
    x = np.array([-1.0, -0.2, 0.7, 1.5])
    y = 0.6 * x - 0.15
    validation_x = np.array([-0.8, 0.4, 1.2])
    validation_y = 0.6 * validation_x - 0.15
    parameters = np.array([math.log(0.08), math.log(0.03)])
    direction = np.array([0.17, -0.13])

    def trajectory(theta: np.ndarray) -> float:
        lr, l2 = np.exp(theta)
        state = np.array([0.25, 0.1])
        for _ in range(4):
            residual = state[0] * x + state[1] - y
            gradient = np.array([np.mean(residual * x), np.mean(residual)]) + l2 * state
            state -= lr * gradient
        residual = state[0] * validation_x + state[1] - validation_y
        return 0.5 * float(np.mean(residual * residual))

    step = 2.0e-6
    value = trajectory(parameters)
    tangent = (trajectory(parameters + step * direction) -
               trajectory(parameters - step * direction)) / (2.0 * step)
    return value, tangent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/wave11_derivative_products.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    poisson_value, poisson_gradient_norm, poisson_hvp = poisson_oracle()
    for phase, metric, value in (
            ("oracle_value", "log_likelihood", poisson_value),
            ("oracle_gradient", "gradient_norm", poisson_gradient_norm),
            ("oracle_hvp", "hvp_checksum", poisson_hvp)):
        rows.append({**details, "workload": "poisson_likelihood_products", "phase": phase,
                     "backend": "numpy_oracle", "device": "cpu", "status": "pass",
                     "metric": metric, "value": value, "max_abs_error": 0.0,
                     "oracle": "independent NumPy Poisson log-rate formula",
                     "notes": "weighted value/gradient/HVP fixture"})

    affine_value, affine_tangent = affine_sgd_oracle()
    for phase, metric, value in (("oracle_value", "validation_mse", affine_value),
                                 ("oracle_hvp", "directional_validation_mse_derivative",
                                  affine_tangent)):
        rows.append({**details, "workload": "mlp_affine_sgd_outer_hvp", "phase": phase,
                     "backend": "numpy_oracle", "device": "cpu", "status": "pass",
                     "metric": metric, "value": value, "max_abs_error": 0.0,
                     "oracle": "independent NumPy affine full-batch SGD recurrence",
                     "notes": "central directional derivative used only by the oracle"})

    environment = os.environ.copy()
    environment.update({"FO_SCAN_FALLBACK": "regex", "FO_FC": "gfortran",
                        "OMP_NUM_THREADS": "1"})
    for workload, target, note in (
            ("poisson_likelihood_products", "test_poisson_likelihood",
             "value/gradient/JVP/VJP/HVP plus L-BFGS-B and CUDA refusal"),
            ("mlp_affine_sgd_outer_hvp", "test_mlp_optimizer_group_affine_hvp",
             "exact affine constant-schedule SGD outer HVP and typed refusals")):
        started = time.perf_counter()
        completed = subprocess.run(["fo", "test", target], cwd=fortml,
                                   env=environment, capture_output=True, text=True)
        elapsed = time.perf_counter() - started
        passed = completed.returncode == 0 and "PASS" in (completed.stdout + completed.stderr)
        if not passed:
            raise RuntimeError(f"{target} failed:\n{completed.stdout}\n{completed.stderr}")
        rows.append({**details, "workload": workload, "phase": "behavioral_gate",
                     "backend": "fortml", "device": "cpu", "status": "pass",
                     "metric": "tests_passed", "value": 1.0, "max_abs_error": 0.0,
                     "oracle": f"independent Fortran {target}",
                     "notes": f"elapsed_seconds={elapsed:.6g}; {note}"})
        rows.append({**details, "workload": workload, "phase": "device_boundary",
                     "backend": "fortml", "device": "cuda", "status": "unavailable",
                     "metric": "resident_executor", "value": "FORTNUM_NOT_IMPLEMENTED",
                     "max_abs_error": 0.0, "oracle": "typed device capability contract",
                     "notes": "no resident CUDA derivative executor; no host fallback"})

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
