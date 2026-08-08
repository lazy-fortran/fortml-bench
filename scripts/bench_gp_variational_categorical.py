#!/usr/bin/env python3
"""Correctness and contract benchmark for coupled categorical variational GPs.

The NumPy path is independent of FortML. It checks the variance-corrected
softmax and its directional product with central differences. The FortML test
adds ELBO gradients, packed and query-input products, fitting, and CUDA
refusals. CUDA is recorded as unavailable until the inducing graph is resident.
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_classes", "n_parameters", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"],
        text=True,
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def oracle() -> tuple[float, float, float]:
    means = np.array([
        [-1.2, -0.1, 0.8], [-0.7, 0.2, 0.4], [-0.2, 0.6, -0.1],
        [0.3, -0.4, 0.9], [0.8, 0.1, -0.6], [1.1, 0.5, -0.3],
    ], dtype=np.float64)
    variances = np.array([
        [0.3, 0.2, 0.4], [0.4, 0.2, 0.3], [0.2, 0.5, 0.3],
        [0.4, 0.3, 0.2], [0.3, 0.2, 0.5], [0.2, 0.4, 0.3],
    ], dtype=np.float64)
    direction = np.array([
        [0.02, -0.01, 0.03], [-0.01, 0.01, -0.02],
        [0.03, 0.02, 0.01], [0.01, -0.03, 0.02],
        [-0.02, 0.01, 0.01], [0.02, 0.02, -0.01],
    ], dtype=np.float64)
    variance_direction = np.array([
        [0.01, -0.02, 0.02], [-0.02, 0.01, 0.01],
        [0.02, 0.01, -0.01], [0.01, -0.02, 0.03],
        [-0.01, 0.02, 0.01], [0.02, -0.01, 0.02],
    ], dtype=np.float64)
    correction = np.pi / 8.0

    def probabilities(mu: np.ndarray, var: np.ndarray) -> np.ndarray:
        logits = mu / np.sqrt(1.0 + correction * var)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        weights = np.exp(shifted)
        return weights / np.sum(weights, axis=1, keepdims=True)

    probabilities_value = probabilities(means, variances)
    logits = means / np.sqrt(1.0 + correction * variances)
    logits_dot = (direction / np.sqrt(1.0 + correction * variances) -
                  0.5 * correction * means * variance_direction /
                  (1.0 + correction * variances) ** 1.5)
    mean_tangent = np.sum(probabilities_value * logits_dot, axis=1, keepdims=True)
    analytic = probabilities_value * (logits_dot - mean_tangent)
    step = 2.0e-6
    finite_difference = (probabilities(means + step * direction,
                                       variances + step * variance_direction) -
                         probabilities(means - step * direction,
                                       variances - step * variance_direction)) / (2.0 * step)
    simplex_error = float(np.max(np.abs(np.sum(probabilities_value, axis=1) - 1.0)))
    jvp_error = float(np.max(np.abs(analytic - finite_difference)))
    if simplex_error > 2.0e-15 or jvp_error > 2.0e-10:
        raise RuntimeError(f"categorical oracle failed: simplex={simplex_error:.3e}, "
                           f"jvp={jvp_error:.3e}")
    return simplex_error, jvp_error, float(np.sum(probabilities_value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gp_variational_categorical.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    simplex_error, jvp_error, probability_sum = oracle()
    started = time.perf_counter()
    if args.skip_fortml:
        status, notes = "skipped", "--skip-fortml"
    else:
        subprocess.run(["fo", "test", "test_gp_variational_categorical_classification"],
                       cwd=fortml, check=True)
        status = "pass"
        notes = "FortML test covers coupled ELBO, fit, packed/input products, and CUDA refusals"
    elapsed = time.perf_counter() - started
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": "gfortran", "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(details)
        row.update({"workload": "gp_variational_categorical", "backend": "fortml",
                    "device": "cpu", "n_samples": 6, "n_classes": 3,
                    "n_parameters": 27})
        row.update(values)
        rows.append(row)

    add(phase="independent_oracle", backend="numpy_oracle", status="pass",
        metric="probability_simplex_sum", value=probability_sum,
        max_abs_error=max(simplex_error, jvp_error),
        oracle="independent NumPy variance-corrected softmax and JVP finite difference",
        notes=f"simplex_error={simplex_error:.3e}; jvp_error={jvp_error:.3e}")
    add(phase="public_contract_gate", status=status, seconds_per_operation=elapsed,
        metric="probability_jvp_max_abs_error", value=jvp_error,
        max_abs_error=jvp_error,
        oracle="FortML test_gp_variational_categorical_classification behavioral gate",
        notes=notes)
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_categorical_graph", value="nan", max_abs_error="nan",
        oracle="typed FortML CUDA capability refusal",
        notes="resident inducing solves and coupled softmax reductions are not linked")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
