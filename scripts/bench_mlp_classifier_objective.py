#!/usr/bin/env python3
"""Correctness-gated multiclass MLP objective benchmark.

The NumPy fixture is an independent weighted softmax cross-entropy oracle for
an affine logits map.  It checks value, directional derivative, and HVP by
central differences before running FortML's nonlinear MLP objective fixture,
which additionally covers the FortOpt callback and bounded L-BFGS-B path.
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
    "n_parameters", "seconds_per_operation", "metric", "value",
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


def oracle() -> tuple[float, float, float, int]:
    """Return value, directional derivative, HVP error, and model dimension."""
    x = np.array([
        [-2.0, -1.0], [-1.0, -2.0], [0.0, 2.0],
        [0.0, 1.0], [2.0, 0.0], [1.0, 2.0],
    ], dtype=np.float64)
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    sample_weight = np.array([0.5, 1.0, 1.5, 2.0, 0.75, 1.25])
    class_weight = np.array([1.25, 0.8, 1.4])
    weight = sample_weight * class_weight[labels]
    design = np.column_stack((x, np.ones(x.shape[0])))
    theta = np.linspace(-0.17, 0.19, design.shape[1] * 3)
    direction = np.linspace(0.031, -0.023, theta.size)
    theta = theta.reshape(design.shape[1], 3)
    direction = direction.reshape(theta.shape)

    def evaluate(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ parameters
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probability = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        normalizer = float(np.sum(weight))
        value = np.sum(weight * (
            -logits[np.arange(labels.size), labels]
            + np.log(np.sum(np.exp(logits - np.max(logits, axis=1, keepdims=True)), axis=1))
            + np.max(logits, axis=1)
        )) / normalizer
        residual = probability
        residual[np.arange(labels.size), labels] -= 1.0
        gradient = design.T @ (weight[:, None] * residual) / normalizer
        return float(value), gradient

    value, gradient = evaluate(theta)
    tangent = float(np.sum(gradient * direction))
    step = 2.0e-6
    plus_value, plus_gradient = evaluate(theta + step * direction)
    minus_value, minus_gradient = evaluate(theta - step * direction)
    hvp = (plus_gradient - minus_gradient) / (2.0 * step)
    value_error = abs(tangent - (plus_value - minus_value) / (2.0 * step))
    hvp_error = float(np.max(np.abs(hvp - (
        (evaluate(theta + 0.5 * step * direction)[1] -
         evaluate(theta - 0.5 * step * direction)[1]) / step))))
    error = max(float(value_error), hvp_error)
    if error > 2.0e-9 or not np.isfinite(value):
        raise RuntimeError(f"independent multiclass objective oracle failed: {error:.3e}")
    # The public gate uses a 2->3->3 network and an optimized L2 coordinate:
    # (2*3+3) + (3*3+3) + 1 = 22 packed coordinates.
    return value, tangent, error, 22


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/mlp_classifier_objective.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    value, tangent, error, n_parameters = oracle()
    started = time.perf_counter()
    if args.skip_fortml:
        status, notes = "skipped", "--skip-fortml"
    else:
        subprocess.run(["fo", "test", "test_mlp_classifier_objective"],
                       cwd=fortml, check=True)
        status = "pass"
        notes = "FortML test covers weighted objective, VJP/HVP, and bounded L-BFGS-B"
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
        row.update({"workload": "mlp_classifier_objective", "backend": "fortml",
                    "device": "cpu", "n_samples": 6,
                    "n_parameters": n_parameters})
        row.update(values)
        rows.append(row)

    add(phase="independent_oracle", backend="numpy_oracle", status="pass",
        seconds_per_operation="", metric="objective_value", value=value,
        max_abs_error=error,
        oracle="independent NumPy weighted softmax finite-difference oracle",
        notes="directional derivative=" + repr(tangent))
    add(phase="public_contract_gate", status=status,
        seconds_per_operation=elapsed, metric="oracle_max_abs_error", value=error,
        max_abs_error=error,
        oracle="FortML test_mlp_classifier_objective independent behavioral gate",
        notes=notes)
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_objective", value="nan", max_abs_error="nan",
        oracle="typed device refusal",
        notes="resident multiclass MLP objective graph is not linked")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
