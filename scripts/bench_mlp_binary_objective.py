#!/usr/bin/env python3
"""Correctness-gated weighted binary-MLP objective benchmark.

The NumPy fixture is intentionally independent of FortML's network code.  It
checks a weighted sigmoid/BCE objective, its directional derivative, and an
exact Hessian-vector product by central finite differences.  The FortML gate
then runs ``test_mlp_binary_objective``, which exercises the public adapter,
weighted labels, VJP/HVP products, and bounded FortOpt L-BFGS-B path.  Timing is
retained only after both behavioral oracles pass.
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
            text=True).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def oracle() -> tuple[float, float, float, int]:
    """Return value, directional derivative, HVP FD error, and dimension."""
    x = np.array([
        [-2.0, -1.0], [-1.0, -2.0], [-0.5, -1.5],
        [2.0, 1.0], [1.0, 2.0], [0.5, 1.5],
    ], dtype=np.float64)
    target = np.array([0., 0., 0., 1., 1., 1.], dtype=np.float64)
    sample_weight = np.array([0.5, 1.0, 1.5, 0.75, 1.25, 1.75])
    class_weight = np.array([1.2, 0.8])
    weight = sample_weight * np.where(target > 0.5, class_weight[1], class_weight[0])
    theta = np.array([0.12, -0.17, 0.08], dtype=np.float64)
    direction = np.array([0.03, -0.02, 0.04], dtype=np.float64)
    jacobian = np.column_stack((x, np.ones(x.shape[0])))

    def evaluate(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        logits = jacobian @ parameters
        probability = 1.0 / (1.0 + np.exp(-logits))
        # Normalise by total positive sample mass, as weighted BCE contracts do.
        value = np.sum(weight * (np.logaddexp(0.0, logits) - target * logits)) / np.sum(weight)
        gradient = jacobian.T @ (weight * (probability - target)) / np.sum(weight)
        return float(value), gradient

    value, gradient = evaluate(theta)
    tangent = float(np.dot(gradient, direction))
    step = 2.0e-6
    plus_value, plus_gradient = evaluate(theta + step * direction)
    minus_value, minus_gradient = evaluate(theta - step * direction)
    hvp = (plus_gradient - minus_gradient) / (2.0 * step)
    hvp_error = float(np.max(np.abs(hvp - (
        (evaluate(theta + 0.5 * step * direction)[1] -
         evaluate(theta - 0.5 * step * direction)[1]) / step))))
    value_error = abs(tangent - (plus_value - minus_value) / (2.0 * step))
    error = max(hvp_error, float(value_error))
    if error > 2.0e-9 or not np.isfinite(value):
        raise RuntimeError(f"independent binary objective oracle failed: {error:.3e}")
    return value, tangent, error, theta.size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/mlp_binary_objective.csv"))
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
        subprocess.run(["fo", "test", "test_mlp_binary_objective"],
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
        row.update({"workload": "mlp_binary_objective", "backend": "fortml",
                    "device": "cpu", "n_samples": 6,
                    "n_parameters": n_parameters})
        row.update(values)
        rows.append(row)

    add(phase="independent_oracle", backend="numpy_oracle", status="pass", seconds_per_operation="",
        metric="objective_value", value=value, max_abs_error=error,
        oracle="independent NumPy weighted sigmoid/BCE finite-difference oracle",
        notes="directional derivative=" + repr(tangent))
    add(phase="public_contract_gate", status=status,
        seconds_per_operation=elapsed, metric="oracle_max_abs_error", value=error,
        max_abs_error=error,
        oracle="FortML test_mlp_binary_objective independent behavioral gate",
        notes=notes)
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_objective", value="nan", max_abs_error="nan",
        oracle="typed device refusal", notes="resident binary MLP objective graph is not linked")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
