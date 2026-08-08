#!/usr/bin/env python3
"""Correctness-gated affine constant-schedule outer-HVP benchmark.

The NumPy recurrence is independent of FortML and uses central differences of
the complete packed gradient as its outer-HVP oracle.  FortML output is retained
only after value, gradient, JVP, and HVP arrays agree component by component.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np


TRAIN_COUNT = 96
VALIDATION_COUNT = 32
STEPS = 8
REPETITIONS = 64
BASE_RATE = 0.08
L2 = 0.03
INITIAL = np.array([0.15, -0.1], dtype=np.float64)
DIRECTION = np.array([0.31, -0.27, 0.18, -0.22], dtype=np.float64)
FD_STEP = 2.0e-5
TOLERANCE = 5.0e-8
FIELDS = (
    "workload", "phase", "backend", "device", "status", "steps",
    "repetitions", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.linspace(-2.0, 2.0, TRAIN_COUNT, dtype=np.float64)[:, None]
    validation_x = np.linspace(-1.8, 1.8, VALIDATION_COUNT, dtype=np.float64)[:, None]
    return (train_x, 0.7 * train_x - 0.2,
            validation_x, 0.7 * validation_x - 0.2)


def loss_gradient(theta: np.ndarray, x: np.ndarray, y: np.ndarray,
                  l2: float) -> tuple[float, np.ndarray]:
    prediction = x[:, 0] * theta[0] + theta[1]
    residual = prediction - y[:, 0]
    value = 0.5 * np.mean(residual * residual) + 0.5 * l2 * np.dot(theta, theta)
    gradient = np.array([np.mean(x[:, 0] * residual), np.mean(residual)]) + l2 * theta
    return float(value), gradient


def objective(parameters: np.ndarray) -> float:
    base_rate = math.exp(float(parameters[0]))
    l2 = math.exp(float(parameters[1]))
    train_x, train_y, validation_x, validation_y = fixture()
    theta = INITIAL.copy()
    for _ in range(STEPS):
        _, gradient = loss_gradient(theta, train_x, train_y, l2)
        theta -= base_rate * gradient
    value, _ = loss_gradient(theta, validation_x, validation_y, 0.0)
    return value


def gradient_oracle(parameters: np.ndarray) -> np.ndarray:
    gradient = np.empty(4, dtype=np.float64)
    for index in range(4):
        plus = parameters.copy(); plus[index] += FD_STEP
        minus = parameters.copy(); minus[index] -= FD_STEP
        gradient[index] = (objective(plus) - objective(minus)) / (2.0 * FD_STEP)
    return gradient


def oracle() -> tuple[float, np.ndarray, float, np.ndarray]:
    parameters = np.array([math.log(BASE_RATE), math.log(L2), 0.0, 0.0])
    value = objective(parameters)
    gradient = gradient_oracle(parameters)
    tangent = float(np.dot(gradient, DIRECTION))
    plus = parameters + FD_STEP * DIRECTION
    minus = parameters - FD_STEP * DIRECTION
    hvp = (gradient_oracle(plus) - gradient_oracle(minus)) / (2.0 * FD_STEP)
    return value, gradient, tangent, hvp


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def make_row(details: dict[str, str], phase: str, device: str, status: str,
             metric: str, value: object, error: object, seconds: object,
             notes: str) -> dict[str, object]:
    row: dict[str, object] = {field: "" for field in FIELDS}
    row.update(details)
    row.update({
        "workload": "mlp_constant_schedule_hvp", "phase": phase,
        "backend": "fortml", "device": device, "status": status,
        "steps": STEPS, "repetitions": REPETITIONS,
        "seconds_per_operation": seconds, "metric": metric, "value": value,
        "max_abs_error": error,
        "oracle": "independent NumPy affine recurrence with central-FD HVP",
        "notes": notes,
    })
    return row


def run_fortml(fortml: Path, target: str, oracle_path: Path) -> tuple[bool, str, dict[str, float]]:
    env = os.environ.copy()
    env["FORTML_BENCH_MLP_CONSTANT_SCHEDULE_HVP_ORACLE"] = str(oracle_path)
    env["FORTML_BENCH_ORACLE_ONLY"] = "1"
    completed = subprocess.run(["fo", "exec", target], cwd=fortml, env=env,
                               text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        return False, completed.stdout + completed.stderr, {}
    if not oracle_path.exists():
        return False, "release app did not write its complete-array oracle", {}
    values: dict[str, float] = {}
    with oracle_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            values[f"{row['quantity']}:{row['index']}"] = float(row["value"])
    return True, "complete-array oracle accepted", values


def timing_seconds(fortml: Path, target: str, prefix: str) -> float:
    completed = subprocess.run(["fo", "exec", target], cwd=fortml,
                               env=os.environ.copy(), text=True,
                               capture_output=True, check=True)
    for line in completed.stdout.splitlines():
        if line.startswith(prefix):
            return float(line.split(",", 1)[1].strip())
    raise RuntimeError(f"release app did not emit {prefix} timing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/mlp_constant_schedule_hvp.csv"))
    parser.add_argument("--target", default="fortml_bench_mlp_constant_schedule_hvp")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    details = metadata(root, fortml, output)
    expected_value, expected_gradient, expected_tangent, expected_hvp = oracle()
    oracle_fd, oracle_name = tempfile.mkstemp(
        prefix="fortml-constant-schedule-hvp-", suffix=".csv", dir="/mnt/storage"
    )
    os.close(oracle_fd)
    oracle_path = Path(oracle_name)
    oracle_path.unlink()
    success, note, values = run_fortml(fortml, args.target, oracle_path)
    oracle_path.unlink(missing_ok=True)
    rows: list[dict[str, object]] = []
    if success:
        observed_gradient = np.array([
            values.get(f"gradient:{i}", float("nan")) for i in range(1, 5)
        ])
        observed_hvp = np.array([
            values.get(f"hvp:{i}", float("nan")) for i in range(1, 5)
        ])
        observed_value = values.get("value:1", float("nan"))
        observed_tangent = values.get("jvp:1", float("nan"))
        errors = np.concatenate((
            [abs(observed_value - expected_value),
             abs(observed_tangent - expected_tangent)],
            abs(observed_gradient - expected_gradient),
            abs(observed_hvp - expected_hvp),
        ))
        passed = bool(np.all(np.isfinite(errors)) and np.max(errors) <= TOLERANCE)
        status = "pass" if passed else "fail"
        max_error = float(np.max(errors))
        timing_value = timing_seconds(
            fortml, args.target, "mlp_constant_schedule_hvp_value_gradient,") if passed else float("nan")
        timing_hvp = timing_seconds(
            fortml, args.target, "mlp_constant_schedule_hvp_hvp,") if passed else float("nan")
        rows.append(make_row(details, "value_gradient", "cpu", status,
                             "validation_mse", observed_value, max_error, timing_value, note))
        rows.append(make_row(details, "jvp", "cpu", status,
                             "directional_validation_mse_derivative", observed_tangent,
                             max_error, timing_value, note))
        for index, value in enumerate(observed_gradient, 1):
            rows.append(make_row(details, "gradient_component", "cpu", status,
                                 f"gradient_parameter_{index}", value, max_error,
                                 timing_value, note))
        for index, value in enumerate(observed_hvp, 1):
            rows.append(make_row(details, "hvp_component", "cpu", status,
                                 f"hvp_parameter_{index}", value, max_error,
                                 timing_hvp, note))
    else:
        rows.append(make_row(details, "value_gradient", "cpu", "unavailable",
                             "validation_mse", "", "", "", note))
    rows.append(make_row(details, "hvp", "cuda", "unavailable", "", "", "", "",
                         "resident CUDA trajectory kernel is not linked; no host fallback"))
    rows.append(make_row(details, "hvp", "cpu", "unavailable", "", "", "", "",
                         "nonconstant schedule and nonlinear MLPs retain typed HVP refusal"))
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
