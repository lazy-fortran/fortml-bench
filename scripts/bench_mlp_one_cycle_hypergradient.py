#!/usr/bin/env python3
"""Correctness-gated one-cycle MLP schedule hypergradient benchmark.

The NumPy affine trajectory below is independent of FortML.  It checks the
validation value, all four packed outer derivatives, and a directional JVP
before recording timing and the explicit resident-CUDA boundary.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


STEPS = 5
BASE_RATE = 0.12
L2 = 0.07
WARMUP_UPDATES = 2
TOTAL_UPDATES = 8
PEAK_FRACTION = 1.8
FINAL_FRACTION = 0.08
REPETITIONS = 16
FD_STEP = 2.0e-6
TOLERANCE = 5.0e-8
DIRECTION = np.array([0.17, -0.13, 0.21, -0.19], dtype=np.float64)
FIELDS = (
    "workload", "phase", "backend", "device", "status", "steps",
    "repetitions", "seconds_per_operation", "metric", "value",
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
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=np.float64)
    train_y = 0.7 * train_x - 0.2
    validation_x = np.array([[-1.5], [0.5], [1.75]], dtype=np.float64)
    validation_y = 0.7 * validation_x - 0.2
    return train_x, train_y, validation_x, validation_y


def loss_gradient(theta: np.ndarray, x: np.ndarray, y: np.ndarray,
                  l2: float) -> tuple[float, np.ndarray]:
    prediction = x[:, 0] * theta[0] + theta[1]
    residual = prediction - y[:, 0]
    value = 0.5 * np.mean(residual * residual) + 0.5 * l2 * np.dot(theta, theta)
    gradient = np.array([np.mean(residual * x[:, 0]), np.mean(residual)], dtype=np.float64)
    return float(value), gradient + l2 * theta


def schedule_rate(update: int, base_rate: float, peak: float, final: float) -> float:
    if update <= WARMUP_UPDATES:
        factor = 1.0 + (peak - 1.0) * update / WARMUP_UPDATES
    else:
        progress = min(1.0, max(0.0, (update - WARMUP_UPDATES) /
                                 (TOTAL_UPDATES - WARMUP_UPDATES)))
        fraction = 0.5 * (1.0 + math.cos(math.pi * progress))
        factor = final + (peak - final) * fraction
    return base_rate * factor


def objective(parameters: np.ndarray) -> float:
    base_rate = math.exp(parameters[0])
    l2 = math.exp(parameters[1])
    peak = math.exp(parameters[2])
    final = math.exp(parameters[3])
    train_x, train_y, validation_x, validation_y = fixture()
    theta = np.array([0.15, -0.1], dtype=np.float64)
    for update in range(1, STEPS + 1):
        _, gradient = loss_gradient(theta, train_x, train_y, l2)
        theta = theta - schedule_rate(update, base_rate, peak, final) * gradient
    value, _ = loss_gradient(theta, validation_x, validation_y, 0.0)
    return value


def oracle() -> tuple[float, np.ndarray, float]:
    parameters = np.log([BASE_RATE, L2, PEAK_FRACTION, FINAL_FRACTION])
    value = objective(parameters)
    gradient = np.empty(4, dtype=np.float64)
    for index in range(4):
        plus = parameters.copy(); plus[index] += FD_STEP
        minus = parameters.copy(); minus[index] -= FD_STEP
        gradient[index] = (objective(plus) - objective(minus)) / (2.0 * FD_STEP)
    return value, gradient, float(np.dot(gradient, DIRECTION))


def make_row(details: dict[str, str], phase: str, backend: str, device: str,
             status: str, metric: str, value: object, error: object,
             seconds: object, notes: str) -> dict[str, object]:
    result: dict[str, object] = {field: "" for field in FIELDS}
    result.update(details)
    result.update({
        "workload": "mlp_one_cycle_hypergradient", "phase": phase,
        "backend": backend, "device": device, "status": status,
        "steps": STEPS, "repetitions": REPETITIONS, "seconds_per_operation": seconds,
        "metric": metric, "value": value, "max_abs_error": error,
        "oracle": "independent NumPy affine trajectory with central-FD outer oracle",
        "notes": notes,
    })
    return result


def run_fortml(fortml: Path, target: str, oracle_path: Path) -> tuple[bool, str, dict[str, float]]:
    env = os.environ.copy()
    env["FORTML_BENCH_MLP_ONE_CYCLE_HYPERGRADIENT_ORACLE"] = str(oracle_path)
    env["FORTML_BENCH_ORACLE_ONLY"] = "1"
    try:
        completed = subprocess.run(["fo", "exec", target], cwd=fortml, env=env,
                                   text=True, capture_output=True, check=False)
    except OSError as exc:
        return False, str(exc), {}
    if completed.returncode != 0:
        return False, completed.stdout + completed.stderr, {}
    if not oracle_path.exists():
        return False, "release app did not write its complete-array oracle", {}
    values: dict[str, float] = {}
    with oracle_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            values[f"{row['quantity']}:{row['index']}"] = float(row["value"])
    return True, "complete-array oracle accepted", values


def timing_seconds(fortml: Path, target: str) -> float:
    completed = subprocess.run(["fo", "exec", target], cwd=fortml,
                               env=os.environ.copy(), text=True,
                               capture_output=True, check=True)
    for line in completed.stdout.splitlines():
        if line.startswith("mlp_one_cycle_hypergradient_value_gradient,"):
            return float(line.split(",", 1)[1].strip())
    raise RuntimeError("release app did not emit one-cycle timing")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/mlp_one_cycle_hypergradient.csv"))
    parser.add_argument("--target", default="fortml_bench_mlp_one_cycle_hypergradient")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml), "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    expected_value, expected_gradient, expected_tangent = oracle()
    temp_root = Path(tempfile.mkdtemp(prefix="fortml-one-cycle-", dir="/mnt/storage"))
    oracle_path = temp_root / "mlp_one_cycle_hypergradient.csv"
    try:
        success, note, values = run_fortml(fortml, args.target, oracle_path)
        rows: list[dict[str, object]] = []
        if success:
            observed = [values.get(f"gradient:{index}", float("nan"))
                        for index in range(1, 5)]
            observed_value = values.get("value:1", float("nan"))
            observed_tangent = values.get("jvp:1", float("nan"))
            errors = [abs(observed_value - expected_value),
                      *[abs(a - b) for a, b in zip(observed, expected_gradient)],
                      abs(observed_tangent - expected_tangent)]
            passed = all(np.isfinite(errors)) and max(errors) <= TOLERANCE
            status = "pass" if passed else "fail"
            timing = timing_seconds(fortml, args.target) if passed else float("nan")
            rows.append(make_row(details, "value_gradient", "fortml", "cpu", status,
                                 "validation_mse", observed_value, max(errors), timing, note))
            rows.append(make_row(details, "jvp", "fortml", "cpu", status,
                                 "directional_validation_mse_derivative", observed_tangent,
                                 max(errors), timing, note))
            for index, component in enumerate(observed, 1):
                rows.append(make_row(details, "gradient_component", "fortml", "cpu", status,
                                     f"gradient_parameter_{index}", component, max(errors),
                                     timing, note))
        else:
            rows.append(make_row(details, "value_gradient", "fortml", "cpu", "unavailable",
                                 "validation_mse", "", "", "", note))
        for phase in ("value_gradient", "jvp"):
            rows.append(make_row(details, phase, "fortml", "cuda", "unavailable", "", "", "", "",
                                 "resident CUDA trajectory kernel is not linked; no host fallback"))
        rows.append(make_row(details, "hvp", "fortml", "cpu", "unavailable", "", "", "", "",
                             "outer hyper-HVP requires third network derivatives; typed refusal"))
        with output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {output}")
        if not success or not rows or rows[0]["status"] != "pass":
            raise SystemExit(1)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
