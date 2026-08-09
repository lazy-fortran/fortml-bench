#!/usr/bin/env python3
"""Benchmark weighted fixed-trajectory RMSprop hypergradients.

The oracle is an independent NumPy recurrence.  Central differences of that
recurrence certify the five packed derivatives, directional JVP, and affine
outer HVP before FortML timing rows are retained.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


STEPS = 4
N_TRAIN = 5
N_VALIDATION = 3
N_PARAMETERS = 5
REPETITIONS = 32
FD_STEP = 2.0e-6
HVP_STEP = 2.0e-4
PARAMETERS = np.array([np.log(0.12), np.log(0.07), 0.78, np.log(0.03), 0.21])
DIRECTION = np.array([0.31, -0.27, 0.17, -0.13, 0.19])
TRAIN_WEIGHT = np.array([0.25, 1.5, 0.0, 2.0, 0.75])
VALIDATION_WEIGHT = np.array([2.0, 0.5, 1.25])
FIELDS = (
    "workload", "phase", "variant", "backend", "device", "status",
    "n_train", "n_validation", "n_parameters", "steps", "repetitions",
    "seconds_per_operation", "metric", "value", "max_abs_error", "oracle",
    "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    )
    for line in status.splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    train_target = 0.7 * train_x - 0.2
    validation_x = np.array([-1.5, 0.5, 1.75])
    validation_target = 0.7 * validation_x - 0.2
    return train_x, train_target, validation_x, validation_target


def trajectory(parameters: np.ndarray) -> float:
    train_x, train_target, validation_x, validation_target = fixture()
    learning_rate, l2 = np.exp(parameters[:2])
    decay, epsilon, momentum = parameters[2], np.exp(parameters[3]), parameters[4]
    theta = np.array([0.15, -0.1])
    square_average = np.zeros(2)
    gradient_average = np.zeros(2)
    momentum_buffer = np.zeros(2)
    for _ in range(STEPS):
        residual = theta[0] * train_x + theta[1] - train_target
        gradient = np.array([
            np.dot(TRAIN_WEIGHT, residual * train_x) / TRAIN_WEIGHT.sum()
            + l2 * theta[0],
            np.dot(TRAIN_WEIGHT, residual) / TRAIN_WEIGHT.sum() + l2 * theta[1],
        ])
        square_average = decay * square_average + (1.0 - decay) * gradient**2
        gradient_average = decay * gradient_average + (1.0 - decay) * gradient
        variance = np.maximum(square_average - gradient_average**2, 0.0)
        update = gradient / (np.sqrt(variance) + epsilon)
        momentum_buffer = momentum * momentum_buffer + update
        theta = theta - learning_rate * momentum_buffer
    residual = theta[0] * validation_x + theta[1] - validation_target
    return 0.5 * float(
        np.dot(VALIDATION_WEIGHT, residual**2) / VALIDATION_WEIGHT.sum()
    )


def finite_difference_gradient(parameters: np.ndarray) -> np.ndarray:
    gradient = np.empty(N_PARAMETERS)
    for index in range(N_PARAMETERS):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += FD_STEP
        minus[index] -= FD_STEP
        gradient[index] = (trajectory(plus) - trajectory(minus)) / (2.0 * FD_STEP)
    return gradient


def oracle() -> dict[str, Any]:
    value = trajectory(PARAMETERS)
    gradient = finite_difference_gradient(PARAMETERS)
    jvp = (
        trajectory(PARAMETERS + FD_STEP * DIRECTION)
        - trajectory(PARAMETERS - FD_STEP * DIRECTION)
    ) / (2.0 * FD_STEP)
    hvp = (
        finite_difference_gradient(PARAMETERS + HVP_STEP * DIRECTION)
        - finite_difference_gradient(PARAMETERS - HVP_STEP * DIRECTION)
    ) / (2.0 * HVP_STEP)
    started = time.perf_counter()
    for _ in range(REPETITIONS):
        finite_difference_gradient(PARAMETERS)
    seconds = (time.perf_counter() - started) / REPETITIONS
    if not all(np.all(np.isfinite(item)) for item in (value, gradient, jvp, hvp)):
        raise RuntimeError("weighted RMSprop NumPy oracle is nonfinite")
    return {
        "value": value,
        "gradient": gradient,
        "jvp": jvp,
        "hvp": hvp,
        "seconds": seconds,
    }


def base(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    row.update(details)
    row.update({
        "device": "cpu",
        "n_train": N_TRAIN,
        "n_validation": N_VALIDATION,
        "n_parameters": N_PARAMETERS,
        "steps": STEPS,
        "repetitions": REPETITIONS,
    })
    row.update(values)
    return row


def oracle_rows(details: dict[str, str], expected: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [base(
        details,
        workload="mlp_rmsprop_weighted_hypergradient",
        phase="value_gradient",
        variant="centered_nonuniform_rows",
        backend="numpy_oracle",
        status="pass",
        seconds_per_operation=expected["seconds"],
        metric="weighted_validation_mse",
        value=expected["value"],
        max_abs_error=0.0,
        oracle="independent NumPy weighted RMSprop recurrence with central differences",
        notes="packed=[log_lr,log_l2,decay,log_epsilon,momentum]",
    )]
    for index, value in enumerate(expected["gradient"], start=1):
        rows.append(base(
            details,
            workload="mlp_rmsprop_weighted_hypergradient",
            phase="gradient_component",
            variant="centered_nonuniform_rows",
            backend="numpy_oracle",
            status="pass",
            metric=f"gradient_{index}",
            value=float(value),
            max_abs_error=0.0,
            oracle="independent central finite difference of weighted trajectory",
            notes=f"coordinate={index}; h={FD_STEP:g}",
        ))
    rows.append(base(
        details,
        workload="mlp_rmsprop_weighted_hypergradient",
        phase="jvp",
        variant="centered_nonuniform_rows",
        backend="numpy_oracle",
        status="pass",
        metric="weighted_validation_jvp",
        value=expected["jvp"],
        max_abs_error=0.0,
        oracle="independent central finite-difference directional product",
        notes=f"direction={DIRECTION.tolist()}; h={FD_STEP:g}",
    ))
    for index, value in enumerate(expected["hvp"], start=1):
        rows.append(base(
            details,
            workload="mlp_rmsprop_weighted_hypergradient",
            phase="hvp_component",
            variant="centered_nonuniform_rows",
            backend="numpy_oracle",
            status="pass",
            metric=f"hvp_{index}",
            value=float(value),
            max_abs_error=0.0,
            oracle="independent nested central differences of weighted trajectory",
            notes=f"direction={DIRECTION.tolist()}; outer_h={HVP_STEP:g}",
        ))
    return rows


def parse_output(stdout: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in stdout.splitlines():
        if line.startswith("rmsprop_weighted_"):
            key, value = line.split(",", 1)
            values[key] = float(value)
    return values


def run_fortml(
    fortml: Path, target: str, details: dict[str, str], expected: dict[str, Any]
) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({
        "FO_FC": environment.get("FO_FC", "gfortran"),
        "FO_SCAN_FALLBACK": environment.get("FO_SCAN_FALLBACK", "regex"),
        "OMP_NUM_THREADS": "1",
    })
    build = subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
        capture_output=True, text=True,
    )
    if build.returncode != 0:
        raise RuntimeError(f"FortML optimized build failed:\n{build.stderr}")
    run = subprocess.run(
        ["fo", "exec", "--no-build", target], cwd=fortml, env=environment,
        capture_output=True, text=True,
    )
    if run.returncode != 0:
        raise RuntimeError(f"FortML release app failed:\n{run.stderr}")
    actual = parse_output(run.stdout)
    required = {
        "rmsprop_weighted_value", "rmsprop_weighted_jvp",
        "rmsprop_weighted_train_mass", "rmsprop_weighted_validation_mass",
        "rmsprop_weighted_gradient_seconds", "rmsprop_weighted_hvp_seconds",
        "rmsprop_weighted_cuda_status",
    }
    required |= {f"rmsprop_weighted_gradient_{index}" for index in range(1, 6)}
    required |= {f"rmsprop_weighted_hvp_{index}" for index in range(1, 6)}
    missing = sorted(required - set(actual))
    if missing:
        raise RuntimeError(f"FortML release app omitted: {', '.join(missing)}")

    actual_gradient = np.array([
        actual[f"rmsprop_weighted_gradient_{index}"] for index in range(1, 6)
    ])
    actual_hvp = np.array([
        actual[f"rmsprop_weighted_hvp_{index}"] for index in range(1, 6)
    ])
    value_error = abs(actual["rmsprop_weighted_value"] - expected["value"])
    gradient_error = np.abs(actual_gradient - expected["gradient"])
    jvp_error = abs(actual["rmsprop_weighted_jvp"] - expected["jvp"])
    hvp_error = np.abs(actual_hvp - expected["hvp"])
    mass_error = max(
        abs(actual["rmsprop_weighted_train_mass"] - TRAIN_WEIGHT.sum()),
        abs(actual["rmsprop_weighted_validation_mass"] - VALIDATION_WEIGHT.sum()),
    )
    if max(value_error, float(gradient_error.max()), jvp_error) > 5.0e-7:
        raise RuntimeError("FortML weighted RMSprop value/gradient/JVP mismatch")
    if float(hvp_error.max()) > 6.0e-5:
        raise RuntimeError("FortML weighted RMSprop HVP mismatch")
    if mass_error > 1.0e-14:
        raise RuntimeError("FortML weighted RMSprop metadata mass mismatch")
    if int(actual["rmsprop_weighted_cuda_status"]) <= 0:
        raise RuntimeError("FortML weighted RMSprop CUDA request was not refused")

    common = {
        "workload": "mlp_rmsprop_weighted_hypergradient",
        "variant": "centered_nonuniform_rows",
        "backend": "fortml",
        "status": "pass",
        "oracle": "independent NumPy weighted recurrence and central differences",
        "notes": target,
    }
    rows = [base(
        details,
        **common,
        phase="value_gradient",
        seconds_per_operation=actual["rmsprop_weighted_gradient_seconds"],
        metric="weighted_validation_mse",
        value=actual["rmsprop_weighted_value"],
        max_abs_error=max(value_error, float(gradient_error.max())),
    )]
    for index, value in enumerate(actual_gradient, start=1):
        rows.append(base(
            details,
            **common,
            phase="gradient_component",
            metric=f"gradient_{index}",
            value=float(value),
            max_abs_error=float(gradient_error[index - 1]),
        ))
    rows.append(base(
        details,
        **common,
        phase="jvp",
        metric="weighted_validation_jvp",
        value=actual["rmsprop_weighted_jvp"],
        max_abs_error=jvp_error,
    ))
    for index, value in enumerate(actual_hvp, start=1):
        rows.append(base(
            details,
            **common,
            phase="hvp_component",
            seconds_per_operation=actual["rmsprop_weighted_hvp_seconds"],
            metric=f"hvp_{index}",
            value=float(value),
            max_abs_error=float(hvp_error[index - 1]),
        ))
    rows.append(base(
        details,
        workload="mlp_rmsprop_weighted_hypergradient",
        phase="device_contract",
        variant="centered_nonuniform_rows",
        backend="fortml",
        device="cuda",
        status="refused",
        metric="resident_cuda_trajectory",
        oracle="FortML typed device contract",
        notes=("CUDA trajectory execution is explicitly refused; no silent CPU "
               f"fallback (status={int(actual['rmsprop_weighted_cuda_status'])})"),
    ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/mlp_rmsprop_weighted_hypergradient.csv"),
    )
    parser.add_argument(
        "--target", default="fortml_bench_rmsprop_weighted_hypergradient"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    expected = oracle()
    details = metadata(root, fortml, output)
    rows = oracle_rows(details, expected)
    rows.extend(run_fortml(fortml, args.target, details, expected))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
