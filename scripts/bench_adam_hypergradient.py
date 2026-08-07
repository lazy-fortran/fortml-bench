#!/usr/bin/env python3
"""Correctness-gated coupled-L2 Adam trajectory hypergradient benchmark.

The NumPy path is an independent two-parameter linear-MSE recurrence.  It
finite-differences the validation objective in all four packed coordinates and
in one direction.  A FortML timing is retained only after its complete
value/gradient/JVP array agrees with that oracle.  CUDA is an explicit
unavailable capability row: no host trajectory is relabeled as GPU evidence.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


STEPS = 4
LEARNING_RATE = 0.12
L2 = 0.07
BETA1 = 0.82
BETA2 = 0.91
EPSILON = 0.03
FD_STEP = 2.0e-6
REPETITIONS = 32
ORACLE_TOLERANCE = 3.0e-6
DIRECTION = np.array([0.31, -0.27, 0.13, -0.22], dtype=np.float64)
PARAMETERS = np.array([
    np.log(LEARNING_RATE), np.log(L2),
    np.log(BETA1 / (1.0 - BETA1)), np.log(BETA2 / (1.0 - BETA2)),
], dtype=np.float64)

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
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines():
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
    train_x = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]], dtype=np.float64)
    train_target = 0.7 * train_x - 0.2
    validation_x = np.array([[-1.5], [0.5], [1.75]], dtype=np.float64)
    validation_target = 0.7 * validation_x - 0.2
    return train_x, train_target, validation_x, validation_target


def loss_gradient(theta: np.ndarray, x: np.ndarray, target: np.ndarray,
                  l2: float) -> np.ndarray:
    residual = x[:, 0] * theta[0] + theta[1] - target[:, 0]
    return np.array([
        float(np.mean(residual * x[:, 0])) + l2 * theta[0],
        float(np.mean(residual)) + l2 * theta[1],
    ])


def sigmoid(value: float) -> float:
    if value >= 0.0:
        return float(1.0 / (1.0 + np.exp(-value)))
    exp_value = np.exp(value)
    return float(exp_value / (1.0 + exp_value))


def trajectory(parameters: np.ndarray) -> float:
    learning_rate, l2 = np.exp(parameters[:2])
    beta1, beta2 = sigmoid(float(parameters[2])), sigmoid(float(parameters[3]))
    train_x, train_target, validation_x, validation_target = fixture()
    theta = np.array([0.15, -0.1], dtype=np.float64)
    first = np.zeros(2, dtype=np.float64)
    second = np.zeros(2, dtype=np.float64)
    for step in range(1, STEPS + 1):
        gradient = loss_gradient(theta, train_x, train_target, l2)
        first = beta1 * first + (1.0 - beta1) * gradient
        second = beta2 * second + (1.0 - beta2) * gradient * gradient
        first_hat = first / (1.0 - beta1**step)
        second_hat = second / (1.0 - beta2**step)
        theta = theta - learning_rate * first_hat / (np.sqrt(second_hat) + EPSILON)
    residual = validation_x[:, 0] * theta[0] + theta[1] - validation_target[:, 0]
    return 0.5 * float(np.mean(residual * residual))


def oracle() -> dict[str, Any]:
    value = trajectory(PARAMETERS)
    gradient = np.empty(4, dtype=np.float64)
    started = time.perf_counter()
    for index in range(4):
        plus, minus = PARAMETERS.copy(), PARAMETERS.copy()
        plus[index] += FD_STEP
        minus[index] -= FD_STEP
        gradient[index] = (trajectory(plus) - trajectory(minus)) / (2.0 * FD_STEP)
    tangent = (trajectory(PARAMETERS + FD_STEP * DIRECTION)
               - trajectory(PARAMETERS - FD_STEP * DIRECTION)) / (2.0 * FD_STEP)
    elapsed = (time.perf_counter() - started) / REPETITIONS
    if not np.all(np.isfinite(gradient)) or not np.isfinite(value) or not np.isfinite(tangent):
        raise RuntimeError("coupled Adam NumPy oracle is nonfinite")
    return {"value": value, "gradient": gradient, "tangent": tangent,
            "seconds": elapsed}


def base(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    row.update(details)
    row.update({
        "variant": "fixed_full_batch_coupled_l2_adam",
        "device": "cpu", "n_train": 5, "n_validation": 3,
        "n_parameters": 4, "steps": STEPS, "repetitions": REPETITIONS,
        "oracle": "independent NumPy coupled-L2 Adam recurrence",
    })
    row.update(values)
    return row


def oracle_rows(details: dict[str, str], expected: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [base(
        details, workload="mlp_adam_hypergradient", phase="value_gradient",
        backend="numpy_oracle", status="pass", seconds_per_operation=expected["seconds"],
        metric="validation_mse", value=expected["value"], max_abs_error=0.0,
        notes="packed=[log_lr,log_l2,logit_beta1,logit_beta2]; coupled L2",
    ), base(
        details, workload="mlp_adam_hypergradient", phase="jvp",
        backend="numpy_oracle", status="pass", seconds_per_operation=expected["seconds"],
        metric="directional_validation_mse_derivative", value=expected["tangent"],
        max_abs_error=0.0, notes=f"direction={DIRECTION.tolist()}; h={FD_STEP:g}",
    )]
    names = ("log_learning_rate", "log_l2", "logit_beta1", "logit_beta2")
    rows.extend(base(
        details, workload="mlp_adam_hypergradient", phase="gradient_component",
        backend="numpy_oracle", status="pass", repetitions=1,
        metric=f"gradient_{name}", value=float(value), max_abs_error=0.0,
        notes="independent central finite difference",
    ) for name, value in zip(names, expected["gradient"]))
    return rows


def unavailable_rows(details: dict[str, str], backend: str, device: str,
                     notes: str) -> list[dict[str, Any]]:
    rows = [base(
        details, workload="mlp_adam_hypergradient", phase="value_gradient",
        backend=backend, device=device, status="unavailable",
        oracle="FortML release-app protocol", notes=notes,
    ), base(
        details, workload="mlp_adam_hypergradient", phase="jvp",
        backend=backend, device=device, status="unavailable",
        metric="directional_validation_mse_derivative",
        oracle="FortML release-app protocol", notes=notes,
    )]
    rows.extend(base(
        details, workload="mlp_adam_hypergradient", phase="gradient_component",
        backend=backend, device=device, status="unavailable", metric=f"gradient_{name}",
        oracle="FortML release-app protocol", notes=notes,
    ) for name in ("log_learning_rate", "log_l2", "logit_beta1", "logit_beta2"))
    return rows


def read_oracle(path: Path) -> dict[tuple[str, int], float]:
    values: dict[tuple[str, int], float] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            values[(row["quantity"], int(row["index"]))] = float(row["value"])
    return values


def run_fortml(fortml: Path, target: str, details: dict[str, str],
               expected: dict[str, Any]) -> list[dict[str, Any]]:
    source = fortml / "app" / f"{target}.f90"
    if not source.is_file():
        return unavailable_rows(details, "fortml", "cpu", f"source absent: {source.name}")
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                           env=environment, capture_output=True, text=True)
    if build.returncode != 0:
        return unavailable_rows(details, "fortml", "cpu", "fo build failed")
    with tempfile.TemporaryDirectory(dir="/mnt/storage", prefix="fortml-adam-hypergradient-") as directory:
        oracle_path = Path(directory) / "oracle.csv"
        check_environment = dict(environment)
        check_environment.update({
            "FORTML_BENCH_ADAM_HYPERGRADIENT_ORACLE": str(oracle_path),
            "FORTML_BENCH_ORACLE_ONLY": "1",
        })
        check = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                               env=check_environment, capture_output=True, text=True)
        if check.returncode != 0 or not oracle_path.is_file():
            return unavailable_rows(details, "fortml", "cpu", "release app emitted no complete oracle")
        actual = read_oracle(oracle_path)
        required = {("value", 1), ("jvp", 1)} | {("gradient", i) for i in range(1, 5)}
        if set(actual) != required:
            raise RuntimeError("FortML coupled Adam app omitted a complete array")
        errors = [abs(actual[("value", 1)] - expected["value"]),
                  abs(actual[("jvp", 1)] - expected["tangent"])]
        errors.extend(abs(actual[("gradient", i)] - expected["gradient"][i - 1])
                      for i in range(1, 5))
        error = float(max(errors))
        if error > ORACLE_TOLERANCE:
            raise RuntimeError(f"FortML coupled Adam oracle mismatch: {error:.3e}")
        timed = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                               env=environment, capture_output=True, text=True)
    if timed.returncode != 0:
        return unavailable_rows(details, "fortml", "cpu", "release timing failed")
    marker = "mlp_adam_hypergradient_value_gradient,"
    timing = next((float(line.split(",", 1)[1].strip())
                   for line in timed.stdout.splitlines() if line.startswith(marker)), None)
    if timing is None:
        raise RuntimeError("FortML coupled Adam app emitted no timing marker")
    rows = [base(
        details, workload="mlp_adam_hypergradient", phase="value_gradient",
        backend="fortml", status="pass", seconds_per_operation=timing,
        metric="validation_mse", value=actual[("value", 1)], max_abs_error=error,
        oracle="complete NumPy value/gradient/JVP array", notes=target,
    ), base(
        details, workload="mlp_adam_hypergradient", phase="jvp", backend="fortml",
        status="pass", metric="directional_validation_mse_derivative",
        value=actual[("jvp", 1)], max_abs_error=error,
        oracle="complete NumPy value/gradient/JVP array", notes=target,
    )]
    names = ("log_learning_rate", "log_l2", "logit_beta1", "logit_beta2")
    rows.extend(base(
        details, workload="mlp_adam_hypergradient", phase="gradient_component",
        backend="fortml", status="pass", repetitions=1, metric=f"gradient_{name}",
        value=actual[("gradient", i)], max_abs_error=error,
        oracle="complete NumPy value/gradient/JVP array", notes=target,
    ) for i, name in enumerate(names, start=1))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/adam_hypergradient.csv"))
    parser.add_argument("--target", default="fortml_bench_adam_hypergradient")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    details = metadata(root, fortml, output)
    expected = oracle()
    rows = oracle_rows(details, expected)
    if args.skip_fortml:
        rows.extend(unavailable_rows(details, "fortml", "cpu", "--skip-fortml requested"))
    else:
        rows.extend(run_fortml(fortml, args.target, details, expected))
    rows.extend(unavailable_rows(details, "fortml", "cuda", "typed CUDA refusal: resident Adam trajectory unavailable"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
