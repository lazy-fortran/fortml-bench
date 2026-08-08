#!/usr/bin/env python3
"""Correctness-gated weighted-validation SGD trajectory benchmark.

The NumPy fixture is independent of FortML and finite-differences the
weighted held-out objective.  The release app must agree on value, all packed
hypergradient components, the directional JVP, and the certified uniform
outer HVP.  Non-uniform HVP and CUDA rows are retained as typed boundaries.
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
N_TRAIN = 5
N_VALIDATION = 3
N_PARAMETERS = 3
LEARNING_RATE = 0.11
L2 = 0.06
MOMENTUM = 0.29
FD_STEP = 2.0e-6
HVP_STEP = 2.0e-4
REPETITIONS = 16
PARAMETERS = np.array([np.log(LEARNING_RATE), np.log(L2), MOMENTUM])
DIRECTION = np.array([0.23, -0.17, 0.11])
WEIGHTS = np.array([1.0, 2.0, 4.0])
ORACLE_TOLERANCE = 5.0e-7

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
    train_target = 0.65 * train_x - 0.1
    validation_x = np.array([[-1.5], [0.5], [1.75]], dtype=np.float64)
    validation_target = 0.65 * validation_x - 0.1
    return train_x, train_target, validation_x, validation_target


def loss_gradient(theta: np.ndarray, x: np.ndarray, target: np.ndarray,
                  l2: float) -> tuple[float, np.ndarray]:
    prediction = x[:, 0] * theta[0] + theta[1]
    residual = prediction - target[:, 0]
    value = 0.5 * float(np.mean(residual * residual)) + 0.5 * l2 * float(np.dot(theta, theta))
    gradient = np.array([
        float(np.mean(residual * x[:, 0])) + l2 * theta[0],
        float(np.mean(residual)) + l2 * theta[1],
    ])
    return value, gradient


def trajectory(parameters: np.ndarray, weights: np.ndarray | None = None) -> float:
    train_x, train_target, validation_x, validation_target = fixture()
    learning_rate, l2 = np.exp(parameters[:2])
    momentum = parameters[2]
    theta = np.array([0.13, -0.08], dtype=np.float64)
    velocity = np.zeros(2, dtype=np.float64)
    for _ in range(STEPS):
        _, gradient = loss_gradient(theta, train_x, train_target, l2)
        velocity = momentum * velocity + gradient
        theta = theta - learning_rate * velocity
    residual = validation_x[:, 0] * theta[0] + theta[1] - validation_target[:, 0]
    if weights is None:
        return 0.5 * float(np.mean(residual * residual))
    return 0.5 * float(np.dot(weights, residual * residual) / np.sum(weights))


def finite_difference_gradient(parameters: np.ndarray,
                               weights: np.ndarray | None = None) -> np.ndarray:
    gradient = np.empty(N_PARAMETERS, dtype=np.float64)
    for index in range(N_PARAMETERS):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += FD_STEP
        minus[index] -= FD_STEP
        gradient[index] = (trajectory(plus, weights) - trajectory(minus, weights)) / (2.0 * FD_STEP)
    return gradient


def finite_difference_oracle() -> dict[str, Any]:
    weighted_value = trajectory(PARAMETERS, WEIGHTS)
    weighted_gradient = finite_difference_gradient(PARAMETERS, WEIGHTS)
    weighted_tangent = (trajectory(PARAMETERS + FD_STEP * DIRECTION, WEIGHTS)
                        - trajectory(PARAMETERS - FD_STEP * DIRECTION, WEIGHTS)) / (2.0 * FD_STEP)
    uniform_gradient = finite_difference_gradient(PARAMETERS)
    uniform_hvp = (finite_difference_gradient(PARAMETERS + HVP_STEP * DIRECTION)
                   - finite_difference_gradient(PARAMETERS - HVP_STEP * DIRECTION)) / (2.0 * HVP_STEP)
    started = time.perf_counter()
    for _ in range(REPETITIONS):
        trajectory(PARAMETERS, WEIGHTS)
    elapsed = (time.perf_counter() - started) / REPETITIONS
    if not np.all(np.isfinite(weighted_gradient)) or not np.isfinite(weighted_value):
        raise RuntimeError("weighted validation NumPy oracle is nonfinite")
    return {
        "weighted_value": weighted_value, "weighted_gradient": weighted_gradient,
        "weighted_tangent": weighted_tangent, "uniform_gradient": uniform_gradient,
        "uniform_hvp": uniform_hvp, "seconds": elapsed,
    }


def base(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    row.update(details)
    row.update({
        "device": "cpu", "n_train": N_TRAIN, "n_validation": N_VALIDATION,
        "n_parameters": N_PARAMETERS, "steps": STEPS, "repetitions": REPETITIONS,
    })
    row.update(values)
    return row


def oracle_rows(details: dict[str, str], expected: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [base(
        details, workload="mlp_weighted_validation_hypergradient", phase="value_gradient",
        variant="nonuniform_validation", backend="numpy_oracle", status="pass",
        seconds_per_operation=expected["seconds"], metric="weighted_validation_mse",
        value=expected["weighted_value"], max_abs_error=0.0,
        oracle="independent NumPy fixed SGD trajectory with weighted central-FD products",
        notes="weights=[1,2,4]; packed=[log_lr,log_l2,momentum]",
    )]
    for index, value in enumerate(expected["weighted_gradient"], start=1):
        rows.append(base(
            details, workload="mlp_weighted_validation_hypergradient", phase="gradient_component",
            variant="nonuniform_validation", backend="numpy_oracle", status="pass",
            metric=f"weighted_gradient_{index}", value=float(value), max_abs_error=0.0,
            oracle="independent central finite-difference weighted validation objective",
            notes="positive-support validation measure",
        ))
    rows.append(base(
        details, workload="mlp_weighted_validation_hypergradient", phase="jvp",
        variant="nonuniform_validation", backend="numpy_oracle", status="pass",
        metric="weighted_validation_jvp", value=expected["weighted_tangent"], max_abs_error=0.0,
        oracle="independent central finite-difference directional product",
        notes=f"direction={DIRECTION.tolist()}; h={FD_STEP:g}",
    ))
    for index, value in enumerate(expected["uniform_hvp"], start=1):
        rows.append(base(
            details, workload="mlp_weighted_validation_hypergradient", phase="hvp_component",
            variant="uniform_validation", backend="numpy_oracle", status="pass",
            metric=f"uniform_hvp_{index}", value=float(value), max_abs_error=0.0,
            oracle="independent central finite-difference of uniform validation gradient",
            notes=f"direction={DIRECTION.tolist()}; outer h={HVP_STEP:g}",
        ))
    return rows


def unavailable_rows(details: dict[str, str], phase: str, variant: str,
                     device: str, status: str, metric: str, notes: str) -> dict[str, Any]:
    return base(details, workload="mlp_weighted_validation_hypergradient", phase=phase,
                variant=variant, backend="fortml", device=device, status=status,
                metric=metric, oracle="FortML typed capability contract", notes=notes)


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
        return [unavailable_rows(details, "value_gradient", "nonuniform_validation", "cpu",
                                 "unavailable", "weighted_validation_mse", f"missing {source.name}")]
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                           env=environment, capture_output=True, text=True)
    if build.returncode != 0:
        return [unavailable_rows(details, "value_gradient", "nonuniform_validation", "cpu",
                                 "unavailable", "weighted_validation_mse", "fo build failed")]
    with tempfile.TemporaryDirectory(dir="/mnt/storage", prefix="fortml-weighted-validation-") as temporary:
        oracle_path = Path(temporary) / "oracle.csv"
        check_environment = dict(environment)
        check_environment.update({"FORTML_BENCH_MLP_WEIGHTED_VALIDATION_ORACLE": str(oracle_path),
                                  "FORTML_BENCH_ORACLE_ONLY": "1"})
        check = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                               env=check_environment, capture_output=True, text=True)
        if check.returncode != 0 or not oracle_path.is_file():
            return [unavailable_rows(details, "value_gradient", "nonuniform_validation", "cpu",
                                     "unavailable", "weighted_validation_mse",
                                     "release target did not emit complete oracle")]
        actual = read_oracle(oracle_path)
        required = {("weighted_value", 1), ("weighted_jvp", 1), ("nonuniform_hvp_status", 1)}
        required |= {("weighted_gradient", i) for i in range(1, 4)}
        required |= {("uniform_hvp", i) for i in range(1, 4)}
        if set(actual) != required:
            raise RuntimeError("FortML weighted validation app emitted an incomplete oracle")
        errors = [abs(actual[("weighted_value", 1)] - expected["weighted_value"]),
                  abs(actual[("weighted_jvp", 1)] - expected["weighted_tangent"])]
        errors.extend(abs(actual[("weighted_gradient", i)] - expected["weighted_gradient"][i - 1])
                      for i in range(1, 4))
        errors.extend(abs(actual[("uniform_hvp", i)] - expected["uniform_hvp"][i - 1])
                      for i in range(1, 4))
        error = float(max(errors))
        if error > ORACLE_TOLERANCE:
            raise RuntimeError(f"FortML weighted validation mismatch: {error:.3e}")
        if int(actual[("nonuniform_hvp_status", 1)]) <= 0:
            raise RuntimeError("FortML weighted validation HVP refusal was not typed")
        timed = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                               env=environment, capture_output=True, text=True)
    timing = next((float(line.split(",", 1)[1].strip()) for line in timed.stdout.splitlines()
                   if line.startswith("mlp_weighted_validation_hypergradient_value_gradient,")), None)
    if timed.returncode != 0 or timing is None:
        return [unavailable_rows(details, "value_gradient", "nonuniform_validation", "cpu",
                                 "unavailable", "weighted_validation_mse", "timing execution failed")]
    rows = [base(details, workload="mlp_weighted_validation_hypergradient", phase="value_gradient",
                 variant="nonuniform_validation", backend="fortml", status="pass",
                 seconds_per_operation=timing, metric="weighted_validation_mse",
                 value=actual[("weighted_value", 1)], max_abs_error=error,
                 oracle="independent NumPy weighted trajectory and central-FD products",
                 notes=target)]
    rows.extend(base(details, workload="mlp_weighted_validation_hypergradient", phase="gradient_component",
                     variant="nonuniform_validation", backend="fortml", status="pass",
                     metric=f"weighted_gradient_{i}", value=actual[("weighted_gradient", i)],
                     max_abs_error=abs(actual[("weighted_gradient", i)] - expected["weighted_gradient"][i - 1]),
                     oracle="independent NumPy weighted trajectory", notes=target) for i in range(1, 4))
    rows.append(base(details, workload="mlp_weighted_validation_hypergradient", phase="jvp",
                     variant="nonuniform_validation", backend="fortml", status="pass",
                     metric="weighted_validation_jvp", value=actual[("weighted_jvp", 1)],
                     max_abs_error=abs(actual[("weighted_jvp", 1)] - expected["weighted_tangent"]),
                     oracle="independent NumPy directional product", notes=target))
    rows.extend(base(details, workload="mlp_weighted_validation_hypergradient", phase="hvp_component",
                     variant="uniform_validation", backend="fortml", status="pass",
                     metric=f"uniform_hvp_{i}", value=actual[("uniform_hvp", i)],
                     max_abs_error=abs(actual[("uniform_hvp", i)] - expected["uniform_hvp"][i - 1]),
                     oracle="independent NumPy uniform HVP oracle", notes=target) for i in range(1, 4))
    rows.append(unavailable_rows(details, "hvp", "nonuniform_validation", "cpu", "refused",
                                 "hvp", "typed FORTNUM_NOT_IMPLEMENTED boundary"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/mlp_weighted_validation_hypergradient.csv"))
    parser.add_argument("--target", default="fortml_bench_mlp_weighted_validation_hypergradient")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    details = metadata(root, fortml, output)
    expected = finite_difference_oracle()
    rows = oracle_rows(details, expected)
    if args.skip_fortml:
        rows.append(unavailable_rows(details, "value_gradient", "nonuniform_validation", "cpu",
                                     "skipped", "weighted_validation_mse", "--skip-fortml"))
    else:
        rows.extend(run_fortml(fortml, args.target, details, expected))
    rows.append(unavailable_rows(details, "hvp", "nonuniform_validation", "cuda", "unavailable",
                                 "hvp", "resident weighted trajectory derivatives are not implemented"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
