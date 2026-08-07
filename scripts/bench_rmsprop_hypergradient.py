#!/usr/bin/env python3
"""Correctness-gated fixed-trajectory RMSprop hypergradient benchmark.

The NumPy implementation is deliberately independent of FortML: it evaluates
the scalar validation objective and obtains every packed derivative by central
finite differences.  A FortML release app is retained only after its value,
five gradient components, and directional JVP agree with that oracle.
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
N_PARAMETERS = 5
H = 2.0e-6
REPETITIONS = 16
PARAMETERS = np.array([np.log(0.12), np.log(0.07), 0.78, np.log(0.03), 0.21])
DIRECTION = np.array([0.31, -0.27, 0.17, -0.13, 0.19])
FIELDS = (
    "workload", "phase", "variant", "backend", "device", "status", "n_train",
    "n_validation", "n_parameters", "steps", "repetitions", "seconds_per_operation",
    "metric", "value", "max_abs_error", "oracle", "python_version", "numpy_version",
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


def metadata(root: Path, fortml: Path) -> dict[str, str]:
    ignored = (root / "results" / "rmsprop_hypergradient.csv",)
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.array([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
    train_target = 0.7 * train_x - 0.2
    validation_x = np.array([[-1.5], [0.5], [1.75]])
    validation_target = 0.7 * validation_x - 0.2
    return train_x, train_target, validation_x, validation_target


def loss_gradient(theta: np.ndarray, x: np.ndarray, target: np.ndarray, l2: float) -> tuple[float, np.ndarray]:
    prediction = x[:, 0] * theta[0] + theta[1]
    residual = prediction - target[:, 0]
    value = 0.5 * float(np.mean(residual**2)) + 0.5 * l2 * float(np.dot(theta, theta))
    gradient = np.array([
        float(np.mean(residual * x[:, 0])) + l2 * theta[0],
        float(np.mean(residual)) + l2 * theta[1],
    ])
    return value, gradient


def trajectory(parameters: np.ndarray, centered: bool) -> float:
    train_x, train_target, validation_x, validation_target = fixture()
    learning_rate, l2 = np.exp(parameters[:2])
    decay, epsilon, momentum = parameters[2], np.exp(parameters[3]), parameters[4]
    theta = np.array([0.15, -0.1])
    square = np.zeros(2)
    gradient_average = np.zeros(2)
    momentum_buffer = np.zeros(2)
    for _ in range(STEPS):
        _, gradient = loss_gradient(theta, train_x, train_target, l2)
        square = decay * square + (1.0 - decay) * gradient**2
        if centered:
            gradient_average = decay * gradient_average + (1.0 - decay) * gradient
            variance = np.maximum(square - gradient_average**2, 0.0)
        else:
            variance = square
        update = gradient / (np.sqrt(variance) + epsilon)
        momentum_buffer = momentum * momentum_buffer + update
        theta = theta - learning_rate * momentum_buffer
    residual = validation_x[:, 0] * theta[0] + theta[1] - validation_target[:, 0]
    return 0.5 * float(np.mean(residual**2))


def finite_difference_oracle(centered: bool) -> dict[str, Any]:
    value = trajectory(PARAMETERS, centered)
    gradient = np.empty(5)
    started = time.perf_counter()
    for index in range(5):
        plus = PARAMETERS.copy()
        minus = PARAMETERS.copy()
        plus[index] += H
        minus[index] -= H
        gradient[index] = (trajectory(plus, centered) - trajectory(minus, centered)) / (2.0 * H)
    tangent = (trajectory(PARAMETERS + H * DIRECTION, centered)
        - trajectory(PARAMETERS - H * DIRECTION, centered)) / (2.0 * H)
    elapsed = (time.perf_counter() - started) / REPETITIONS
    return {"value": value, "gradient": gradient, "tangent": tangent,
            "seconds": elapsed}


def base(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    row.update(details)
    row.update({"device": "cpu", "n_train": N_TRAIN, "n_validation": N_VALIDATION,
                "n_parameters": N_PARAMETERS, "steps": STEPS, "repetitions": REPETITIONS})
    row.update(values)
    return row


def numpy_rows(details: dict[str, str], expected: dict[str, Any], variant: str) -> list[dict[str, Any]]:
    rows = [base(details, workload="rmsprop_hypergradient", phase="value_gradient",
                 variant=variant, backend="numpy_oracle", status="pass",
                 seconds_per_operation=expected["seconds"], metric="validation_mse",
                 value=expected["value"], max_abs_error=0.0,
                 oracle="independent NumPy RMSprop trajectory with central FD products",
                 notes="packed=[log_lr,log_l2,decay,log_epsilon,momentum]; h=2e-6")]
    for index, value in enumerate(expected["gradient"], start=1):
        rows.append(base(details, workload="rmsprop_hypergradient", phase="gradient_component",
                         variant=variant, backend="numpy_oracle", status="pass",
                         metric=f"gradient_{index}", value=float(value), max_abs_error=0.0,
                         oracle="independent central finite-difference outer objective",
                         notes="all five packed components checked"))
    rows.append(base(details, workload="rmsprop_hypergradient", phase="jvp",
                     variant=variant, backend="numpy_oracle", status="pass",
                     metric="directional_validation_mse_derivative", value=expected["tangent"],
                     max_abs_error=0.0,
                     oracle="independent central finite-difference directional product",
                     notes=f"direction={DIRECTION.tolist()}; h={H:g}"))
    return rows


def unavailable_rows(details: dict[str, str], device: str, status: str, notes: str,
                     variant: str = "centered_fixed") -> list[dict[str, Any]]:
    return [base(details, workload="rmsprop_hypergradient", phase=phase,
                 variant=variant, backend="fortml", device=device, status=status,
                 oracle="FortML release-app protocol", notes=notes)
            for phase in ("value_gradient", "jvp")]


def run_fortml(fortml: Path, target: str, details: dict[str, str],
               expected: dict[str, Any]) -> list[dict[str, Any]]:
    source = fortml / "app" / f"{target}.f90"
    if not source.is_file():
        return unavailable_rows(details, "cpu", "unavailable", f"release source absent: {source.name}")
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
                           capture_output=True, text=True)
    if build.returncode != 0:
        return unavailable_rows(details, "cpu", "unavailable", "fo build failed")
    with tempfile.TemporaryDirectory(prefix="fortml-rmsprop-hypergradient-") as temporary:
        oracle_path = Path(temporary) / "oracle.csv"
        environment["FORTML_BENCH_RMSPROP_HYPERGRADIENT_ORACLE"] = str(oracle_path)
        run = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml, env=environment,
                             capture_output=True, text=True)
        if run.returncode != 0 or not oracle_path.is_file():
            return unavailable_rows(details, "cpu", "unavailable", "release target execution failed")
        values: dict[tuple[str, int], float] = {}
        with oracle_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                values[(row["quantity"], int(row["index"]))] = float(row["value"])
        actual_value = values[("value", 1)]
        actual_gradient = np.array([values[("gradient", index)] for index in range(1, 6)])
        actual_tangent = values[("jvp", 1)]
        error = max(abs(actual_value - expected["value"]),
                    float(np.max(np.abs(actual_gradient - expected["gradient"]))),
                    abs(actual_tangent - expected["tangent"]))
        if error > 3.0e-10:
            raise RuntimeError(f"FortML RMSprop hypergradient oracle mismatch: {error:.3e}")
        timing = next((float(line.split(",", 1)[1]) for line in run.stdout.splitlines()
                       if line.startswith("rmsprop_hypergradient_value_gradient,")), None)
        if timing is None:
            return unavailable_rows(details, "cpu", "unavailable", "release app emitted no timing row")
        rows = [base(details, workload="rmsprop_hypergradient", phase="value_gradient",
                     variant="centered_fixed", backend="fortml", status="pass",
                     seconds_per_operation=timing, metric="validation_mse", value=actual_value,
                     max_abs_error=error,
                     oracle="independent NumPy trajectory and central-FD products",
                     notes=f"{target}; centered branch; five gradients and JVP checked")]
        rows.extend(base(details, workload="rmsprop_hypergradient", phase="gradient_component",
                         variant="centered_fixed", backend="fortml", status="pass",
                         metric=f"gradient_{index}", value=float(actual_gradient[index - 1]),
                         max_abs_error=abs(actual_gradient[index - 1] - expected["gradient"][index - 1]),
                         oracle="independent NumPy trajectory and central-FD products",
                         notes=target) for index in range(1, 6))
        rows.append(base(details, workload="rmsprop_hypergradient", phase="jvp",
                         variant="centered_fixed", backend="fortml", status="pass",
                         metric="directional_validation_mse_derivative", value=actual_tangent,
                         max_abs_error=abs(actual_tangent - expected["tangent"]),
                         oracle="independent NumPy trajectory and central-FD products",
                         notes=target))
        return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/rmsprop_hypergradient.csv"))
    parser.add_argument("--target", default="fortml_bench_rmsprop_hypergradient")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    details = metadata(root, fortml)
    centered = finite_difference_oracle(centered=True)
    uncentered = finite_difference_oracle(centered=False)
    if not np.all(np.isfinite(uncentered["gradient"])):
        raise RuntimeError("uncentered RMSprop oracle is not finite")
    rows = numpy_rows(details, centered, "centered_fixed")
    rows.extend(numpy_rows(details, uncentered, "uncentered_fixed"))
    if args.skip_fortml:
        rows.extend(unavailable_rows(details, "cpu", "skipped", "--skip-fortml"))
    else:
        rows.extend(run_fortml(fortml, args.target, details, centered))
    rows.extend(unavailable_rows(
        details, "cpu", "unavailable",
        "release app currently exports the centered fixture only", "uncentered_fixed"))
    for variant in ("centered_fixed", "uncentered_fixed"):
        rows.extend(unavailable_rows(
            details, "cuda", "unavailable",
            "RMSprop hypergradient release app is CPU-only until resident CUDA state exists",
            variant))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
