#!/usr/bin/env python3
"""Correctness-gated fixed-trajectory SGD momentum hypergradient benchmark.

The NumPy implementation is an independent two-parameter linear-MSE
trajectory.  It finite-differences the validation objective with respect to
the packed ``[log_learning_rate, log_l2, momentum]`` vector and checks the
complete FortML value/gradient/JVP/HVP oracle before retaining a CPU timing.
CUDA rows remain typed refusals until the optimizer state derivatives are
resident.
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
LEARNING_RATE = 0.12
L2 = 0.07
MOMENTUM = 0.31
FD_STEP = 2.0e-6
HVP_STEP = 2.0e-4
REPETITIONS = 16
PARAMETERS = np.array([np.log(LEARNING_RATE), np.log(L2), MOMENTUM])
DIRECTION = np.array([0.31, -0.27, 0.17])
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
    train_target = 0.7 * train_x - 0.2
    validation_x = np.array([[-1.5], [0.5], [1.75]], dtype=np.float64)
    validation_target = 0.7 * validation_x - 0.2
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


def trajectory(parameters: np.ndarray) -> float:
    train_x, train_target, validation_x, validation_target = fixture()
    learning_rate, l2 = np.exp(parameters[:2])
    momentum = parameters[2]
    theta = np.array([0.15, -0.1], dtype=np.float64)
    velocity = np.zeros(2, dtype=np.float64)
    for _ in range(STEPS):
        _, gradient = loss_gradient(theta, train_x, train_target, l2)
        velocity = momentum * velocity + gradient
        theta = theta - learning_rate * velocity
    residual = validation_x[:, 0] * theta[0] + theta[1] - validation_target[:, 0]
    return 0.5 * float(np.mean(residual * residual))


def finite_difference_gradient(parameters: np.ndarray) -> np.ndarray:
    gradient = np.empty(N_PARAMETERS, dtype=np.float64)
    for index in range(N_PARAMETERS):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += FD_STEP
        minus[index] -= FD_STEP
        gradient[index] = (trajectory(plus) - trajectory(minus)) / (2.0 * FD_STEP)
    return gradient


def finite_difference_oracle() -> dict[str, Any]:
    value = trajectory(PARAMETERS)
    started = time.perf_counter()
    gradient = finite_difference_gradient(PARAMETERS)
    tangent = (trajectory(PARAMETERS + FD_STEP * DIRECTION)
               - trajectory(PARAMETERS - FD_STEP * DIRECTION)) / (2.0 * FD_STEP)
    gradient_plus = finite_difference_gradient(PARAMETERS + HVP_STEP * DIRECTION)
    gradient_minus = finite_difference_gradient(PARAMETERS - HVP_STEP * DIRECTION)
    hvp = (gradient_plus - gradient_minus) / (2.0 * HVP_STEP)
    elapsed = (time.perf_counter() - started) / REPETITIONS
    if (not np.all(np.isfinite(gradient)) or not np.isfinite(value) or
            not np.isfinite(tangent) or not np.all(np.isfinite(hvp))):
        raise RuntimeError("SGD momentum hypergradient NumPy oracle is nonfinite")
    return {"value": value, "gradient": gradient, "tangent": tangent,
            "hvp": hvp, "seconds": elapsed}


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
        details, workload="mlp_sgd_momentum_hypergradient", phase="value_gradient",
        variant="fixed_full_batch", backend="numpy_oracle", status="pass",
        seconds_per_operation=expected["seconds"], metric="validation_mse",
        value=expected["value"], max_abs_error=0.0,
        oracle="independent NumPy SGD momentum trajectory with central-FD products",
        notes="packed=[log_lr,log_l2,momentum]; v <- momentum*v + gradient; h=2e-6",
    )]
    for index, value in enumerate(expected["gradient"], start=1):
        rows.append(base(
            details, workload="mlp_sgd_momentum_hypergradient", phase="gradient_component",
            variant="fixed_full_batch", backend="numpy_oracle", status="pass",
            metric=f"gradient_{index}", value=float(value), max_abs_error=0.0,
            oracle="independent central finite-difference outer objective",
            notes="all three packed components checked",
        ))
    rows.append(base(
        details, workload="mlp_sgd_momentum_hypergradient", phase="jvp",
        variant="fixed_full_batch", backend="numpy_oracle", status="pass",
        metric="directional_validation_mse_derivative", value=expected["tangent"],
        max_abs_error=0.0,
        oracle="independent central finite-difference directional product",
        notes=f"direction={DIRECTION.tolist()}; h={FD_STEP:g}",
    ))
    for index, value in enumerate(expected["hvp"], start=1):
        rows.append(base(
            details, workload="mlp_sgd_momentum_hypergradient", phase="hvp_component",
            variant="affine_one_layer", backend="numpy_oracle", status="pass",
            metric=f"hvp_{index}", value=float(value), max_abs_error=0.0,
            oracle="independent central finite-difference of trajectory gradient",
            notes=f"direction={DIRECTION.tolist()}; nested h={FD_STEP:g}",
        ))
    return rows


def unavailable_rows(details: dict[str, str], device: str, status: str,
                     notes: str) -> list[dict[str, Any]]:
    rows = []
    for phase, metric in (
        ("value_gradient", "validation_mse"),
        ("gradient_component", "gradient_1"),
        ("gradient_component", "gradient_2"),
        ("gradient_component", "gradient_3"),
        ("jvp", "directional_validation_mse_derivative"),
        ("hvp_component", "hvp_1"),
        ("hvp_component", "hvp_2"),
        ("hvp_component", "hvp_3"),
    ):
        rows.append(base(
            details, workload="mlp_sgd_momentum_hypergradient", phase=phase,
            variant="fixed_full_batch", backend="fortml", device=device,
            status=status, metric=metric, oracle="FortML release-app protocol",
            notes=notes,
        ))
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
        return unavailable_rows(details, "cpu", "unavailable",
                                f"release source absent: {source.name}")
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"),
                         "OMP_NUM_THREADS": "1"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                           env=environment, capture_output=True, text=True)
    if build.returncode != 0:
        return unavailable_rows(details, "cpu", "unavailable", "fo build failed")
    with tempfile.TemporaryDirectory(dir="/mnt/storage",
                                      prefix="fortml-sgd-momentum-hypergradient-") as temporary:
        oracle_path = Path(temporary) / "oracle.csv"
        check_environment = dict(environment)
        check_environment.update({
            "FORTML_BENCH_SGD_MOMENTUM_HYPERGRADIENT_ORACLE": str(oracle_path),
            "FORTML_BENCH_ORACLE_ONLY": "1",
        })
        check = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                               env=check_environment, capture_output=True, text=True)
        if check.returncode != 0 or not oracle_path.is_file():
            return unavailable_rows(details, "cpu", "unavailable",
                                    "release target did not emit its complete oracle")
        actual = read_oracle(oracle_path)
        required = {("value", 1), ("jvp", 1)} | {
            ("gradient", index) for index in range(1, N_PARAMETERS + 1)
        } | {("hvp", index) for index in range(1, N_PARAMETERS + 1)}
        if set(actual) != required:
            raise RuntimeError("FortML SGD momentum app omitted a complete value/gradient/JVP array")
        errors = [abs(actual[("value", 1)] - expected["value"]),
                  abs(actual[("jvp", 1)] - expected["tangent"])]
        errors.extend(abs(actual[("gradient", index)] - expected["gradient"][index - 1])
                      for index in range(1, N_PARAMETERS + 1))
        errors.extend(abs(actual[("hvp", index)] - expected["hvp"][index - 1])
                      for index in range(1, N_PARAMETERS + 1))
        error = float(max(errors))
        if error > ORACLE_TOLERANCE:
            raise RuntimeError(f"FortML SGD momentum hypergradient oracle mismatch: {error:.3e}")
        timed = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                               env=environment, capture_output=True, text=True)
    if timed.returncode != 0:
        return unavailable_rows(details, "cpu", "unavailable",
                                "release target timing execution failed")
    timing = next((float(line.split(",", 1)[1].strip())
                   for line in timed.stdout.splitlines()
                   if line.startswith("sgd_momentum_hypergradient_value_gradient,")), None)
    if timing is None:
        raise RuntimeError("FortML SGD momentum app emitted no value-gradient timing")
    rows = [base(
        details, workload="mlp_sgd_momentum_hypergradient", phase="value_gradient",
        variant="fixed_full_batch", backend="fortml", status="pass",
        seconds_per_operation=timing, metric="validation_mse",
        value=actual[("value", 1)], max_abs_error=error,
        oracle="independent NumPy trajectory and central-FD products",
        notes=f"{target}; three gradients and JVP checked",
    )]
    rows.extend(base(
        details, workload="mlp_sgd_momentum_hypergradient", phase="gradient_component",
        variant="fixed_full_batch", backend="fortml", status="pass",
        metric=f"gradient_{index}", value=actual[("gradient", index)],
        max_abs_error=abs(actual[("gradient", index)] - expected["gradient"][index - 1]),
        oracle="independent NumPy trajectory and central-FD products", notes=target,
    ) for index in range(1, N_PARAMETERS + 1))
    rows.append(base(
        details, workload="mlp_sgd_momentum_hypergradient", phase="jvp",
        variant="fixed_full_batch", backend="fortml", status="pass",
        metric="directional_validation_mse_derivative", value=actual[("jvp", 1)],
        max_abs_error=abs(actual[("jvp", 1)] - expected["tangent"]),
        oracle="independent NumPy trajectory and central-FD products", notes=target,
    ))
    rows.extend(base(
        details, workload="mlp_sgd_momentum_hypergradient", phase="hvp_component",
        variant="affine_one_layer", backend="fortml", status="pass",
        metric=f"hvp_{index}", value=actual[("hvp", index)],
        max_abs_error=abs(actual[("hvp", index)] - expected["hvp"][index - 1]),
        oracle="independent NumPy nested central-FD trajectory oracle", notes=target,
    ) for index in range(1, N_PARAMETERS + 1))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/sgd_momentum_hypergradient.csv"))
    parser.add_argument("--target", default="fortml_bench_sgd_momentum_hypergradient")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    details = metadata(root, fortml, output)
    expected = finite_difference_oracle()
    rows = oracle_rows(details, expected)
    if args.skip_fortml:
        rows.extend(unavailable_rows(details, "cpu", "skipped", "--skip-fortml"))
    else:
        rows.extend(run_fortml(fortml, args.target, details, expected))
    rows.extend(unavailable_rows(
        details, "cuda", "unavailable",
        "complete SGD momentum MLP hypergradient is CPU-only until resident state derivatives exist",
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
