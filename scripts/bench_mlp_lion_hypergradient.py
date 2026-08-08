#!/usr/bin/env python3
"""Correctness-gated fixed-trajectory Lion hypergradient benchmark.

The NumPy trajectory is an independent two-parameter linear-MSE Lion oracle.
It finite-differences all four packed log/logit coordinates away from sign
boundaries before a FortML timing is retained. CUDA remains a typed refusal
because the complete model and optimizer state are not resident yet.
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
N_TRAIN = 6
N_VALIDATION = 3
N_PARAMETERS = 4
LEARNING_RATE = 2.0e-4
L2 = 0.04
BETA1 = 0.9
BETA2 = 0.99
FD_STEP = 2.0e-6
REPETITIONS = 16
PARAMETERS = np.array([np.log(LEARNING_RATE), np.log(L2),
                       np.log(BETA1 / (1.0 - BETA1)),
                       np.log(BETA2 / (1.0 - BETA2))])
DIRECTION = np.array([0.29, -0.23, 0.17, -0.13])
ORACLE_TOLERANCE = 4.0e-10

FIELDS = (
    "workload", "phase", "variant", "backend", "device", "status",
    "n_train", "n_validation", "n_parameters", "steps", "repetitions",
    "seconds_per_operation", "metric", "value", "max_abs_error", "oracle",
    "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
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
    train_x = np.array([[-2.0], [-1.0], [-0.3], [0.4], [1.2], [2.1]], dtype=np.float64)
    train_target = 0.8 * train_x - 0.15
    validation_x = np.array([[-1.7], [0.25], [1.8]], dtype=np.float64)
    validation_target = 0.8 * validation_x - 0.15
    return train_x, train_target, validation_x, validation_target


def loss_gradient(theta: np.ndarray, x: np.ndarray, target: np.ndarray,
                  l2: float) -> tuple[float, np.ndarray]:
    residual = x[:, 0] * theta[0] + theta[1] - target[:, 0]
    value = 0.5 * float(np.mean(residual * residual)) + 0.5 * l2 * float(np.dot(theta, theta))
    gradient = np.array([
        float(np.mean(residual * x[:, 0])) + l2 * theta[0],
        float(np.mean(residual)) + l2 * theta[1],
    ])
    return value, gradient


def trajectory(parameters: np.ndarray) -> float:
    train_x, train_target, validation_x, validation_target = fixture()
    learning_rate, l2 = np.exp(parameters[:2])
    beta1, beta2 = 1.0 / (1.0 + np.exp(-parameters[2])), 1.0 / (1.0 + np.exp(-parameters[3]))
    theta = np.array([0.17, -0.08], dtype=np.float64)
    momentum = np.zeros(2, dtype=np.float64)
    for _ in range(STEPS):
        _, gradient = loss_gradient(theta, train_x, train_target, l2)
        candidate = beta1 * momentum + (1.0 - beta1) * gradient
        if np.any(np.abs(candidate) <= 1.0e-14):
            raise RuntimeError("NumPy Lion oracle reached a sign boundary")
        theta = theta - learning_rate * np.sign(candidate)
        momentum = beta2 * momentum + (1.0 - beta2) * gradient
    residual = validation_x[:, 0] * theta[0] + theta[1] - validation_target[:, 0]
    return 0.5 * float(np.mean(residual * residual))


def finite_difference_oracle() -> dict[str, Any]:
    value = trajectory(PARAMETERS)
    gradient = np.empty(N_PARAMETERS, dtype=np.float64)
    started = time.perf_counter()
    for index in range(N_PARAMETERS):
        plus, minus = PARAMETERS.copy(), PARAMETERS.copy()
        plus[index] += FD_STEP
        minus[index] -= FD_STEP
        gradient[index] = (trajectory(plus) - trajectory(minus)) / (2.0 * FD_STEP)
    tangent = (trajectory(PARAMETERS + FD_STEP * DIRECTION)
               - trajectory(PARAMETERS - FD_STEP * DIRECTION)) / (2.0 * FD_STEP)
    elapsed = (time.perf_counter() - started) / REPETITIONS
    if not np.all(np.isfinite(gradient)) or not np.isfinite(value) or not np.isfinite(tangent):
        raise RuntimeError("NumPy Lion oracle is nonfinite")
    return {"value": value, "gradient": gradient, "tangent": tangent, "seconds": elapsed}


def base(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    row.update(details)
    row.update({"device": "cpu", "n_train": N_TRAIN, "n_validation": N_VALIDATION,
                "n_parameters": N_PARAMETERS, "steps": STEPS, "repetitions": REPETITIONS})
    row.update(values)
    return row


def unavailable_rows(details: dict[str, str], device: str, status: str,
                     notes: str) -> list[dict[str, Any]]:
    return [base(details, workload="mlp_lion_hypergradient", phase=phase,
                 variant="fixed_full_batch", backend="fortml", device=device,
                 status=status, metric=metric, oracle="FortML release-app protocol",
                 notes=notes)
            for phase, metric in (("value_gradient", "validation_mse"),
                                  ("gradient_component", "gradient_1"),
                                  ("gradient_component", "gradient_2"),
                                  ("gradient_component", "gradient_3"),
                                  ("gradient_component", "gradient_4"),
                                  ("jvp", "directional_validation_mse_derivative"))]


def read_oracle(path: Path) -> dict[tuple[str, int], float]:
    with path.open(newline="") as stream:
        return {(row["quantity"], int(row["index"])): float(row["value"])
                for row in csv.DictReader(stream)}


def run_fortml(fortml: Path, target: str, details: dict[str, str],
               expected: dict[str, Any]) -> list[dict[str, Any]]:
    if not (fortml / "app" / f"{target}.f90").is_file():
        return unavailable_rows(details, "cpu", "unavailable", "release source absent")
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                           env=environment, capture_output=True, text=True)
    if build.returncode != 0:
        return unavailable_rows(details, "cpu", "unavailable", "fo build failed")
    with tempfile.TemporaryDirectory(dir="/mnt/storage", prefix="fortml-lion-hypergradient-") as temporary:
        oracle_path = Path(temporary) / "oracle.csv"
        check_environment = dict(environment)
        check_environment.update({"FORTML_BENCH_LION_HYPERGRADIENT_ORACLE": str(oracle_path),
                                   "FORTML_BENCH_ORACLE_ONLY": "1"})
        check = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                               env=check_environment, capture_output=True, text=True)
        if check.returncode != 0 or not oracle_path.is_file():
            return unavailable_rows(details, "cpu", "unavailable", "release target emitted no oracle")
        actual = read_oracle(oracle_path)
        required = {("value", 1), ("jvp", 1)} | {("gradient", i) for i in range(1, N_PARAMETERS + 1)}
        if set(actual) != required:
            raise RuntimeError("FortML Lion app omitted a complete value/gradient/JVP array")
        errors = [abs(actual[("value", 1)] - expected["value"]),
                  abs(actual[("jvp", 1)] - expected["tangent"])]
        errors.extend(abs(actual[("gradient", i)] - expected["gradient"][i - 1])
                      for i in range(1, N_PARAMETERS + 1))
        error = float(max(errors))
        if error > ORACLE_TOLERANCE:
            raise RuntimeError(f"FortML Lion oracle mismatch: {error:.3e}")
        timed = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                               env=environment, capture_output=True, text=True)
    if timed.returncode != 0:
        return unavailable_rows(details, "cpu", "unavailable", "timing execution failed")
    timing = next((float(line.split(",", 1)[1].strip())
                   for line in timed.stdout.splitlines()
                   if line.startswith("mlp_lion_hypergradient_value_gradient,")), None)
    if timing is None:
        raise RuntimeError("FortML Lion app emitted no value-gradient timing")
    rows = [base(details, workload="mlp_lion_hypergradient", phase="value_gradient",
                 variant="fixed_full_batch", backend="fortml", status="pass",
                 seconds_per_operation=timing, metric="validation_mse",
                 value=actual[("value", 1)], max_abs_error=error,
                 oracle="independent NumPy Lion trajectory and central-FD products",
                 notes=f"{target}; four gradients and JVP checked")]
    rows.extend(base(details, workload="mlp_lion_hypergradient", phase="gradient_component",
                     variant="fixed_full_batch", backend="fortml", status="pass",
                     metric=f"gradient_{i}", value=actual[("gradient", i)],
                     max_abs_error=abs(actual[("gradient", i)] - expected["gradient"][i - 1]),
                     oracle="independent NumPy Lion trajectory and central-FD products",
                     notes=target) for i in range(1, N_PARAMETERS + 1))
    rows.append(base(details, workload="mlp_lion_hypergradient", phase="jvp",
                     variant="fixed_full_batch", backend="fortml", status="pass",
                     metric="directional_validation_mse_derivative", value=actual[("jvp", 1)],
                     max_abs_error=abs(actual[("jvp", 1)] - expected["tangent"]),
                     oracle="independent NumPy Lion trajectory and central-FD products",
                     notes=target))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/mlp_lion_hypergradient.csv"))
    parser.add_argument("--target", default="fortml_bench_mlp_lion_hypergradient")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml, output = args.fortml.resolve(), args.output.resolve()
    details, expected = metadata(root, fortml, output), finite_difference_oracle()
    rows = [base(details, workload="mlp_lion_hypergradient", phase="value_gradient",
                 variant="fixed_full_batch", backend="numpy_oracle", status="pass",
                 seconds_per_operation=expected["seconds"], metric="validation_mse",
                 value=expected["value"], max_abs_error=0.0,
                 oracle="independent NumPy Lion trajectory with central-FD products",
                 notes="packed=[log_lr,log_l2,logit_beta1,logit_beta2]; sign branch fixed")]
    rows.extend(base(details, workload="mlp_lion_hypergradient", phase="gradient_component",
                     variant="fixed_full_batch", backend="numpy_oracle", status="pass",
                     metric=f"gradient_{i}", value=float(value), max_abs_error=0.0,
                     oracle="independent central finite-difference outer objective",
                     notes="all four packed components checked away from sign boundaries")
                for i, value in enumerate(expected["gradient"], start=1))
    rows.append(base(details, workload="mlp_lion_hypergradient", phase="jvp",
                     variant="fixed_full_batch", backend="numpy_oracle", status="pass",
                     metric="directional_validation_mse_derivative", value=expected["tangent"],
                     max_abs_error=0.0, oracle="independent central-FD directional product",
                     notes=f"direction={DIRECTION.tolist()}; h={FD_STEP:g}"))
    if args.skip_fortml:
        rows.extend(unavailable_rows(details, "cpu", "skipped", "--skip-fortml"))
    else:
        rows.extend(run_fortml(fortml, args.target, details, expected))
    rows.extend(unavailable_rows(details, "cuda", "unavailable",
                                 "complete Lion model/state derivative is CPU-only until resident kernels exist"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
