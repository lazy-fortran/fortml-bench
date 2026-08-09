#!/usr/bin/env python3
"""Independent oracle for scheduled optimizer-group products."""

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

STEPS, TOTAL_UPDATES = 4, 6
N_TRAIN, N_VALIDATION, N_PARAMETERS = 6, 3, 6
LEARNING_RATE, L2, MIN_FRACTION, DECAY_FACTOR = 0.07, 0.03, 0.2, 0.5
GROUPS = np.array([0.65, 1.25], dtype=np.float64)
FD_STEP, REPETITIONS = 2.0e-6, 16
PARAMETERS = np.array([np.log(LEARNING_RATE), np.log(L2),
                       np.log(MIN_FRACTION/(1-MIN_FRACTION)),
                       np.log(DECAY_FACTOR/(1-DECAY_FACTOR)), *np.log(GROUPS)])
DIRECTION = np.array([0.11, -0.07, 0.09, -0.05, 0.13, -0.17])
FIELDS = ("workload", "phase", "variant", "backend", "device", "status",
          "n_train", "n_validation", "n_parameters", "steps", "repetitions",
          "seconds_per_operation", "metric", "value", "max_abs_error", "oracle",
          "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
          "compiler", "flags", "notes")


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.array([[-1.5], [-0.8], [-0.1], [0.6], [1.4], [2.0]], dtype=np.float64)
    validation_x = np.array([[-1.2], [0.25], [1.7]], dtype=np.float64)
    return train_x, 0.75*train_x-0.2, validation_x, 0.75*validation_x-0.2


def loss_gradient(theta: np.ndarray, x: np.ndarray, target: np.ndarray, l2: float) -> np.ndarray:
    residual = x[:, 0]*theta[0] + theta[1] - target[:, 0]
    return np.array([float(np.mean(residual*x[:, 0])) + l2*theta[0],
                     float(np.mean(residual)) + l2*theta[1]])


def trajectory(parameters: np.ndarray) -> float:
    train_x, train_target, validation_x, validation_target = fixture()
    learning_rate, l2 = np.exp(parameters[:2])
    minimum = 1/(1+np.exp(-parameters[2]))
    scales = np.exp(parameters[4:])
    theta = np.array([0.21, 0.06], dtype=np.float64)
    for update in range(1, STEPS+1):
        progress = min(1.0, update/TOTAL_UPDATES)
        factor = minimum + (1-minimum)*0.5*(1+np.cos(np.pi*progress))
        gradient = loss_gradient(theta, train_x, train_target, l2)
        theta = theta-learning_rate*factor*scales*gradient
    residual = validation_x[:, 0]*theta[0] + theta[1] - validation_target[:, 0]
    return 0.5*float(np.mean(residual*residual))


def oracle() -> dict[str, Any]:
    value = trajectory(PARAMETERS)
    gradient = np.empty(N_PARAMETERS, dtype=np.float64)
    started = time.perf_counter()
    for index in range(N_PARAMETERS):
        plus, minus = PARAMETERS.copy(), PARAMETERS.copy()
        plus[index] += FD_STEP
        minus[index] -= FD_STEP
        gradient[index] = (trajectory(plus)-trajectory(minus))/(2*FD_STEP)
    tangent = (trajectory(PARAMETERS+FD_STEP*DIRECTION)-trajectory(PARAMETERS-FD_STEP*DIRECTION))/(2*FD_STEP)
    elapsed = (time.perf_counter()-started)/REPETITIONS
    return {"value": value, "gradient": gradient, "tangent": tangent, "seconds": elapsed}


def base(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    row.update(details)
    row.update({"device": "cpu", "n_train": N_TRAIN, "n_validation": N_VALIDATION,
                "n_parameters": N_PARAMETERS, "steps": STEPS, "repetitions": REPETITIONS})
    row.update(values)
    return row


def read_oracle(path: Path) -> dict[tuple[str, int], float]:
    with path.open(newline="") as stream:
        return {(row["quantity"], int(row["index"])): float(row["value"])
                for row in csv.DictReader(stream)}


def unavailable(details: dict[str, str], device: str, status: str, notes: str) -> list[dict[str, Any]]:
    phases = [("value_gradient", "validation_mse")] + [("gradient_component", f"gradient_{i}") for i in range(1, 7)] + [("jvp", "directional_validation_mse_derivative")]
    return [base(details, workload="mlp_optimizer_group_schedule_hypergradient", phase=phase,
                 variant="fixed_full_batch_cosine", backend="fortml", device=device,
                 status=status, metric=metric, oracle="FortML release-app protocol", notes=notes)
            for phase, metric in phases]


def run_fortml(fortml: Path, target: str, details: dict[str, str], expected: dict[str, Any]) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment, capture_output=True, text=True)
    if build.returncode != 0:
        return unavailable(details, "cpu", "unavailable", "fo build failed")
    with tempfile.TemporaryDirectory(dir="/mnt/storage", prefix="fortml-optimizer-group-schedule-") as temporary:
        oracle_path = Path(temporary)/"oracle.csv"
        check_environment = dict(environment)
        check_environment.update({"FORTML_BENCH_OPTIMIZER_GROUP_SCHEDULE_ORACLE": str(oracle_path), "FORTML_BENCH_ORACLE_ONLY": "1"})
        check = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml, env=check_environment, capture_output=True, text=True)
        if check.returncode != 0 or not oracle_path.is_file():
            return unavailable(details, "cpu", "unavailable", "release target emitted no oracle")
        actual = read_oracle(oracle_path)
        required = {("value", 1), ("jvp", 1)} | {("gradient", i) for i in range(1, 7)}
        if set(actual) != required:
            raise RuntimeError("scheduled optimizer-group app omitted value/gradient/JVP contract")
        errors = [abs(actual[("value", 1)]-expected["value"]), abs(actual[("jvp", 1)]-expected["tangent"])]
        errors.extend(abs(actual[("gradient", i)]-expected["gradient"][i-1]) for i in range(1, 7))
        error = float(max(errors))
        if error > 5.0e-9:
            raise RuntimeError(f"scheduled optimizer-group oracle mismatch: {error:.3e}")
        timed = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml, env=environment, capture_output=True, text=True)
    timing = next((float(line.split(",", 1)[1]) for line in timed.stdout.splitlines() if line.startswith("mlp_optimizer_group_schedule_hypergradient_value_gradient,")), None)
    if timed.returncode != 0 or timing is None:
        return unavailable(details, "cpu", "unavailable", "timing execution failed")
    rows = [base(details, workload="mlp_optimizer_group_schedule_hypergradient", phase="value_gradient", variant="fixed_full_batch_cosine", backend="fortml", status="pass", seconds_per_operation=timing, metric="validation_mse", value=actual[("value", 1)], max_abs_error=error, oracle="independent NumPy cosine trajectory and central-FD products", notes="packed=[log_lr,log_l2,logit_min,logit_decay,log_multiplier_1,log_multiplier_2]")]
    rows.extend(base(details, workload="mlp_optimizer_group_schedule_hypergradient", phase="gradient_component", variant="fixed_full_batch_cosine", backend="fortml", status="pass", metric=f"gradient_{i}", value=actual[("gradient", i)], max_abs_error=abs(actual[("gradient", i)]-expected["gradient"][i-1]), oracle="independent NumPy cosine trajectory and central-FD products", notes="analytic schedule recurrence") for i in range(1, 7))
    rows.append(base(details, workload="mlp_optimizer_group_schedule_hypergradient", phase="jvp", variant="fixed_full_batch_cosine", backend="fortml", status="pass", metric="directional_validation_mse_derivative", value=actual[("jvp", 1)], max_abs_error=abs(actual[("jvp", 1)]-expected["tangent"]), oracle="independent NumPy cosine trajectory and central-FD products", notes="schedule plus group direction checked"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/mlp_optimizer_group_schedule_hypergradient.csv"))
    parser.add_argument("--target", default="fortml_bench_mlp_optimizer_group_schedule_hypergradient")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root, fortml, output = Path(__file__).resolve().parents[1], args.fortml.resolve(), args.output.resolve()
    details = {"python_version": platform.python_version(), "numpy_version": np.__version__, "fortml_revision": revision(fortml), "benchmark_revision": revision(root, (output.resolve(),)), "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3"}
    expected = oracle()
    rows = [base(details, workload="mlp_optimizer_group_schedule_hypergradient", phase="value_gradient", variant="fixed_full_batch_cosine", backend="numpy_oracle", status="pass", seconds_per_operation=expected["seconds"], metric="validation_mse", value=expected["value"], max_abs_error=0.0, oracle="independent NumPy cosine trajectory with central-FD products", notes="fixed total_updates=6; cosine min=0.2")]
    rows.extend(base(details, workload="mlp_optimizer_group_schedule_hypergradient", phase="gradient_component", variant="fixed_full_batch_cosine", backend="numpy_oracle", status="pass", metric=f"gradient_{i}", value=float(value), max_abs_error=0.0, oracle="independent central-FD scheduled outer objective", notes="schedule and group coordinates checked") for i, value in enumerate(expected["gradient"], 1))
    rows.append(base(details, workload="mlp_optimizer_group_schedule_hypergradient", phase="jvp", variant="fixed_full_batch_cosine", backend="numpy_oracle", status="pass", metric="directional_validation_mse_derivative", value=expected["tangent"], max_abs_error=0.0, oracle="independent central-FD scheduled directional product", notes=f"direction={DIRECTION.tolist()}; h={FD_STEP:g}"))
    rows.extend(unavailable(details, "cpu", "skipped", "--skip-fortml") if args.skip_fortml else run_fortml(fortml, args.target, details, expected))
    rows.extend(unavailable(details, "cuda", "unavailable", "resident scheduled optimizer-group kernels are not linked"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
