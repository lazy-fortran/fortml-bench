#!/usr/bin/env python3
"""Correctness-gated optimizer-group trajectory benchmark.

An independent NumPy two-parameter linear-MSE trajectory reproduces global
gradient clipping followed by post-SGD group scaling, central-differences all
packed coordinates on a fixed clipping active set, and gates the FortML timing
on the complete value/gradient/JVP oracle. CUDA rows are typed refusals until
the full group state is resident.
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
LEARNING_RATE = 0.07
L2 = 0.03
GRADIENT_CLIP_NORM = 0.1
GROUPS = np.array([0.65, 1.25], dtype=np.float64)
FD_STEP = 2.0e-6
REPETITIONS = 16
PARAMETERS = np.array([np.log(LEARNING_RATE), np.log(L2), *np.log(GROUPS)])
DIRECTION = np.array([0.23, -0.17, 0.13, -0.11])
ORACLE_TOLERANCE = 4.0e-10
FIELDS = (
    "workload", "phase", "variant", "backend", "device", "status",
    "n_train", "n_validation", "n_parameters", "steps", "repetitions",
    "seconds_per_operation", "metric", "value", "max_abs_error", "oracle",
    "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {"python_version": platform.python_version(), "numpy_version": np.__version__,
            "fortml_revision": revision(fortml), "benchmark_revision": revision(root, (output.resolve(),)),
            "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3"}


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.array([[-1.5], [-0.8], [-0.1], [0.6], [1.4], [2.0]], dtype=np.float64)
    train_target = 0.75 * train_x - 0.2
    validation_x = np.array([[-1.2], [0.25], [1.7]], dtype=np.float64)
    validation_target = 0.75 * validation_x - 0.2
    return train_x, train_target, validation_x, validation_target


def loss_gradient(theta: np.ndarray, x: np.ndarray, target: np.ndarray, l2: float) -> np.ndarray:
    residual = x[:, 0] * theta[0] + theta[1] - target[:, 0]
    return np.array([float(np.mean(residual * x[:, 0])) + l2 * theta[0],
                     float(np.mean(residual)) + l2 * theta[1]])


def trajectory(parameters: np.ndarray) -> float:
    train_x, train_target, validation_x, validation_target = fixture()
    learning_rate, l2 = np.exp(parameters[:2])
    scales = np.exp(parameters[2:])
    theta = np.array([0.21, 0.06], dtype=np.float64)
    for _ in range(STEPS):
        gradient = loss_gradient(theta, train_x, train_target, l2)
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm > GRADIENT_CLIP_NORM:
            gradient = gradient*GRADIENT_CLIP_NORM/gradient_norm
        theta = theta - learning_rate * scales * gradient
    residual = validation_x[:, 0] * theta[0] + theta[1] - validation_target[:, 0]
    return 0.5 * float(np.mean(residual * residual))


def oracle() -> dict[str, Any]:
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
    if not np.isfinite(value) or not np.all(np.isfinite(gradient)) or not np.isfinite(tangent):
        raise RuntimeError("optimizer-group NumPy oracle is nonfinite")
    return {"value": value, "gradient": gradient, "tangent": tangent, "seconds": elapsed}


def base(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    row.update(details)
    row.update({"device": "cpu", "n_train": N_TRAIN, "n_validation": N_VALIDATION,
                "n_parameters": N_PARAMETERS, "steps": STEPS, "repetitions": REPETITIONS})
    row.update(values)
    return row


def unavailable(details: dict[str, str], device: str, status: str, notes: str) -> list[dict[str, Any]]:
    phases = [("value_gradient", "validation_mse"), ("gradient_component", "gradient_1"),
              ("gradient_component", "gradient_2"), ("gradient_component", "gradient_3"),
              ("gradient_component", "gradient_4"), ("jvp", "directional_validation_mse_derivative"),
              ("hvp_refusal", "status_code")]
    return [base(details, workload="mlp_optimizer_group_hypergradient", phase=phase,
                 variant="fixed_full_batch", backend="fortml", device=device, status=status,
                 metric=metric, oracle="FortML release-app protocol", notes=notes)
            for phase, metric in phases]


def read_oracle(path: Path) -> dict[tuple[str, int], float]:
    with path.open(newline="") as stream:
        return {(row["quantity"], int(row["index"])): float(row["value"])
                for row in csv.DictReader(stream)}


def run_fortml(fortml: Path, target: str, details: dict[str, str], expected: dict[str, Any]) -> list[dict[str, Any]]:
    if not (fortml / "app" / f"{target}.f90").is_file():
        return unavailable(details, "cpu", "unavailable", "release source absent")
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
                           capture_output=True, text=True)
    if build.returncode != 0:
        return unavailable(details, "cpu", "unavailable", "fo build failed")
    with tempfile.TemporaryDirectory(dir="/mnt/storage", prefix="fortml-optimizer-group-") as temporary:
        oracle_path = Path(temporary) / "oracle.csv"
        check_environment = dict(environment)
        check_environment.update({"FORTML_BENCH_OPTIMIZER_GROUP_HYPERGRADIENT_ORACLE": str(oracle_path),
                                   "FORTML_BENCH_ORACLE_ONLY": "1"})
        check = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                               env=check_environment, capture_output=True, text=True)
        if check.returncode != 0 or not oracle_path.is_file():
            return unavailable(details, "cpu", "unavailable", "release target emitted no oracle")
        actual = read_oracle(oracle_path)
        required = {("value", 1), ("jvp", 1), ("hvp_status", 1)} | {("gradient", i) for i in range(1, N_PARAMETERS + 1)}
        if set(actual) != required:
            raise RuntimeError("FortML optimizer-group app omitted complete value/gradient/JVP/HVP contract")
        errors = [abs(actual[("value", 1)] - expected["value"]), abs(actual[("jvp", 1)] - expected["tangent"])]
        errors.extend(abs(actual[("gradient", i)] - expected["gradient"][i - 1]) for i in range(1, N_PARAMETERS + 1))
        error = float(max(errors))
        if error > ORACLE_TOLERANCE:
            raise RuntimeError(f"FortML optimizer-group oracle mismatch: {error:.3e}")
        if actual[("hvp_status", 1)] != 3.0:
            raise RuntimeError("FortML optimizer-group HVP did not return FORTNUM_NOT_IMPLEMENTED")
        timed = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                               env=environment, capture_output=True, text=True)
    if timed.returncode != 0:
        return unavailable(details, "cpu", "unavailable", "timing execution failed")
    timing = next((float(line.split(",", 1)[1].strip()) for line in timed.stdout.splitlines()
                   if line.startswith("mlp_optimizer_group_hypergradient_value_gradient,")), None)
    if timing is None:
        raise RuntimeError("FortML optimizer-group app emitted no timing")
    rows = [base(details, workload="mlp_optimizer_group_hypergradient", phase="value_gradient",
                 variant="fixed_full_batch", backend="fortml", status="pass",
                 seconds_per_operation=timing, metric="validation_mse", value=actual[("value", 1)],
                 max_abs_error=error, oracle="independent NumPy clipped group trajectory and central-FD products",
                 notes=f"{target}; fixed_clip={GRADIENT_CLIP_NORM:g}; four gradients and JVP checked")]
    rows.extend(base(details, workload="mlp_optimizer_group_hypergradient", phase="gradient_component",
                     variant="fixed_full_batch", backend="fortml", status="pass", metric=f"gradient_{i}",
                     value=actual[("gradient", i)], max_abs_error=abs(actual[("gradient", i)] - expected["gradient"][i - 1]),
                     oracle="independent NumPy clipped group trajectory and central-FD products",
                     notes=f"{target}; fixed_clip={GRADIENT_CLIP_NORM:g}")
                for i in range(1, N_PARAMETERS + 1))
    rows.append(base(details, workload="mlp_optimizer_group_hypergradient", phase="jvp",
                     variant="fixed_full_batch", backend="fortml", status="pass",
                     metric="directional_validation_mse_derivative", value=actual[("jvp", 1)],
                     max_abs_error=abs(actual[("jvp", 1)] - expected["tangent"]),
                     oracle="independent NumPy clipped group trajectory and central-FD products",
                     notes=f"{target}; fixed_clip={GRADIENT_CLIP_NORM:g}"))
    rows.append(base(details, workload="mlp_optimizer_group_hypergradient", phase="hvp_refusal",
                     variant="fixed_full_batch", backend="fortml", status="pass",
                     metric="status_code", value=actual[("hvp_status", 1)], max_abs_error=0.0,
                     oracle="public typed third-derivative boundary",
                     notes="HVP returns FORTNUM_NOT_IMPLEMENTED with zero product"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/mlp_optimizer_group_hypergradient.csv"))
    parser.add_argument("--target", default="fortml_bench_mlp_optimizer_group_hypergradient")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root, fortml, output = Path(__file__).resolve().parents[1], args.fortml.resolve(), args.output.resolve()
    details, expected = metadata(root, fortml, output), oracle()
    rows = [base(details, workload="mlp_optimizer_group_hypergradient", phase="value_gradient",
                 variant="fixed_full_batch", backend="numpy_oracle", status="pass",
                 seconds_per_operation=expected["seconds"], metric="validation_mse", value=expected["value"],
                 max_abs_error=0.0, oracle="independent NumPy clipped group trajectory with central-FD products",
                 notes=f"packed=[log_lr,log_l2,log_multiplier_1,log_multiplier_2]; fixed_clip={GRADIENT_CLIP_NORM:g}")]
    rows.extend(base(details, workload="mlp_optimizer_group_hypergradient", phase="gradient_component",
                     variant="fixed_full_batch", backend="numpy_oracle", status="pass", metric=f"gradient_{i}",
                     value=float(value), max_abs_error=0.0,
                     oracle="independent central finite-difference clipped outer objective",
                     notes=f"fixed_clip={GRADIENT_CLIP_NORM:g}; all group coordinates checked")
                for i, value in enumerate(expected["gradient"], start=1))
    rows.append(base(details, workload="mlp_optimizer_group_hypergradient", phase="jvp",
                     variant="fixed_full_batch", backend="numpy_oracle", status="pass",
                     metric="directional_validation_mse_derivative", value=expected["tangent"], max_abs_error=0.0,
                     oracle="independent central-FD clipped directional product",
                     notes=f"fixed_clip={GRADIENT_CLIP_NORM:g}; direction={DIRECTION.tolist()}; h={FD_STEP:g}"))
    rows.append(base(details, workload="mlp_optimizer_group_hypergradient", phase="hvp_refusal",
                     variant="fixed_full_batch", backend="numpy_oracle", status="not_applicable",
                     metric="status_code", value="nan", max_abs_error="nan",
                     oracle="public typed third-derivative boundary",
                     notes="NumPy oracle does not emulate unsupported third network derivatives"))
    rows.extend(unavailable(details, "cpu", "skipped", "--skip-fortml") if args.skip_fortml
                else run_fortml(fortml, args.target, details, expected))
    rows.extend(unavailable(details, "cuda", "unavailable", "complete group state derivatives are CPU-only until resident kernels exist"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
