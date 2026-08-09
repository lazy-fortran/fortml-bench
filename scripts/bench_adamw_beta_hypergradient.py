#!/usr/bin/env python3
"""Correctness-gated five-parameter AdamW beta-logit hypergradient lane.

FortML now exposes an exact fixed-trajectory objective over log learning rate,
log L2, log decoupled weight decay, and unconstrained logits for beta1/beta2.
There is not yet a release app exporting complete value/gradient/JVP arrays,
so this lane records an independent NumPy oracle and explicit FortML
``unavailable`` rows.  A future app must emit the complete arrays under the
strict protocol documented in ``results/ADAMW_BETA_HYPERGRADIENT.md``.
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


N_SAMPLES = 8
N_TRAIN = 5
N_VALIDATION = 3
N_FEATURES = 1
N_HIDDEN = 1
N_OUTPUTS = 1
STEPS = 4
LEARNING_RATE = 0.12
L2 = 0.07
WEIGHT_DECAY = 0.03
BETA1 = 0.82
BETA2 = 0.91
EPSILON = 1.0e-8
FD_STEP = 2.0e-6
HVP_STEP = 1.0e-4
REPETITIONS = 32
ORACLE_TOLERANCE = 2.0e-6

FIELDS = (
    "workload", "phase", "variant", "backend", "device", "status",
    "n_samples", "n_features", "n_hidden", "n_outputs", "steps",
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


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    generated_outputs = tuple(root / "results" / name for name in (
        "adamw_beta_hypergradient.csv", "cuda_adamw.csv", "ridge.csv",
        "xgboost_workloads.csv"))
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, generated_outputs),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def base(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = {field: "" for field in FIELDS}
    row.update(details)
    row.update({
        "variant": "fixed_adamw_beta_logits", "device": "cpu",
        "n_samples": N_SAMPLES, "n_features": N_FEATURES,
        "n_hidden": N_HIDDEN, "n_outputs": N_OUTPUTS, "steps": STEPS,
        "oracle": "independent NumPy AdamW beta-logit recurrence",
    })
    row.update(values)
    return row


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)[:, None]
    train_target = 0.7 * train_x - 0.2
    validation_x = np.array([-1.5, 0.5, 1.75], dtype=np.float64)[:, None]
    validation_target = 0.7 * validation_x - 0.2
    return train_x, train_target, validation_x, validation_target


def scalar_value_gradient(theta: np.ndarray, x: np.ndarray, target: np.ndarray,
                          l2: float) -> tuple[float, np.ndarray]:
    prediction = x[:, 0] * theta[0] + theta[1]
    residual = prediction - target[:, 0]
    value = 0.5 * float(np.mean(residual * residual)) + 0.5 * l2 * float(np.sum(theta * theta))
    gradient = np.array([
        float(np.mean(residual * x[:, 0])) + l2 * theta[0],
        float(np.mean(residual)) + l2 * theta[1],
    ])
    return value, gradient


def logit(probability: float) -> float:
    return float(np.log(probability / (1.0 - probability)))


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def full_value(parameters: np.ndarray) -> float:
    """Evaluate validation MSE after the complete inner AdamW trajectory."""
    learning_rate, l2, weight_decay = np.exp(parameters[:3])
    beta1, beta2 = sigmoid(float(parameters[3])), sigmoid(float(parameters[4]))
    train_x, train_target, validation_x, validation_target = fixture()
    theta = np.array([0.15, -0.1], dtype=np.float64)
    first = np.zeros(2, dtype=np.float64)
    second = np.zeros(2, dtype=np.float64)
    for step in range(1, STEPS + 1):
        _, gradient = scalar_value_gradient(theta, train_x, train_target, l2)
        first = beta1 * first + (1.0 - beta1) * gradient
        second = beta2 * second + (1.0 - beta2) * gradient * gradient
        first_hat = first / (1.0 - beta1**step)
        second_hat = second / (1.0 - beta2**step)
        theta = ((1.0 - learning_rate * weight_decay) * theta -
                 learning_rate * first_hat / (np.sqrt(second_hat) + EPSILON))
    prediction = validation_x[:, 0] * theta[0] + theta[1]
    residual = prediction - validation_target[:, 0]
    return 0.5 * float(np.mean(residual * residual))


def finite_gradient(parameters: np.ndarray) -> np.ndarray:
    """Independent scalar-oracle gradient used only by the benchmark."""
    result = np.empty(5, dtype=np.float64)
    for index in range(5):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += FD_STEP
        minus[index] -= FD_STEP
        result[index] = (full_value(plus) - full_value(minus)) / (2.0 * FD_STEP)
    return result


def oracle() -> dict[str, Any]:
    parameters = np.array([
        np.log(LEARNING_RATE), np.log(L2), np.log(WEIGHT_DECAY),
        logit(BETA1), logit(BETA2),
    ], dtype=np.float64)
    gradient = finite_gradient(parameters)
    direction = np.array([0.31, -0.27, 0.19, 0.13, -0.22], dtype=np.float64)
    jvp = ((full_value(parameters + FD_STEP * direction) -
            full_value(parameters - FD_STEP * direction)) / (2.0 * FD_STEP))
    hvp = (finite_gradient(parameters + HVP_STEP * direction) -
           finite_gradient(parameters - HVP_STEP * direction)) / (2.0 * HVP_STEP)
    value = full_value(parameters)
    if (not np.all(np.isfinite(gradient)) or not np.isfinite(value) or
            not np.isfinite(jvp) or not np.all(np.isfinite(hvp))):
        raise RuntimeError("AdamW beta-logit NumPy oracle is nonfinite")
    return {"parameters": parameters, "value": value, "gradient": gradient,
            "direction": direction, "jvp": float(jvp), "hvp": hvp}


def oracle_rows(details: dict[str, str], expected: dict[str, Any]) -> list[dict[str, Any]]:
    started = time.perf_counter()
    for _ in range(REPETITIONS):
        checked = oracle()
    seconds = (time.perf_counter() - started) / REPETITIONS
    value_error = abs(checked["value"] - expected["value"])
    gradient_error = float(np.max(np.abs(checked["gradient"] - expected["gradient"])))
    jvp_error = abs(checked["jvp"] - expected["jvp"])
    hvp_error = float(np.max(np.abs(checked["hvp"] - expected["hvp"])))
    if max(value_error, gradient_error, jvp_error, hvp_error) != 0.0:
        raise RuntimeError("repeated AdamW beta-logit oracle is not deterministic")
    rows = [base(
        details, workload="mlp_adamw_beta_hypergradient", phase="value_gradient",
        backend="numpy_oracle", status="pass", repetitions=REPETITIONS,
        seconds_per_operation=seconds, metric="validation_mse", value=expected["value"],
        max_abs_error=0.0,
        notes=(f"parameters={expected['parameters'].tolist()}; fd_step={FD_STEP:g}; "
               f"beta1={BETA1:g}; beta2={BETA2:g}"),
    ), base(
        details, workload="mlp_adamw_beta_hypergradient", phase="jvp",
        backend="numpy_oracle", status="pass", repetitions=REPETITIONS,
        seconds_per_operation=seconds,
        metric="directional_validation_mse_derivative", value=expected["jvp"],
        max_abs_error=0.0,
        notes=f"direction={expected['direction'].tolist()}; fd_step={FD_STEP:g}",
    )]
    names = ("log_learning_rate", "log_l2", "log_weight_decay", "logit_beta1", "logit_beta2")
    rows.extend(base(
        details, workload="mlp_adamw_beta_hypergradient", phase="gradient_component",
        backend="numpy_oracle", status="pass", repetitions=1,
        metric=f"gradient_{name}", value=value, max_abs_error=0.0,
    ) for name, value in zip(names, expected["gradient"]))
    rows.extend(base(
        details, workload="mlp_adamw_beta_hypergradient", phase="hvp_component",
        backend="numpy_oracle", status="pass", repetitions=1,
        metric=f"hvp_{name}", value=value, max_abs_error=0.0,
        notes=(f"direction={expected['direction'].tolist()}; gradient_fd_step={FD_STEP:g}; "
               f"hvp_fd_step={HVP_STEP:g}"),
    ) for name, value in zip(names, expected["hvp"]))
    return rows


def unavailable_rows(details: dict[str, str], note: str) -> list[dict[str, Any]]:
    rows = [base(
        details, workload="mlp_adamw_beta_hypergradient", phase="value_gradient",
        backend="fortml", device="cpu", status="unavailable", repetitions="",
        oracle="FortML complete-array release-app protocol", notes=note,
    ), base(
        details, workload="mlp_adamw_beta_hypergradient", phase="jvp",
        backend="fortml", device="cpu", status="unavailable", repetitions="",
        oracle="FortML complete-array release-app protocol", notes=note,
    )]
    rows.extend(base(
        details, workload="mlp_adamw_beta_hypergradient", phase="gradient_component",
        backend="fortml", device="cpu", status="unavailable", repetitions="",
        metric=f"gradient_{name}", oracle="FortML complete-array release-app protocol",
        notes=note,
    ) for name in ("log_learning_rate", "log_l2", "log_weight_decay", "logit_beta1", "logit_beta2"))
    rows.extend(base(
        details, workload="mlp_adamw_beta_hypergradient", phase="hvp_component",
        backend="fortml", device="cpu", status="unavailable", repetitions="",
        metric=f"hvp_{name}", oracle="FortML complete-array release-app protocol",
        notes=note,
    ) for name in ("log_learning_rate", "log_l2", "log_weight_decay", "logit_beta1", "logit_beta2"))
    return rows


def read_oracle(path: Path) -> dict[tuple[str, int], float]:
    values: dict[tuple[str, int], float] = {}
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            values[(record["quantity"], int(record["index"]))] = float(record["value"])
    return values


def run_fortml(fortml: Path, target: str, details: dict[str, str],
               expected: dict[str, Any]) -> list[dict[str, Any]]:
    """Require complete value/gradient/JVP/HVP output before retaining timing."""
    source = fortml / "app" / f"{target}.f90"
    if not source.is_file():
        return unavailable_rows(details, f"release target source is absent: {source.name}")
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                           env=environment, capture_output=True, text=True)
    if build.returncode != 0:
        return unavailable_rows(details, "fo build failed; no FortML timing retained")
    with tempfile.TemporaryDirectory(dir="/mnt/storage") as directory:
        oracle_path = Path(directory) / "adamw_beta_oracle.csv"
        oracle_environment = dict(environment)
        oracle_environment.update({
            "FORTML_BENCH_ADAMW_BETA_ORACLE": str(oracle_path),
            "FORTML_BENCH_ORACLE_ONLY": "1",
        })
        check = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                               env=oracle_environment, capture_output=True, text=True)
        if check.returncode != 0 or not oracle_path.is_file():
            return unavailable_rows(details, "release target did not emit its complete oracle")
        actual = read_oracle(oracle_path)
        required = {("value", 1), ("jvp", 1)} | {("gradient", index) for index in range(1, 6)} | \
            {("hvp", index) for index in range(1, 6)}
        if set(actual) != required:
            raise RuntimeError("FortML AdamW beta app omitted a complete value/gradient/JVP/HVP array")
        errors = [abs(actual[("value", 1)] - expected["value"]),
                  abs(actual[("jvp", 1)] - expected["jvp"])]
        errors.extend(abs(actual[("gradient", index)] - expected["gradient"][index - 1])
                      for index in range(1, 6))
        errors.extend(abs(actual[("hvp", index)] - expected["hvp"][index - 1])
                      for index in range(1, 6))
        error = float(max(errors))
        if error > ORACLE_TOLERANCE:
            raise RuntimeError(f"FortML AdamW beta oracle mismatch: {error:.3e}")
        timed = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                               env=environment, capture_output=True, text=True)
    if timed.returncode != 0:
        return unavailable_rows(details, "release target timing execution failed")
    timing = None
    marker = "mlp_adamw_beta_hypergradient_value_gradient,"
    for line in timed.stdout.splitlines():
        if line.startswith(marker):
            timing = float(line.split(",", 1)[1].strip())
            break
    if timing is None:
        raise RuntimeError("FortML AdamW beta app emitted no value-gradient timing")
    rows = [base(
        details, workload="mlp_adamw_beta_hypergradient", phase="value_gradient",
        backend="fortml", status="pass", repetitions=REPETITIONS,
        seconds_per_operation=timing, metric="validation_mse", value=actual[("value", 1)],
        max_abs_error=error, oracle="complete NumPy value/gradient/JVP/HVP arrays", notes=target,
    ), base(
        details, workload="mlp_adamw_beta_hypergradient", phase="jvp",
        backend="fortml", status="pass", repetitions=REPETITIONS,
        metric="directional_validation_mse_derivative", value=actual[("jvp", 1)],
        max_abs_error=error, oracle="complete NumPy value/gradient/JVP/HVP arrays", notes=target,
    )]
    names = ("log_learning_rate", "log_l2", "log_weight_decay", "logit_beta1", "logit_beta2")
    rows.extend(base(
        details, workload="mlp_adamw_beta_hypergradient", phase="gradient_component",
        backend="fortml", status="pass", repetitions=1,
        metric=f"gradient_{name}", value=actual[("gradient", index)],
        max_abs_error=error, oracle="complete NumPy value/gradient/JVP/HVP arrays", notes=target,
    ) for index, name in enumerate(names, start=1))
    rows.extend(base(
        details, workload="mlp_adamw_beta_hypergradient", phase="hvp_component",
        backend="fortml", status="pass", repetitions=1,
        metric=f"hvp_{name}", value=actual[("hvp", index)],
        max_abs_error=error, oracle="complete NumPy value/gradient/JVP/HVP arrays", notes=target,
    ) for index, name in enumerate(names, start=1))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/adamw_beta_hypergradient.csv"))
    parser.add_argument("--target", default="fortml_bench_adamw_beta_hypergradient")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    details = metadata(root, fortml, output)
    expected = oracle()
    rows = oracle_rows(details, expected)
    source = fortml / "app" / f"{args.target}.f90"
    if args.skip_fortml:
        rows.extend(unavailable_rows(details, "--skip-fortml requested"))
    elif not source.is_file():
        rows.extend(unavailable_rows(
            details, f"release target source is absent: {source.name}; complete-array protocol reserved"))
    else:
        rows.extend(run_fortml(fortml, args.target, details, expected))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
