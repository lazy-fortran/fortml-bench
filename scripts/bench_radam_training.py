#!/usr/bin/env python3
"""Independent RAdam recurrence and MLP-trainer benchmark."""

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


N_PARAMETERS = 4096
STEPS = 128
N_SAMPLES = 3
EPOCHS = 32
REPETITIONS = 16
MLP_REPETITIONS = 8
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_parameters", "steps",
    "n_samples", "epochs", "seconds_per_operation", "metric", "value", "max_abs_error",
    "oracle", "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty_lines = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines()
    dirty = [line for line in dirty_lines
             if (repository / line[3:].split(" -> ")[-1].strip()).resolve() not in ignored_paths]
    return head + ("+dirty" if dirty else "")


def base(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    row.update(details)
    row.update({"device": "cpu", "n_parameters": N_PARAMETERS, "steps": STEPS,
                "n_samples": N_SAMPLES, "epochs": EPOCHS})
    row.update(values)
    return row


def radam_update(theta: np.ndarray, gradient: np.ndarray, first: np.ndarray,
                 second: np.ndarray, step: int, learning_rate: float,
                 beta1: float, beta2: float, epsilon: float) -> None:
    """Apply one RAdam update in-place using the paper's rho_t threshold."""
    first *= beta1
    first += (1.0 - beta1) * gradient
    second *= beta2
    second += (1.0 - beta2) * gradient**2
    bias1 = 1.0 - beta1**step
    bias2 = 1.0 - beta2**step
    first_corrected = first / bias1
    rho_inf = 2.0 / (1.0 - beta2) - 1.0
    rho_t = rho_inf - 2.0 * step * beta2**step / bias2
    if rho_t > 4.0:
        rectification = np.sqrt(
            (rho_t - 4.0) * (rho_t - 2.0) * rho_inf
            / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
        )
        theta -= learning_rate * rectification * first_corrected / (
            np.sqrt(second / bias2) + epsilon
        )
    else:
        theta -= learning_rate * first_corrected


def direct_oracle() -> tuple[float, float]:
    index = np.arange(1, N_PARAMETERS + 1, dtype=np.float64)
    theta = 0.1 * np.cos(0.003 * index)
    target = 0.25 * np.sin(0.0017 * index)
    first = np.zeros_like(theta)
    second = np.zeros_like(theta)
    started = time.perf_counter()
    for step in range(1, STEPS + 1):
        radam_update(theta, theta - target, first, second, step, 1.0e-2, 0.9, 0.99, 1.0e-8)
    elapsed = (time.perf_counter() - started) / REPETITIONS
    return float(np.linalg.norm(theta)), elapsed


def mlp_oracle() -> float:
    theta = np.zeros(2, dtype=np.float64)
    first = np.zeros(2, dtype=np.float64)
    second = np.zeros(2, dtype=np.float64)
    for step in range(1, EPOCHS + 1):
        gradient = np.array([2.0 * (theta[0] - 1.0) / 3.0, theta[1]])
        radam_update(theta, gradient, first, second, step, 0.08, 0.85, 0.95, 1.0e-5)
    residual = theta[0] * np.array([-1.0, 0.0, 1.0]) + theta[1] - np.array([-1.0, 0.0, 1.0])
    return float(0.5 * np.mean(residual**2))


def run_fortml(fortml: Path, target: str, details: dict[str, str], expected_norm: float,
               expected_loss: float) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    source = fortml / "app" / f"{target}.f90"
    names = ("radam_training", "radam_mlp")
    if not source.is_file():
        return [base(details, workload=name, phase="train", backend="fortml", status="unavailable",
                     oracle="FortML release-app protocol", notes=f"release target source is absent: {source.name}")
                for name in names]
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
                           capture_output=True, text=True)
    if build.returncode != 0:
        return [base(details, workload=name, phase="train", backend="fortml", status="unavailable",
                     oracle="FortML release-app protocol", notes="fo build failed") for name in names]
    run = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml, env=environment,
                         capture_output=True, text=True)
    if run.returncode != 0:
        return [base(details, workload=name, phase="train", backend="fortml", status="unavailable",
                     oracle="FortML release-app protocol", notes="release target execution failed") for name in names]
    patterns = {"radam_training": (expected_norm, "parameter_l2_norm"),
                "radam_mlp": (expected_loss, "final_loss")}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in run.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if not fields or fields[0] not in patterns:
            continue
        if (fields[0] == "radam_training" and len(fields) != 5) or (
                fields[0] == "radam_mlp" and len(fields) != 6):
            continue
        name = fields[0]
        seen.add(name)
        expected, metric = patterns[name]
        actual = float(fields[3])
        seconds = float(fields[4])
        error = abs(actual - expected)
        if error > 3.0e-11:
            raise RuntimeError(f"FortML RAdam {name} oracle mismatch: {error:.3e}")
        rows.append(base(details, workload=name, phase="train", backend="fortml", status="pass",
                         seconds_per_operation=seconds, metric=metric, value=actual,
                         max_abs_error=error,
                         oracle="independent NumPy RAdam recurrence and MSE oracle",
                         notes="bias-corrected moments with rho_t rectification threshold"))
    for name in names:
        if name not in seen:
            rows.append(base(details, workload=name, phase="train", backend="fortml", status="unavailable",
                             oracle="FortML release-app protocol", notes="release app emitted no parseable row"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/radam.csv"))
    parser.add_argument("--target", default="fortml_bench_radam_training")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    details = {"python_version": platform.python_version(), "numpy_version": np.__version__,
               "fortml_revision": revision(fortml),
               "benchmark_revision": revision(root, (output, root / "scripts" / "__pycache__")),
               "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3"}
    norm, norm_seconds = direct_oracle()
    loss = mlp_oracle()
    rows = [base(details, workload="radam_training", phase="train", backend="numpy_oracle", status="pass",
                 seconds_per_operation=norm_seconds, metric="parameter_l2_norm", value=norm,
                 max_abs_error=0.0, oracle="independent RAdam recurrence",
                 notes="beta1=0.9; beta2=0.99; epsilon=1e-8; rho_t threshold=4"),
            base(details, workload="radam_mlp", phase="train", backend="numpy_oracle", status="pass",
                 metric="final_loss", value=loss, max_abs_error=0.0,
                 oracle="independent RAdam MLP recurrence",
                 notes="beta1=0.85; beta2=0.95; epsilon=1e-5; rho_t threshold=4")]
    if args.skip_fortml:
        rows.extend(base(details, workload=name, phase="train", backend="fortml", status="skipped",
                         oracle="FortML release-app protocol", notes="--skip-fortml")
                    for name in ("radam_training", "radam_mlp"))
    else:
        rows.extend(run_fortml(fortml, args.target, details, norm, loss))
    for name in ("radam_training", "radam_mlp"):
        rows.append(base(details, workload=name, phase="train", backend="fortml", device="cuda",
                         status="unavailable", seconds_per_operation="", metric="", value="",
                         max_abs_error="", oracle="FortML device contract",
                         notes="resident RAdam state kernel is not linked; no host fallback"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
