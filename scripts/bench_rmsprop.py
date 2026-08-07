#!/usr/bin/env python3
"""Independent RMSprop recurrence and MLP-training benchmark."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
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
            "fortml_revision": revision(fortml), "benchmark_revision": revision(root, (output,)),
            "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3"}


def direct_oracle() -> tuple[float, float]:
    index = np.arange(1, N_PARAMETERS + 1, dtype=np.float64)
    theta = 0.1 * np.cos(0.003 * index)
    target = 0.25 * np.sin(0.0017 * index)
    square = np.zeros_like(theta)
    started = time.perf_counter()
    for _ in range(STEPS):
        gradient = theta - target
        square = 0.9 * square + 0.1 * gradient**2
        theta -= 1.0e-2 * gradient / (np.sqrt(square) + 1.0e-8)
    elapsed = (time.perf_counter() - started) / REPETITIONS
    return float(np.linalg.norm(theta)), elapsed


def mlp_oracle() -> tuple[float, float]:
    weight, bias = 0.0, 0.0
    square = np.zeros(2)
    mean_gradient = np.zeros(2)
    momentum_buffer = np.zeros(2)
    for _ in range(EPOCHS):
        gradient = np.array([2.0 * (weight - 1.0) / 3.0, bias])
        square = 0.8 * square + 0.2 * gradient**2
        mean_gradient = 0.8 * mean_gradient + 0.2 * gradient
        variance = np.maximum(square - mean_gradient**2, 0.0)
        direction = gradient / (np.sqrt(variance) + 1.0e-5)
        momentum_buffer = 0.2 * momentum_buffer + direction
        weight, bias = np.array([weight, bias]) - 0.08 * momentum_buffer
    residual = weight * np.array([-1.0, 0.0, 1.0]) + bias - np.array([-1.0, 0.0, 1.0])
    return float(0.5 * np.mean(residual**2)), 0.0


def base(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    row.update(details)
    row.update({"device": "cpu", "n_parameters": N_PARAMETERS, "steps": STEPS,
                "n_samples": N_SAMPLES, "epochs": EPOCHS})
    row.update(values)
    return row


def run_fortml(fortml: Path, target: str, details: dict[str, str], expected_norm: float,
               expected_loss: float) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    source = fortml / "app" / f"{target}.f90"
    if not source.is_file():
        return [base(details, workload="rmsprop", phase="train", backend="fortml", status="unavailable",
                     oracle="FortML release-app protocol", notes=f"release target source is absent: {source.name}")]
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
                           capture_output=True, text=True)
    if build.returncode != 0:
        return [base(details, workload="rmsprop", phase="train", backend="fortml", status="unavailable",
                     oracle="FortML release-app protocol", notes="fo build failed")]
    run = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml, env=environment,
                         capture_output=True, text=True)
    if run.returncode != 0:
        return [base(details, workload="rmsprop", phase="train", backend="fortml", status="unavailable",
                     oracle="FortML release-app protocol", notes="release target execution failed")]
    patterns = {
        "rmsprop_training": (expected_norm, "parameter_l2_norm", 1),
        "rmsprop_mlp": (expected_loss, "final_loss", 1),
    }
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in run.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if not fields or fields[0] not in patterns:
            continue
        name = fields[0]
        if name == "rmsprop_training" and len(fields) != 5:
            continue
        if name == "rmsprop_mlp" and len(fields) != 6:
            continue
        seen.add(name)
        expected, metric, value_index = patterns[name]
        actual = float(fields[3])
        seconds = float(fields[4])
        error = abs(actual - expected)
        if error > 2.0e-11:
            raise RuntimeError(f"FortML RMSprop {name} oracle mismatch: {error:.3e}")
        rows.append(base(details, workload=name, phase="train", backend="fortml", status="pass",
                         seconds_per_operation=seconds, metric=metric, value=actual,
                         max_abs_error=error,
                         oracle="independent NumPy RMSprop recurrence and MSE oracle",
                         notes="stateful update; MLP lane uses centered decay and momentum"))
    for name in patterns:
        if name not in seen:
            rows.append(base(details, workload=name, phase="train", backend="fortml", status="unavailable",
                             oracle="FortML release-app protocol", notes="release app emitted no parseable row"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/rmsprop.csv"))
    parser.add_argument("--target", default="fortml_bench_rmsprop_training")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    details = metadata(root, fortml, output)
    norm, norm_seconds = direct_oracle()
    loss, _ = mlp_oracle()
    rows = [base(details, workload="rmsprop_training", phase="train", backend="numpy_oracle", status="pass",
                 seconds_per_operation=norm_seconds, metric="parameter_l2_norm", value=norm,
                 max_abs_error=0.0, oracle="independent FortOpt RMSprop recurrence",
                 notes="decay=0.9; epsilon=1e-8; uncentered, no momentum"),
            base(details, workload="rmsprop_mlp", phase="train", backend="numpy_oracle", status="pass",
                 seconds_per_operation="", metric="final_loss", value=loss, max_abs_error=0.0,
                 oracle="independent centered MLP RMSprop recurrence", notes="decay=0.8; momentum=0.2")]
    if args.skip_fortml:
        rows.extend(base(details, workload=name, phase="train", backend="fortml", status="skipped",
                         oracle="FortML release-app protocol", notes="--skip-fortml")
                    for name in ("rmsprop_training", "rmsprop_mlp"))
    else:
        rows.extend(run_fortml(fortml, args.target, details, norm, loss))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
