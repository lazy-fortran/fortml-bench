#!/usr/bin/env python3
"""Independent AMSGrad recurrence and MLP-trainer benchmark."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
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


def direct_oracle() -> tuple[float, float]:
    index = np.arange(1, N_PARAMETERS + 1, dtype=np.float64)
    theta = 0.1 * np.cos(0.003 * index)
    target = 0.25 * np.sin(0.0017 * index)
    first = np.zeros_like(theta)
    second = np.zeros_like(theta)
    maximum = np.zeros_like(theta)
    started = __import__("time").perf_counter()
    for _ in range(STEPS):
        gradient = theta - target
        first = 0.9 * first + 0.1 * gradient
        second = 0.99 * second + 0.01 * gradient**2
        maximum = np.maximum(maximum, second)
        step = float(_ + 1)
        theta -= 1.0e-2 * (first / (1.0 - 0.9**step)) / (
            np.sqrt(maximum / (1.0 - 0.99**step)) + 1.0e-8
        )
    elapsed = (__import__("time").perf_counter() - started) / REPETITIONS
    return float(np.linalg.norm(theta)), elapsed


def mlp_oracle() -> float:
    theta = np.zeros(2, dtype=np.float64)
    first = np.zeros(2, dtype=np.float64)
    second = np.zeros(2, dtype=np.float64)
    maximum = np.zeros(2, dtype=np.float64)
    for step in range(1, EPOCHS + 1):
        gradient = np.array([2.0 * (theta[0] - 1.0) / 3.0, theta[1]])
        first = 0.85 * first + 0.15 * gradient
        second = 0.95 * second + 0.05 * gradient**2
        maximum = np.maximum(maximum, second)
        theta -= 0.08 * (first / (1.0 - 0.85**step)) / (
            np.sqrt(maximum / (1.0 - 0.95**step)) + 1.0e-5
        )
    residual = theta[0] * np.array([-1.0, 0.0, 1.0]) + theta[1] - np.array([-1.0, 0.0, 1.0])
    return float(0.5 * np.mean(residual**2))


def run_fortml(fortml: Path, target: str, details: dict[str, str], expected_norm: float,
               expected_loss: float) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    source = fortml / "app" / f"{target}.f90"
    names = ("amsgrad_training", "amsgrad_mlp")
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
    patterns = {"amsgrad_training": (expected_norm, "parameter_l2_norm"),
                "amsgrad_mlp": (expected_loss, "final_loss")}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in run.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if not fields or fields[0] not in patterns:
            continue
        if (fields[0] == "amsgrad_training" and len(fields) != 5) or (
                fields[0] == "amsgrad_mlp" and len(fields) != 6):
            continue
        name = fields[0]
        seen.add(name)
        expected, metric = patterns[name]
        actual = float(fields[3])
        seconds = float(fields[4])
        error = abs(actual - expected)
        if error > 3.0e-11:
            raise RuntimeError(f"FortML AMSGrad {name} oracle mismatch: {error:.3e}")
        rows.append(base(details, workload=name, phase="train", backend="fortml", status="pass",
                         seconds_per_operation=seconds, metric=metric, value=actual,
                         max_abs_error=error,
                         oracle="independent NumPy AMSGrad recurrence and MSE oracle",
                         notes="bias-corrected elementwise max-second-moment state"))
    for name in names:
        if name not in seen:
            rows.append(base(details, workload=name, phase="train", backend="fortml", status="unavailable",
                             oracle="FortML release-app protocol", notes="release app emitted no parseable row"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/amsgrad.csv"))
    parser.add_argument("--target", default="fortml_bench_amsgrad_training")
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
    rows = [base(details, workload="amsgrad_training", phase="train", backend="numpy_oracle", status="pass",
                 seconds_per_operation=norm_seconds, metric="parameter_l2_norm", value=norm,
                 max_abs_error=0.0, oracle="independent AMSGrad recurrence",
                 notes="beta1=0.9; beta2=0.99; epsilon=1e-8"),
            base(details, workload="amsgrad_mlp", phase="train", backend="numpy_oracle", status="pass",
                 metric="final_loss", value=loss, max_abs_error=0.0,
                 oracle="independent AMSGrad MLP recurrence",
                 notes="beta1=0.85; beta2=0.95; epsilon=1e-5")]
    if args.skip_fortml:
        rows.extend(base(details, workload=name, phase="train", backend="fortml", status="skipped",
                         oracle="FortML release-app protocol", notes="--skip-fortml")
                    for name in ("amsgrad_training", "amsgrad_mlp"))
    else:
        rows.extend(run_fortml(fortml, args.target, details, norm, loss))
    for name in ("amsgrad_training", "amsgrad_mlp"):
        rows.append(base(details, workload=name, phase="train", backend="fortml", device="cuda",
                         status="unavailable", seconds_per_operation="", metric="", value="",
                         max_abs_error="", oracle="FortML device contract",
                         notes="resident AMSGrad state kernel is not linked; no host fallback"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
