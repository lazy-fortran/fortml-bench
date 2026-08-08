#!/usr/bin/env python3
"""Correctness-gated production Lion trainer benchmark.

The NumPy recurrence is independent of FortML and supplies the expected final
loss, EMA norm, and checkpoint trajectory. The Fortran release app contributes
timings and the same values. CUDA rows are explicit typed-unavailable records.
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_parameters",
    "steps", "n_samples", "epochs", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
EPOCHS = 64
N_SAMPLES = 8
RATE = 2.0e-3
BETA1 = 0.9
BETA2 = 0.99
WEIGHT_DECAY = 1.0e-3
EMA_DECAY = 0.9


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    value = subprocess.check_output(
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
    return value + ("+dirty" if dirty else "")


def lion_trajectory(epochs: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, N_SAMPLES, dtype=np.float64)
    target = 0.6 * x - 0.15
    theta = np.zeros(2, dtype=np.float64)
    momentum = np.zeros(2, dtype=np.float64)
    ema = theta.copy()
    for _ in range(epochs):
        for batch in (slice(0, 4), slice(4, 8)):
            prediction = theta[0] * x[batch] + theta[1]
            residual = prediction - target[batch]
            gradient = np.array(
                [np.mean(residual * x[batch]), np.mean(residual)], dtype=np.float64
            )
            interpolated = BETA1 * momentum + (1.0 - BETA1) * gradient
            update = np.sign(interpolated)
            theta -= RATE * (update + WEIGHT_DECAY * theta)
            momentum = BETA2 * momentum + (1.0 - BETA2) * gradient
            ema = EMA_DECAY * ema + (1.0 - EMA_DECAY) * theta
    return theta, ema


def expected() -> tuple[float, float, np.ndarray]:
    theta, ema = lion_trajectory(EPOCHS)
    x = np.linspace(-1.0, 1.0, N_SAMPLES, dtype=np.float64)
    target = 0.6 * x - 0.15
    loss = 0.5 * np.mean((theta[0] * x + theta[1] - target) ** 2)
    return float(loss), float(np.linalg.norm(ema)), theta


def row(details: dict[str, str], **values: object) -> dict[str, object]:
    result: dict[str, object] = {field: "" for field in FIELDS}
    result.update(details)
    result.update({
        "backend": "fortml", "device": "cpu", "status": "pass",
        "n_parameters": 2, "n_samples": N_SAMPLES, "epochs": EPOCHS,
        "compiler": "gfortran", "flags": "-O3",
    })
    result.update(values)
    return result


def run_fortml(fortml: Path, target: str, details: dict[str, str],
               loss: float, ema_norm: float) -> list[dict[str, object]]:
    started = time.perf_counter()
    build = subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml,
        capture_output=True, text=True, check=False,
    )
    if build.returncode:
        return [row(details, workload="lion_training", phase="train",
                    status="unavailable", oracle="FortML release app",
                    notes="fo build failed")]
    run = subprocess.run(
        ["fo", "exec", "--no-build", target], cwd=fortml,
        capture_output=True, text=True, check=False,
    )
    if run.returncode:
        return [row(details, workload="lion_training", phase="train",
                    status="unavailable", oracle="FortML release app",
                    notes="release app execution failed")]
    elapsed = time.perf_counter() - started
    rows: list[dict[str, object]] = []
    expected_by_name = {
        "lion_training": ("final_loss", loss),
        "lion_ema": ("ema_parameter_l2_norm", ema_norm),
        "lion_checkpoint_resume": ("resume_max_abs_error", 0.0),
    }
    for line in run.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 5 or fields[0] not in expected_by_name:
            continue
        name, status, metric = fields[:3]
        value = float(fields[3])
        seconds = float(fields[4])
        expected_metric, expected_value = expected_by_name[name]
        error = abs(value - expected_value)
        if status != "pass" or error > 2.0e-12:
            raise RuntimeError(f"Lion oracle mismatch for {name}: {error:.3e}")
        rows.append(row(
            details, workload=name, phase="train" if name == "lion_training"
            else "state", seconds_per_operation=seconds, metric=metric,
            value=value, max_abs_error=error,
            oracle="independent NumPy Lion recurrence",
            notes=f"release app wall time {elapsed:.6e} s",
        ))
    if len(rows) != len(expected_by_name):
        raise RuntimeError("FortML Lion release app emitted incomplete rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/lion_training.csv"))
    parser.add_argument("--target", default="fortml_bench_lion_training")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output, root / "scripts" / "__pycache__")),
        "compiler": "gfortran",
        "flags": "-O3",
    }
    loss, ema_norm, _ = expected()
    rows: list[dict[str, object]] = [
        row(details, backend="numpy_oracle", workload="lion_training",
            phase="train", metric="final_loss", value=loss, max_abs_error=0.0,
            seconds_per_operation="", oracle="independent NumPy Lion recurrence",
            notes="beta1=0.9; beta2=0.99; decoupled weight decay=1e-3"),
        row(details, backend="numpy_oracle", workload="lion_ema",
            phase="state", metric="ema_parameter_l2_norm", value=ema_norm,
            max_abs_error=0.0, seconds_per_operation="",
            oracle="independent NumPy Lion recurrence",
            notes="EMA decay=0.9; two four-sample updates per epoch"),
    ]
    if args.skip_fortml:
        rows.extend(row(details, workload=name, phase="state",
                        status="skipped", oracle="FortML release app",
                        notes="--skip-fortml")
                    for name in ("lion_training", "lion_ema",
                                 "lion_checkpoint_resume"))
    else:
        rows.extend(run_fortml(fortml, args.target, details, loss, ema_norm))
    for name in ("lion_training", "lion_ema", "lion_checkpoint_resume"):
        rows.append(row(details, workload=name, device="cuda", status="unavailable",
                        phase="device_boundary", n_parameters=2,
                        oracle="FortML device contract",
                        notes="resident Lion state is not linked; host fallback is not claimed"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
