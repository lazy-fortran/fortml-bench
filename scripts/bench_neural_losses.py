#!/usr/bin/env python3
"""Correctness-gated benchmark for FortML's differentiable neural losses.

The NumPy formulas are independent value/curvature oracles.  The release app
reports checksums for the same fixed fixture before its CPU timings are kept;
CUDA is recorded as an explicit capability refusal until resident loss kernels
exist.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import subprocess
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "dimensions",
    "repetitions", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
N = 64
REPETITIONS = 2048
EPS = 3.0e-12


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    ignored_names = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, ...]:
    indices = np.arange(1, N + 1, dtype=np.float64)
    logits = np.column_stack((np.sin(0.13 * indices), np.cos(0.07 * indices),
                               0.2 * np.sin(0.19 * indices)))
    targets = np.column_stack((np.where((indices.astype(int) % 3) == 0, 1.0, 0.0),
                               np.where((indices.astype(int) % 3) == 1, 1.0, 0.0),
                               np.where((indices.astype(int) % 3) == 2, 1.0, 0.0)))
    direction = np.column_stack((0.01 * np.sin(0.11 * indices),
                                 -0.02 * np.cos(0.17 * indices),
                                 np.full(N, 0.03)))
    prediction = logits[:, :1]
    target = 0.4 * np.sin(0.05 * indices)[:, None]
    weights = 0.5 + (indices.astype(int) % 7) / 7.0
    return logits, targets, direction, prediction, target, weights


def oracle() -> dict[str, float]:
    logits, targets, direction, prediction, target, weights = fixture()
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    bce = probabilities * (1.0 - probabilities) * direction / logits.size
    exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
    softmax = exponent / exponent.sum(axis=1, keepdims=True)
    softmax_hvp = softmax * (direction -
                             (softmax * direction).sum(axis=1, keepdims=True)) / N
    weighted = weights[:, None] * direction[:, :1] / weights.sum()
    residual = prediction - target
    huber = np.where(np.abs(residual) < 0.75, direction[:, :1], 0.0) / N
    return {
        "bce_hvp": float(bce.sum()),
        "softmax_cross_entropy_hvp": float(softmax_hvp.sum()),
        "weighted_mse_hvp": float(weighted.sum()),
        "huber_hvp": float(huber.sum()),
    }


def row(details: dict[str, str], **updates: object) -> dict[str, str]:
    output = {field: "" for field in FIELDS}
    output.update(details)
    output.update({key: str(value) for key, value in updates.items()})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/neural_losses.csv"))
    parser.add_argument("--target", default="fortml_bench_neural_losses")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    expected = oracle()
    completed = subprocess.run(
        ["fo", "exec", "--no-build", args.target], cwd=fortml,
        check=True, capture_output=True, text=True,
    )
    actual: dict[str, tuple[float, float]] = {}
    for line in completed.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 3 and fields[0] in expected:
            actual[fields[0]] = (float(fields[1]), float(fields[2]))
        elif len(fields) == 3 and fields[0] == "mlp_weighted_objective":
            actual[fields[0]] = (float(fields[1]), float(fields[2]))
    required = set(expected) | {"mlp_weighted_objective"}
    if set(actual) != required:
        raise RuntimeError(f"missing neural-loss release rows: {sorted(actual)}")
    errors = {name: abs(actual[name][1] - expected[name]) for name in expected}
    if max(errors.values()) > EPS:
        raise RuntimeError(f"neural-loss checksum mismatch: {max(errors.values()):.3e}")
    if not math.isfinite(actual["mlp_weighted_objective"][1]):
        raise RuntimeError("MLP weighted objective checksum is not finite")

    details = {
        "workload": "neural_losses", "backend": "fortml", "device": "cpu",
        "status": "pass", "dimensions": "64x3; weighted MLP 64x1",
        "repetitions": str(REPETITIONS),
        "oracle": "independent NumPy loss-curvature formulas",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (args.output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
        "notes": "shared value/JVP/VJP/HVP facade; MLP uses weighted-MSE products",
    }
    rows = []
    for phase, (seconds, checksum) in actual.items():
        rows.append(row(details, phase=phase,
                        seconds_per_operation=f"{seconds:.17e}",
                        metric="seconds_per_operation", value=f"{checksum:.17e}",
                        max_abs_error=(f"{errors[phase]:.17e}" if phase in errors else "")))
    rows.append(row(details, phase="device_capability", device="cuda",
                    status="unavailable", repetitions="", seconds_per_operation="",
                    metric="", value="", max_abs_error="",
                    oracle="typed_device_contract",
                    notes="loss and MLP objective CUDA kernels are not resident; no host fallback"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
