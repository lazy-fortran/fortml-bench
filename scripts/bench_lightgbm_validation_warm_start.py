#!/usr/bin/env python3
"""Correctness-gated validation-aware LightGBM warm-start benchmark.

The NumPy oracle independently replays the weighted validation-loss sequence
for a one-feature leaf-wise Newton booster.  The release app exercises both
restore-best and retain-all continuation policies, malformed validation
transactionality, and the typed CUDA boundary.
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_estimators", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
FORTNUM_DOMAIN_ERROR = 1
FORTNUM_NOT_IMPLEMENTED = 3
LEARNING_RATE = 1.0
L2 = 1.0
PATIENCE = 2


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names: set[str] = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            continue
    dirty = any(
        line[3:].split(" -> ")[-1].strip() not in ignored_names
        for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True,
        ).splitlines()
    )
    return head + ("+dirty" if dirty else "")


def oracle() -> tuple[np.ndarray, int, int]:
    """Return weighted validation stage losses, best round, and stop round."""
    x = np.arange(8, dtype=np.float64)
    target = np.asarray((0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0))
    validation = 10.0 - target
    validation_weight = np.asarray((1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0))
    margin = np.full_like(target, float(np.mean(target)))
    losses: list[float] = []
    for _ in range(4):
        gradient = margin - target
        total_g = float(np.sum(gradient))
        total_h = float(target.size)
        best_gain = 0.0
        correction = np.full_like(target, -total_g / (total_h + L2))
        for split in range(1, x.size):
            left = x < float(split) - 0.5
            right = ~left
            left_g = float(np.sum(gradient[left]))
            right_g = float(np.sum(gradient[right]))
            left_h = float(np.sum(left))
            right_h = float(np.sum(right))
            gain = 0.5 * (
                left_g * left_g / (left_h + L2)
                + right_g * right_g / (right_h + L2)
                - total_g * total_g / (total_h + L2)
            )
            if gain > best_gain:
                best_gain = gain
                correction = np.where(
                    left, -left_g / (left_h + L2), -right_g / (right_h + L2),
                )
        margin = margin + LEARNING_RATE * correction
        losses.append(float(
            0.5 * np.sum(validation_weight * (margin - validation) ** 2)
            / np.sum(validation_weight)
        ))
    values = np.asarray(losses)
    best_iteration = int(np.argmin(values)) + 1
    best_loss = float(values[best_iteration - 1])
    stale = 0
    stop_iteration = values.size
    running_best = np.inf
    for index, value in enumerate(values, start=1):
        if value < running_best:
            running_best = value
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            stop_iteration = index
            break
    return np.asarray((best_loss,)), best_iteration, stop_iteration


def run_app(fortml: Path, target: str) -> dict[str, list[str]]:
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, check=True)
    completed = subprocess.run(
        ["fo", "exec", "--no-build", target], cwd=fortml,
        check=True, capture_output=True, text=True,
    )
    parsed: dict[str, list[str]] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            raise RuntimeError(f"malformed release row: {line!r}")
        parsed[fields[0]] = fields[1:]
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/lightgbm_validation_warm_start.csv"))
    parser.add_argument("--target", default="fortml_bench_lightgbm_validation_warm_start")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    observed = run_app(fortml, args.target)
    expected_loss, best_iteration, stop_iteration = oracle()
    expected_loss = float(expected_loss[0])
    restore = observed["lgbm_warm_restore"]
    retain = observed["lgbm_warm_retain"]
    invalid = observed["lgbm_warm_invalid"]
    cuda = observed["lgbm_warm_cuda"]
    if (int(restore[0]), int(restore[1]), restore[2].lower() in {"t", "true"}, float(restore[3])) != (
            best_iteration, best_iteration, True, expected_loss):
        raise RuntimeError(f"restore-best mismatch: {restore!r}, oracle={best_iteration, expected_loss}")
    if (int(retain[0]), int(retain[1]), retain[2].lower() in {"t", "true"}, float(retain[3])) != (
            best_iteration, stop_iteration, True, expected_loss):
        raise RuntimeError(f"retain-all mismatch: {retain!r}, oracle={best_iteration, stop_iteration, expected_loss}")
    if int(invalid[0]) != FORTNUM_DOMAIN_ERROR or int(invalid[1]) != best_iteration or float(invalid[2]) > 2e-12:
        raise RuntimeError(f"transactional refusal mismatch: {invalid!r}")
    if int(cuda[0]) != FORTNUM_NOT_IMPLEMENTED:
        raise RuntimeError(f"typed CUDA boundary mismatch: {cuda!r}")

    source_revision = revision(fortml)
    bench_revision = revision(root, (args.output.resolve(),))
    oracle_name = "independent NumPy weighted validation Newton oracle"
    common = dict(
        workload="lightgbm_validation_warm_start", backend="fortml",
        device="cpu", status="pass", n_samples=8, n_features=1,
        n_estimators=4, oracle=oracle_name,
        python_version=platform.python_version(), numpy_version=np.__version__,
        fortml_revision=source_revision, benchmark_revision=bench_revision,
        compiler="gfortran", flags="-O3",
    )
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(common)
        row.update(values)
        rows.append(row)

    add(phase="restore_best", seconds_per_operation=float(restore[4]),
        metric="best_validation_loss", value=float(restore[3]),
        max_abs_error=abs(float(restore[3]) - expected_loss),
        notes=f"best={best_iteration}, retained={best_iteration}, patience={PATIENCE}")
    add(phase="retain_all", seconds_per_operation=float(retain[4]),
        metric="best_validation_loss", value=float(retain[3]),
        max_abs_error=abs(float(retain[3]) - expected_loss),
        notes=f"best={best_iteration}, retained={stop_iteration}, patience={PATIENCE}")
    add(phase="transactional_refusal", seconds_per_operation=0.0,
        metric="status_code", value=int(invalid[0]), max_abs_error=float(invalid[2]),
        notes="malformed validation target shape leaves prefix prediction unchanged")
    add(phase="predict", backend="fortml", device="cuda", status="unavailable",
        seconds_per_operation=0.0, metric="status_code", value=int(cuda[0]),
        max_abs_error="nan", notes="FORTNUM_NOT_IMPLEMENTED; no resident CUDA histogram kernel")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
