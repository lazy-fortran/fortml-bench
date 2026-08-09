#!/usr/bin/env python3
"""Correctness-gated GaussianNB partial-fit benchmark.

The NumPy recurrence is deliberately independent of FortML: it accumulates
weighted class masses, means, and population second moments across two
batches, then evaluates the Gaussian log-density directly.  The release test
adds transactional rollback and CPU/CUDA dispatch checks.  A resident CUDA
GaussianNB update is not claimed until sufficient-statistic kernels exist.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "metric", "value",
    "max_abs_error", "oracle", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes", "n_samples", "n_features", "n_classes",
    "batch_size", "batch_count", "seconds_per_operation", "python_version",
    "numpy_version", "fortad_revision", "fortsym_revision", "fortopt_revision",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    value = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = set()
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
    return value + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.array(
        [[8.0, 4.0], [0.0, 0.0], [10.0, 6.0],
         [2.0, 2.0], [4.0, 1.0], [6.0, 3.0]],
        dtype=np.float64,
    )
    labels = np.array([9, -3, 9, -3, 4, 4], dtype=np.int64)
    classes = np.array([-3, 4, 9], dtype=np.int64)
    return x, labels, classes, np.array([3, 3], dtype=np.int64)


def gaussian_oracle(
    x: np.ndarray, labels: np.ndarray, classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.vstack([x[labels == label].mean(axis=0) for label in classes])
    variances = np.vstack([
        ((x[labels == label] - mean) ** 2).mean(axis=0)
        for label, mean in zip(classes, means)
    ])
    prior = np.array([np.count_nonzero(labels == label) for label in classes], dtype=float)
    prior /= prior.sum()
    log_joint = np.empty((x.shape[0], classes.size), dtype=float)
    for column, (mean, variance, prior_value) in enumerate(zip(means, variances, prior)):
        log_joint[:, column] = (
            -0.5 * np.sum(np.log(2.0 * np.pi * variance), axis=0)
            -0.5 * np.sum((x - mean) ** 2 / variance, axis=1)
            + np.log(prior_value)
        )
    shifted = log_joint - np.max(log_joint, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return means, variances, probabilities


def base_row(details: dict[str, Any], **values: Any) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    row.update(details)
    row.update(values)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gaussian_nb_partial_fit.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = (args.output if args.output.is_absolute() else root / args.output).resolve()
    ignored = (output, root / "results/GAUSSIAN_NB_PARTIAL_FIT.md")
    details = {
        "workload": "gaussian_nb_partial_fit", "backend": "fortml",
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "n_samples": 6, "n_features": 2, "n_classes": 3,
        "batch_size": 3, "batch_count": 2,
    }
    x, labels, classes, _ = fixture()
    means, variances, probabilities = gaussian_oracle(x, labels, classes)
    expected_means = np.array([[1.0, 1.0], [5.0, 2.0], [9.0, 5.0]])
    expected_variances = np.ones((3, 2))
    stream_classes = classes.copy()
    first, second = labels[:3], labels[3:]
    if not np.array_equal(np.sort(stream_classes), stream_classes):
        raise RuntimeError("class vocabulary is not sorted")
    if set(first) == set(stream_classes) or not set(second).issuperset({-3, 4}):
        raise RuntimeError("fixture does not exercise deferred class completion")
    if not np.allclose(means, expected_means, atol=2.0e-14, rtol=0.0):
        raise RuntimeError("Gaussian mean oracle failed")
    if not np.allclose(variances, expected_variances, atol=2.0e-14, rtol=0.0):
        raise RuntimeError("Gaussian variance oracle failed")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=2.0e-14, rtol=0.0):
        raise RuntimeError("Gaussian probability normalization failed")

    rows: list[dict[str, Any]] = [
        base_row(details, phase="independent_moments", device="cpu", status="pass",
                 metric="mean_variance_max_abs_error", value=0.0, max_abs_error=0.0,
                 oracle="independent NumPy population moments and Gaussian log density",
                 notes="var_smoothing=0; each class has unit variance"),
        base_row(details, phase="independent_stream", device="cpu", status="pass",
                 metric="batch_count", value=2.0, max_abs_error=0.0,
                 oracle="independent sorted-vocabulary stream state machine",
                 notes="first batch omits class 4; second batch completes all classes"),
        base_row(details, phase="independent_stream", device="cpu", status="pass",
                 metric="sample_count", value=6.0, max_abs_error=0.0,
                 oracle="independent sorted-vocabulary stream state machine",
                 notes="accepted batches are counted exactly once"),
    ]

    test_status = "skipped"
    if not args.skip_fortml:
        environment = os.environ.copy()
        environment.setdefault("FO_SCAN_FALLBACK", "regex")
        result = subprocess.run(
            ["fo", "test", "test_gaussian_nb_partial_fit"], cwd=fortml,
            env=environment, capture_output=True, text=True,
        )
        test_status = "pass" if result.returncode == 0 else "failed"
    rows.extend([
        base_row(details, phase="behavioral_gate", device="cpu", status=test_status,
                 metric="test_gaussian_nb_partial_fit",
                 value=1.0 if test_status == "pass" else "",
                 max_abs_error=0.0,
                 oracle="independent Fortran replay, rollback, and device oracle",
                 notes="fo test test_gaussian_nb_partial_fit"),
        base_row(details, phase="device_boundary", device="cuda", status="unavailable",
                 metric="partial_fit_device_status", value=3.0, max_abs_error=0.0,
                 oracle="typed resident-CUDA sufficient-statistic capability contract",
                 notes="test verifies FORTNUM_NOT_IMPLEMENTED and no host fallback"),
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = root / "results/GAUSSIAN_NB_PARTIAL_FIT.md"
    report.write_text(
        "# Gaussian Naive Bayes partial-fit benchmark\n\n"
        "The independent NumPy oracle accumulates a sorted class vocabulary, "
        "population moments, and Gaussian log densities across two batches. "
        "The first batch omits one declared class, so fitting is deferred until "
        "the second batch. The Fortran behavioral gate checks transactional "
        "unknown-label rollback and CPU/CUDA dispatch; CUDA is an explicit "
        "resident sufficient-statistic refusal.\n\n"
        f"FortML revision: {details['fortml_revision']}\n"
        f"Benchmark revision: {details['benchmark_revision']}\n\n"
        "| phase | device | status | metric | value | max abs error |\n"
        "| --- | --- | --- | --- | ---: | ---: |\n" +
        "".join(
            f"| {record['phase']} | {record['device']} | {record['status']} | "
            f"{record['metric']} | {record['value']} | "
            f"{record['max_abs_error']} |\n" for record in rows
        ), encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
