#!/usr/bin/env python3
"""Correctness-gated weighted reliability-curve benchmark."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


N_SAMPLES = 512
N_CLASSES = 3
BINS = 10
REPETITIONS = 128
FIELDS = (
    "workload", "method", "phase", "backend", "device", "status",
    "n_samples", "n_classes", "bins", "repetitions",
    "seconds_per_operation", "max_abs_error", "oracle",
    "python_version", "numpy_version", "fortml_revision",
    "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    value = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return value + ("+dirty" if dirty else "")


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = np.arange(1, N_SAMPLES + 1, dtype=np.float64)
    raw = np.column_stack((
        0.2 + np.abs(np.sin(0.017 * index)),
        0.3 + np.abs(np.cos(0.013 * index + 0.2)),
        0.4 + np.abs(np.sin(0.011 * index + 0.7)),
    ))
    probabilities = raw / raw.sum(axis=1, keepdims=True)
    selector = (7 * index.astype(np.int64) + 2) % 3
    labels = np.where(selector == 0, -4, np.where(selector == 1, 17, 91))
    weights = 0.5 + ((5 * index.astype(np.int64) + 1) % 7) / 3.0
    return probabilities, labels, weights


def oracle(probabilities: np.ndarray, labels: np.ndarray,
           weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    confidence = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    classes = np.array([-4, 17, 91])
    encoded = np.searchsorted(classes, labels)
    bins = np.minimum((confidence * BINS).astype(np.int64), BINS - 1)
    mass = np.bincount(bins, weights=weights, minlength=BINS)
    confidence_sum = np.bincount(
        bins, weights=weights * confidence, minlength=BINS,
    )
    correct_sum = np.bincount(
        bins, weights=weights * (predictions == encoded), minlength=BINS,
    )
    mean_confidence = np.divide(
        confidence_sum, mass, out=np.zeros(BINS), where=mass > 0,
    )
    mean_accuracy = np.divide(
        correct_sum, mass, out=np.zeros(BINS), where=mass > 0,
    )
    return mean_confidence, mean_accuracy, mass


def row(details: dict[str, str], phase: str, backend: str, status: str,
        **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {field: "" for field in FIELDS}
    result.update(details)
    result.update({
        "workload": "reliability_diagram", "method": "weighted_curve",
        "phase": phase, "backend": backend, "device": "cpu",
        "status": status, "n_samples": N_SAMPLES, "n_classes": N_CLASSES,
        "bins": BINS, "oracle": "", "notes": "",
    })
    result.update(values)
    return result


def read_fortran(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    confidence = np.full(BINS, np.nan)
    accuracy = np.full(BINS, np.nan)
    mass = np.full(BINS, np.nan)
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            index = int(record["bin"]) - 1
            confidence[index] = float(record["mean_confidence"])
            accuracy[index] = float(record["mean_accuracy"])
            mass[index] = float(record["bin_weight"])
    if np.isnan(confidence).any() or np.isnan(accuracy).any() or np.isnan(mass).any():
        raise RuntimeError("FortML omitted reliability-curve bins")
    return confidence, accuracy, mass


def run_numpy(details: dict[str, str]) -> list[dict[str, Any]]:
    probabilities, labels, weights = fixture()
    oracle(probabilities, labels, weights)
    return [row(
        details, "curve", "numpy_oracle", "pass",
        repetitions=0, seconds_per_operation="",
        max_abs_error=0.0,
        oracle="independent NumPy weighted equal-width reliability oracle",
        notes="zero-filled empty bins; first-maximum tie policy",
    )]


def run_fortml(fortml: Path, details: dict[str, str]) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"),
                        "OMP_NUM_THREADS": "1"})
    build = subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
        capture_output=True, text=True,
    )
    if build.returncode != 0:
        return [row(details, "curve", "fortml", "unavailable",
                    oracle="FortML release-app protocol",
                    notes="fo build failed")]
    probabilities, labels, weights = fixture()
    expected = oracle(probabilities, labels, weights)
    with tempfile.TemporaryDirectory(dir="/mnt/storage", prefix="fortml-reliability-") as directory:
        oracle_path = Path(directory) / "reliability.csv"
        check_environment = dict(environment)
        check_environment["FORTML_BENCH_RELIABILITY_ORACLE"] = str(oracle_path)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", "fortml_bench_reliability_diagram"],
            cwd=fortml, env=check_environment, capture_output=True, text=True,
        )
        if completed.returncode != 0 or not oracle_path.is_file():
            return [row(details, "curve", "fortml", "unavailable",
                        oracle="FortML release-app protocol",
                        notes="release app did not emit complete output")]
        actual = read_fortran(oracle_path)
    error = float(max(np.max(np.abs(actual[0] - expected[0])),
                      np.max(np.abs(actual[1] - expected[1])),
                      np.max(np.abs(actual[2] - expected[2]))))
    if error > 2.0e-13:
        raise RuntimeError(f"FortML reliability-curve mismatch {error:.3e}")
    timing = re.search(
        r"^reliability_diagram,weighted_curve,\s*(.+)$",
        completed.stdout, re.MULTILINE,
    )
    seconds = float(timing.group(1)) if timing else float("nan")
    return [row(
        details, "curve", "fortml", "pass" if timing else "parse_failed",
        repetitions=REPETITIONS, seconds_per_operation=seconds,
        max_abs_error=error,
        oracle="independent NumPy weighted equal-width reliability oracle",
        notes="all bin means and masses checked before timing",
    )]


def device_refusal(details: dict[str, str]) -> list[dict[str, Any]]:
    return [row(
        details, "device_capability", "fortml", "unavailable", device="cuda",
        oracle="FortML CUDA capability boundary; no resident metric kernel",
        notes="no host fallback",
    )]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/reliability_diagram.csv"))
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    details = metadata(root, arguments.fortml.resolve(), arguments.output)
    rows = run_numpy(details) + run_fortml(arguments.fortml.resolve(), details)
    rows += device_refusal(details)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: item.get(field, "") for field in FIELDS} for item in rows)


if __name__ == "__main__":
    main()
