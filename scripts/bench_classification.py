#!/usr/bin/env python3
"""Benchmark the binary logistic slice against an independent NumPy oracle."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
import time
from pathlib import Path

import numpy as np


N_SAMPLES = 1024
N_FEATURES = 8
L2 = 1.0e-3


def inputs() -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(1, N_SAMPLES + 1, dtype=np.float64)[:, None]
    columns = np.arange(1, N_FEATURES + 1, dtype=np.float64)[None, :]
    x = np.sin(0.013 * rows + 0.071 * columns)
    x += 0.2 * np.cos(0.009 * rows * columns)
    true = -0.15 + x @ np.sin(0.17 * np.arange(1, N_FEATURES + 1))
    labels = np.where(true >= 0.0, 7, -3).astype(np.int64)
    return x, labels


def git_revision(repository: Path) -> str:
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).strip()
    return revision + ("+dirty" if dirty else "")


def parse_fortran(stdout: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in stdout.splitlines():
        match = re.match(
            r"^(logistic_fit|logistic_predict|logistic_accuracy),([^,]+),([^,]+),([^,]+),(.+)$",
            line,
        )
        if match:
            values[match.group(1)] = float(match.group(5))
        match = re.match(r"^logistic_accuracy,(.*)$", line)
        if match:
            values["logistic_accuracy"] = float(match.group(1))
    return values


def run_fortran(fortml: Path) -> list[dict[str, object]]:
    env = os.environ.copy()
    env.update({"FO_FC": env.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=env, check=True)
    started = time.perf_counter()
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_logistic"],
        cwd=fortml,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - started
    values = parse_fortran(completed.stdout)
    rows = []
    for phase, key in (("fit", "logistic_fit"), ("predict", "logistic_predict")):
        rows.append(
            {
                "backend": "fortml",
                "phase": phase,
                "status": "pass" if key in values else "parse_failed",
                "n_samples": N_SAMPLES,
                "n_features": N_FEATURES,
                "seconds_per_operation": values.get(key, float("nan")),
                "accuracy": values.get("logistic_accuracy", float("nan")),
                "oracle": "FortML internal invariant plus NumPy labels",
                "notes": f"fo wall time including process startup={wall:.6e}s",
            }
        )
    return rows


def run_sklearn(x: np.ndarray, labels: np.ndarray) -> list[dict[str, object]]:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return [{"backend": "sklearn", "phase": "fit", "status": "unavailable"}]
    started = time.perf_counter()
    model = LogisticRegression(
        C=1.0 / L2, fit_intercept=True, solver="lbfgs", max_iter=500
    )
    model.fit(x, labels)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predicted = model.predict(x)
    probabilities = model.predict_proba(x)
    predict_seconds = time.perf_counter() - started
    accuracy = float(np.mean(predicted == labels))
    return [
        {
            "backend": "sklearn",
            "phase": "fit",
            "status": "pass",
            "n_samples": N_SAMPLES,
            "n_features": N_FEATURES,
            "seconds_per_operation": fit_seconds,
            "accuracy": accuracy,
            "oracle": "NumPy generated labels",
            "notes": f"probability normalization error={np.max(np.abs(probabilities.sum(axis=1) - 1.0)):.3e}",
        },
        {
            "backend": "sklearn",
            "phase": "predict",
            "status": "pass",
            "n_samples": N_SAMPLES,
            "n_features": N_FEATURES,
            "seconds_per_operation": predict_seconds,
            "accuracy": accuracy,
            "oracle": "NumPy generated labels",
            "notes": "single-process CPU reference",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/classification_workloads.csv")
    )
    args = parser.parse_args()
    x, labels = inputs()
    rows = run_sklearn(x, labels)
    rows.extend(run_fortran(args.fortml.resolve()))
    metadata = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "benchmark_revision": git_revision(Path.cwd()),
        "fortml_revision": git_revision(args.fortml.resolve()),
    }
    try:
        import sklearn

        metadata["sklearn_version"] = sklearn.__version__
    except ImportError:
        metadata["sklearn_version"] = "unavailable"
    for row in rows:
        row.update(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "backend",
        "phase",
        "status",
        "n_samples",
        "n_features",
        "seconds_per_operation",
        "accuracy",
        "oracle",
        "notes",
        "python_version",
        "numpy_version",
        "sklearn_version",
        "fortml_revision",
        "benchmark_revision",
    ]
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
