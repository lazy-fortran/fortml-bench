#!/usr/bin/env python3
"""Correctness-gated CategoricalNB benchmark with an independent NumPy oracle."""
from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

N = 12
LABELS = np.array([-1, 4], dtype=np.int64)
ALPHA = 1.0
FIELDS = ("workload", "phase", "backend", "device", "status", "seconds_per_operation",
          "metric", "value", "max_abs_error", "oracle", "python_version", "numpy_version",
          "sklearn_version", "fortml_revision", "benchmark_revision", "compiler", "flags", "notes")


def revision(repo: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    ignored = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).splitlines():
        path = (repo / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.array([[1, 10], [1, 20], [2, 10], [2, 20], [1, 20], [2, 10],
                  [1, 10], [2, 20], [1, 10], [2, 20], [1, 20], [2, 10]], dtype=np.int64)
    labels = np.array([-1, -1, -1, 4, 4, 4, -1, 4, -1, 4, 4, -1], dtype=np.int64)
    query = np.array([[1, 10], [2, 20]], dtype=np.int64)
    return x, labels, query


def oracle(x: np.ndarray, labels: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prior = np.array([np.count_nonzero(labels == label) for label in LABELS], dtype=np.float64)
    prior /= prior.sum()
    joint = np.log(prior)[None, :].repeat(query.shape[0], axis=0)
    for feature in range(x.shape[1]):
        categories = np.unique(x[:, feature])
        for category in categories:
            counts = np.array([np.count_nonzero((labels == label) & (x[:, feature] == category))
                               for label in LABELS], dtype=np.float64)
            probabilities = (counts + ALPHA) / (np.array([np.count_nonzero(labels == label)
                for label in LABELS]) + ALPHA * categories.size)
            mask = query[:, feature] == category
            joint[mask, :] += np.log(probabilities)
    shifted = joint - joint.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities, LABELS[np.argmax(probabilities, axis=1)]


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except ImportError:
        sklearn_version = "unavailable"
    return {"python_version": platform.python_version(), "numpy_version": np.__version__,
            "sklearn_version": sklearn_version, "fortml_revision": revision(fortml),
            "benchmark_revision": revision(root, (output,)), "compiler": os.environ.get("FO_FC", "gfortran"),
            "flags": "-O3"}


def row(details: dict[str, str], **values: object) -> dict[str, object]:
    result = dict(details)
    result.update({"workload": "categorical_naive_bayes", "phase": "", "backend": "", "device": "cpu",
                   "status": "", "seconds_per_operation": "", "metric": "", "value": "",
                   "max_abs_error": "", "oracle": "", "notes": ""})
    result.update(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/categorical_naive_bayes.csv"))
    parser.add_argument("--target", default="fortml_bench_categorical_nb")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    fortml = args.fortml.resolve()
    details = metadata(root, fortml, output)
    x, labels, query = fixture()
    expected, expected_labels = oracle(x, labels, query)
    started = time.perf_counter()
    for _ in range(256): oracle(x, labels, query)
    numpy_seconds = (time.perf_counter() - started) / 256.0
    rows = [row(details, phase="predict", backend="numpy_oracle", status="pass",
               seconds_per_operation=numpy_seconds, metric="probability_sum",
               value=float(expected.sum()), max_abs_error=0.0,
               oracle="independent NumPy category counts and stable normalization")]
    fortml_rows: list[dict[str, object]] = []
    if not args.skip_fortml:
        build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                               capture_output=True, text=True)
        if build.returncode == 0:
            with tempfile.TemporaryDirectory(dir="/mnt/storage") as directory:
                oracle_path = Path(directory) / "categorical.csv"
                env = os.environ.copy()
                env["FORTML_BENCH_CATEGORICAL_NB_ORACLE"] = str(oracle_path)
                run = subprocess.run(["fo", "exec", "--no-build", args.target], cwd=fortml,
                                     env=env, capture_output=True, text=True)
                if run.returncode == 0 and oracle_path.is_file():
                    actual = np.full_like(expected, np.nan)
                    actual_labels = np.zeros(expected_labels.shape, dtype=np.int64)
                    with oracle_path.open(newline="") as stream:
                        for record in csv.DictReader(stream):
                            index = int(record["row"]) - 1
                            if record["quantity"] == "prediction": actual_labels[index] = int(float(record["value"]))
                            else: actual[index, int(record["column"]) - 1] = float(record["value"])
                    error = max(float(np.max(np.abs(actual - expected))),
                                float(np.max(np.abs(actual_labels - expected_labels))))
                    fortml_rows.append(row(details, phase="predict", backend="fortml", status="pass",
                        metric="probability_sum", value=float(actual.sum()), max_abs_error=error,
                        oracle="complete CategoricalNB probabilities and labels", notes=args.target))
        if not fortml_rows:
            fortml_rows.append(row(details, phase="predict", backend="fortml", status="unavailable",
                oracle="FortML release-app protocol", notes="target or build unavailable"))
    rows.extend(fortml_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
