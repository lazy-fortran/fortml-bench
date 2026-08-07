#!/usr/bin/env python3
"""Correctness-gated Extra-Trees classifier benchmark.

The oracle is a direct three-region rule over the public fixture; it does not
reimplement FortML's randomized tree builder or import scikit-learn.  The
benchmark therefore checks behavior and the explicit CUDA refusal separately
from timing.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

N_SAMPLES, N_FEATURES, N_QUERY, N_CLASSES = 240, 3, 6, 3
EXPECTED_QUERY = np.array([-3, -3, 4, 4, 11, 11], dtype=np.int64)
FIELDS = (
    "workload", "model", "phase", "backend", "device", "status",
    "n_samples", "n_features", "n_query", "seconds_per_operation",
    "accuracy", "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names: set[str] = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            pass
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        name = line[3:].split(" -> ")[-1].strip()
        if name not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray]:
    x = np.empty((N_SAMPLES, N_FEATURES), dtype=np.float64)
    index = np.arange(1, N_SAMPLES + 1, dtype=np.float64)
    x[:, 0] = -2.0 + 4.0 * np.mod(index - 1.0, 80.0) / 79.0
    x[:, 1] = np.sin(0.17 * index)
    x[:, 2] = np.cos(0.11 * index)
    query = np.column_stack((
        np.array([-1.5, -0.7, -0.1, 0.1, 0.7, 1.5]),
        np.zeros(N_QUERY), np.ones(N_QUERY),
    ))
    return x, query


def oracle_predict(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.where(x[:, 0] < -0.65, -3,
                      np.where(x[:, 0] > 0.65, 11, 4)).astype(np.int64)
    probabilities = np.zeros((x.shape[0], N_CLASSES), dtype=np.float64)
    for row, label in enumerate(labels):
        probabilities[row, {-3: 0, 4: 1, 11: 2}[int(label)]] = 1.0
    return probabilities, labels


def parse_app(stdout: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0].startswith("extra_trees_"):
            if fields[0] in values:
                raise RuntimeError(f"duplicate benchmark field: {fields[0]}")
            values[fields[0]] = fields[1]
    required = {
        "extra_trees_fit_seconds", "extra_trees_predict_seconds",
        "extra_trees_probability_sum_error", "extra_trees_query_correct",
        "extra_trees_cuda",
    }
    missing = required.difference(values)
    if missing:
        raise RuntimeError(f"FortML benchmark omitted fields: {sorted(missing)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/extra_trees.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output_path = args.output if args.output.is_absolute() else root / args.output
    ignored_outputs = (output_path.resolve(),)
    x, query = fixture()
    expected_probabilities, expected_labels = oracle_predict(query)
    if not np.array_equal(expected_labels, EXPECTED_QUERY):
        raise RuntimeError("internal Extra-Trees fixture oracle changed unexpectedly")
    metadata = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored_outputs),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }

    def row(**values: object) -> dict[str, object]:
        result: dict[str, object] = {
            "workload": "extra_trees", "model": "extra_trees", "phase": "",
            "backend": "", "device": "cpu", "status": "",
            "n_samples": N_SAMPLES, "n_features": N_FEATURES, "n_query": N_QUERY,
            "seconds_per_operation": "", "accuracy": "", "max_abs_error": "",
            "oracle": "", "notes": "", **metadata,
        }
        result.update(values)
        return result

    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for _ in range(128):
        oracle_predict(query)
    oracle_seconds = (time.perf_counter() - started) / 128.0
    oracle_probabilities, oracle_labels = oracle_predict(query)
    rows.append(row(phase="predict", backend="numpy_oracle", status="pass",
                    seconds_per_operation=oracle_seconds, accuracy=1.0,
                    max_abs_error=float(np.max(np.abs(oracle_probabilities - expected_probabilities))),
                    oracle="independent NumPy direct threshold classification oracle",
                    notes="piecewise class rule; no sklearn dependency"))

    environment = os.environ.copy()
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                   env=environment, check=True)
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_extra_trees"],
        cwd=fortml, env=environment, capture_output=True, text=True, check=True,
    )
    values = parse_app(completed.stdout)
    fit_seconds = float(values["extra_trees_fit_seconds"])
    predict_seconds = float(values["extra_trees_predict_seconds"])
    probability_error = float(values["extra_trees_probability_sum_error"])
    query_correct = int(values["extra_trees_query_correct"])
    if not np.isfinite(fit_seconds) or not np.isfinite(predict_seconds):
        raise RuntimeError("FortML emitted non-finite timing")
    if fit_seconds < 0.0 or predict_seconds < 0.0:
        raise RuntimeError("FortML emitted negative timing")
    if probability_error > 2.0e-12:
        raise RuntimeError(f"probability simplex error too large: {probability_error:.3e}")
    if query_correct != N_QUERY:
        raise RuntimeError(f"FortML query oracle mismatch: {query_correct}/{N_QUERY}")
    if values["extra_trees_cuda"] != "unavailable":
        raise RuntimeError(f"unexpected Extra-Trees CUDA status: {values['extra_trees_cuda']}")
    notes = f"{query_correct}/{N_QUERY} direct-rule labels; 32 trees, depth=6, seed=1729"
    rows.extend([
        row(phase="fit", backend="fortml_cpu", status="pass",
            seconds_per_operation=fit_seconds, accuracy=1.0,
            max_abs_error=probability_error,
            oracle="independent NumPy direct threshold classification oracle", notes=notes),
        row(phase="predict", backend="fortml_cpu", status="pass",
            seconds_per_operation=predict_seconds, accuracy=1.0,
            max_abs_error=probability_error,
            oracle="independent NumPy direct threshold classification oracle", notes=notes),
        row(phase="predict", backend="fortml_cuda", device="cuda", status="unavailable",
            oracle="typed_device_contract",
            notes="no resident CUDA Extra-Trees kernel; FORTNUM_NOT_IMPLEMENTED"),
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
