#!/usr/bin/env python3
"""Correctness-gated random-forest OOB decision/score benchmark."""

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
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_trees", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
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
    return head + ("+dirty" if dirty else "")


def parse(stdout: str) -> dict[str, float | int | str]:
    records: dict[str, float | int | str] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields[:2] == ["rf_oob", "metrics"]:
            if len(fields) != 12:
                raise RuntimeError(f"malformed random-forest OOB record: {line!r}")
            names = (
                "n_samples", "n_features", "n_trees", "fit_seconds", "oob_seconds",
                "score", "coverage", "simplex_error", "oracle_correct", "min_oob_count",
            )
            for name, value in zip(names, fields[2:], strict=True):
                records[name] = float(value) if name.endswith("seconds") or name in {
                    "score", "coverage", "simplex_error",
                } else int(value)
        elif fields[:2] == ["rf_oob", "cuda"]:
            records["cuda"] = fields[2]
    required = {
        "n_samples", "n_features", "n_trees", "fit_seconds", "oob_seconds", "score",
        "coverage", "simplex_error", "oracle_correct", "min_oob_count", "cuda",
    }
    missing = required.difference(records)
    if missing:
        raise RuntimeError(f"release app omitted OOB metrics: {sorted(missing)}")
    return records


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/random_forest_oob.csv"))
    parser.add_argument("--target", default="fortml_bench_random_forest_oob")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(["fo", "build", "--flag", "-O2"], cwd=fortml,
                   env=environment, check=True, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE, text=True)
    completed = subprocess.run(
        ["fo", "exec", "--no-build", args.target], cwd=fortml,
        env=environment, check=True, capture_output=True, text=True,
    )
    observed = parse(completed.stdout)
    n_samples = int(observed["n_samples"])
    n_features = int(observed["n_features"])
    n_trees = int(observed["n_trees"])
    # Independent NumPy threshold oracle for the release fixture.
    x1 = -2.0 + 4.0*np.mod(np.arange(n_samples), 80)/79.0
    expected = np.where(x1 < -0.65, -3, np.where(x1 > 0.65, 11, 4))
    expected_correct = int(expected.size)
    if int(observed["oracle_correct"]) != expected_correct:
        raise RuntimeError(
            f"independent OOB threshold oracle mismatch: {observed['oracle_correct']} "
            f"!= {expected_correct}"
        )
    if float(observed["coverage"]) < 1.0 or int(observed["min_oob_count"]) < 1:
        raise RuntimeError("OOB coverage is incomplete; in-bag fallback may be hidden")
    if float(observed["simplex_error"]) > 5.0e-13:
        raise RuntimeError("OOB probabilities violate the simplex")
    if abs(float(observed["score"]) - 1.0) > 5.0e-13:
        raise RuntimeError("OOB score disagrees with the independent threshold oracle")
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O2",
        "oracle": "independent NumPy threshold-label, simplex, and coverage oracle",
    }
    rows = [
        row(details, workload="random_forest_oob", phase="fit", backend="fortml",
            device="cpu", status="pass", n_samples=n_samples, n_features=n_features,
            n_trees=n_trees, seconds_per_operation=observed["fit_seconds"],
            metric="oob_score", value=observed["score"], max_abs_error=0.0,
            notes="64-tree seeded bootstrap CART fixture"),
        row(details, workload="random_forest_oob", phase="oob_decision", backend="fortml",
            device="cpu", status="pass", n_samples=n_samples, n_features=n_features,
            n_trees=n_trees, seconds_per_operation=observed["oob_seconds"],
            metric="probability_simplex_max_abs_error", value=observed["simplex_error"],
            max_abs_error=observed["simplex_error"],
            notes="stored inclusion state; minimum OOB trees per row=" +
            str(observed["min_oob_count"])),
        row(details, workload="random_forest_oob", phase="device_contract", backend="fortml",
            device="cuda", status="unavailable", n_samples=n_samples,
            n_features=n_features, n_trees=n_trees, metric="api_surface",
            value=observed["cuda"], max_abs_error=0.0,
            oracle="typed CUDA refusal preserving output buffer",
            notes="no resident CUDA OOB tree kernel"),
    ]
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
