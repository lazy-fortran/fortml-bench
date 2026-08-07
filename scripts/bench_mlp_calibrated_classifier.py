#!/usr/bin/env python3
"""Correctness-gated calibrated MLP classifier contract benchmark.

The release application owns the nonlinear optimizer trajectory, so the
independent oracle checks the fixture labels, sorted-class policy, probability
simplex, finite bounds, and prediction domain.  These are behavioral
properties rather than a reimplementation of the MLP optimizer.  CPU timing is
retained only after the complete oracle file passes; CUDA remains a typed
refusal until the resident network/calibration graph exists.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np


N_SAMPLES = 64
CLASSES = np.array([-3, 42], dtype=np.int64)
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "method", "metric", "value", "max_abs_error", "oracle",
    "seconds", "python_version", "numpy_version", "fortml_revision",
    "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            pass
    dirty = any(
        line[3:].split(" -> ")[-1].strip() not in ignored_names
        for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True,
        ).splitlines()
    )
    return head + ("+dirty" if dirty else "")


def fixture_labels() -> np.ndarray:
    phase = np.arange(1, N_SAMPLES + 1, dtype=np.float64)
    first = np.sin(0.19 * phase) + 0.02 * phase
    second = np.cos(0.13 * phase) - 0.01 * phase
    return np.where(first + 0.7 * second > 0.0, 42, -3).astype(np.int64)


def parse_oracle(path: Path) -> dict[tuple[str, int, int], float]:
    values: dict[tuple[str, int, int], float] = {}
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            values[(record["quantity"], int(record["row"]), int(record["column"]))] = float(record["value"])
    return values


def row(details: dict[str, str], **values: object) -> dict[str, object]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/mlp_calibrated_classifier.csv"))
    parser.add_argument("--target", default="fortml_bench_mlp_calibrated_classifier")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
        "oracle": "independent NumPy fixture/class-order/simplex/domain contract",
    }
    records: list[dict[str, object]] = []
    expected_labels = fixture_labels()
    if args.skip_fortml:
        records.append(row(details, workload="mlp_calibrated_classifier", phase="contract",
                           backend="fortml", device="cpu", status="skipped",
                           n_samples=N_SAMPLES, n_features=2, method="temperature",
                           metric="oracle", value="nan", max_abs_error="nan",
                           seconds="", notes="--skip-fortml"))
    else:
        source = fortml / "app" / f"{args.target}.f90"
        if not source.is_file():
            raise RuntimeError(f"release source absent: {source}")
        environment = os.environ.copy()
        environment.update({"FO_FC": environment.get("FO_FC", "gfortran"),
                            "OMP_NUM_THREADS": "1"})
        build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                               env=environment, capture_output=True, text=True)
        if build.returncode:
            raise RuntimeError(build.stderr.strip() or build.stdout.strip())
        with tempfile.TemporaryDirectory(dir="/mnt/storage",
                                          prefix="fortml-calibrated-classifier-") as temporary:
            oracle_path = Path(temporary) / "oracle.csv"
            check_environment = dict(environment)
            check_environment["FORTML_BENCH_MLP_CALIBRATED_ORACLE"] = str(oracle_path)
            check = subprocess.run(["fo", "exec", "--no-build", args.target], cwd=fortml,
                                   env=check_environment, capture_output=True, text=True)
            if check.returncode or not oracle_path.is_file():
                raise RuntimeError(check.stderr.strip() or "calibrated app emitted no oracle")
            values = parse_oracle(oracle_path)
            labels = np.array([round(values[("label", i, 1)]) for i in range(1, N_SAMPLES + 1)], dtype=np.int64)
            predictions = np.array([round(values[("prediction", i, 1)]) for i in range(1, N_SAMPLES + 1)], dtype=np.int64)
            probabilities = np.array([
                [values[("probability", i, 1)], values[("probability", i, 2)]]
                for i in range(1, N_SAMPLES + 1)
            ])
            errors = [
                float(np.max(np.abs(labels - expected_labels))),
                float(np.max(np.abs(np.sum(probabilities, axis=1) - 1.0))),
                float(np.max(np.maximum(-probabilities, 0.0))),
                float(np.max(np.maximum(probabilities - 1.0, 0.0))),
                0.0 if np.all(np.isin(predictions, CLASSES)) else 1.0,
            ]
            error = max(errors)
            if error > 1.0e-12 or not np.all(np.isfinite(probabilities)):
                raise RuntimeError(f"calibrated MLP contract mismatch: {error:.3e}")
            fit_seconds = next(float(line.rsplit(",", 1)[1])
                               for line in check.stdout.splitlines()
                               if line.startswith("mlp_calibrated_classifier_fit,"))
            predict_seconds = next(float(line.rsplit(",", 1)[1])
                                   for line in check.stdout.splitlines()
                                   if line.startswith("mlp_calibrated_classifier_predict,"))
        records.extend([
            row(details, workload="mlp_calibrated_classifier", phase="fit",
                backend="fortml", device="cpu", status="pass", n_samples=N_SAMPLES,
                n_features=2, method="temperature", metric="contract_max_abs_error",
                value=error, max_abs_error=error, seconds=fit_seconds,
                notes="sorted classes, fixture labels, finite probability simplex"),
            row(details, workload="mlp_calibrated_classifier", phase="predict",
                backend="fortml", device="cpu", status="pass", n_samples=N_SAMPLES,
                n_features=2, method="temperature", metric="probability_simplex",
                value=1.0, max_abs_error=error, seconds=predict_seconds,
                notes="prediction domain and calibrated probability rows checked"),
        ])
    records.append(row(details, workload="mlp_calibrated_classifier", phase="device_capability",
                       backend="fortml", device="cuda", status="unavailable",
                       n_samples=N_SAMPLES, n_features=2, method="temperature",
                       metric="predict_proba", value="nan", max_abs_error="nan",
                       seconds="", oracle="typed device contract",
                       notes="FORTNUM_NOT_IMPLEMENTED; resident neural/calibration CUDA graph is not linked"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} rows to {output}")


if __name__ == "__main__":
    main()
