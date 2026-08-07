#!/usr/bin/env python3
"""Correctness-gated benchmark for the dense RBF one-class SVM lane."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


N_SAMPLES, N_FEATURES, N_QUERY = 48, 2, 4
NU, GAMMA = 0.5, 0.8
ORACLE_REPETITIONS = 64
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_query", "seconds_per_operation", "accuracy",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
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
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        name = line[3:].split(" -> ")[-1].strip()
        if name not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray]:
    angles = np.arange(N_SAMPLES, dtype=np.float64) * (2.0 * np.pi / N_SAMPLES)
    radii = 1.0 + 0.08 * np.sin(3.0 * angles)
    x = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    query = np.array(((1.0, 0.0), (0.0, 0.0), (1.8, 1.2), (-1.1, 0.2)),
                     dtype=np.float64)
    return x, query


def rbf_kernel(x: np.ndarray, gamma: float) -> np.ndarray:
    differences = x[:, None, :] - x[None, :, :]
    return np.exp(-gamma * np.sum(differences * differences, axis=2))


def project_capped_simplex(values: np.ndarray, cap: float) -> np.ndarray:
    lower = float(values.min() - cap)
    upper = float(values.max())
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        projected = np.clip(values - midpoint, 0.0, cap)
        if float(projected.sum()) > 1.0:
            lower = midpoint
        else:
            upper = midpoint
    return np.clip(values - 0.5 * (lower + upper), 0.0, cap)


def fit_oracle(x: np.ndarray) -> tuple[np.ndarray, float, int]:
    kernel = rbf_kernel(x, GAMMA)
    cap = 1.0 / (NU * len(x))
    alpha = np.full(len(x), 1.0 / len(x), dtype=np.float64)
    lipschitz = float(np.max(kernel.sum(axis=1)))
    step = 0.99 / lipschitz
    for iteration in range(1, 5001):
        previous = alpha.copy()
        alpha = project_capped_simplex(previous - step * (kernel @ previous), cap)
        if float(np.max(np.abs(alpha - previous))) <= 1.0e-10:
            break
    else:
        raise RuntimeError("NumPy one-class SVM dual oracle did not converge")
    scores = kernel @ alpha
    # Construct the complete KKT offset interval, including zero and capped
    # weights.  A projected iterate can leave a numerically inconsistent
    # interval; use the deterministic mean-score fallback in that case.
    lower, upper = -np.inf, np.inf
    free = (alpha > 1.0e-10) & (alpha < cap - 1.0e-10)
    if np.any(alpha <= 1.0e-10):
        lower = max(lower, float(np.max(scores[alpha <= 1.0e-10])))
    if np.any(alpha >= cap - 1.0e-10):
        upper = min(upper, float(np.min(scores[alpha >= cap - 1.0e-10])))
    if np.any(free):
        lower = max(lower, float(np.max(scores[free])))
        upper = min(upper, float(np.min(scores[free])))
    offset = (float(0.5 * (lower + upper))
              if np.isfinite(lower + upper) and lower <= upper
              else float(scores.mean()))
    if (abs(float(alpha.sum()) - 1.0) > 2.0e-10 or
            np.any(alpha < -2.0e-10) or np.any(alpha > cap + 2.0e-10)):
        raise RuntimeError("NumPy one-class SVM dual oracle violated capped-simplex constraints")
    return alpha, offset, iteration


def predict_oracle(x_train: np.ndarray, query: np.ndarray,
                   alpha: np.ndarray, offset: float) -> tuple[np.ndarray, np.ndarray]:
    kernel = np.exp(-GAMMA * np.sum((query[:, None, :] - x_train[None, :, :]) ** 2, axis=2))
    scores = kernel @ alpha - offset
    labels = np.where(scores >= 0.0, 1, -1).astype(np.int64)
    return scores, labels


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }


def make_row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = {field: "" for field in FIELDS}
    row.update(details)
    row.update({
        "workload": "one_class_svm", "phase": "", "backend": "",
        "device": "cpu", "status": "", "n_samples": N_SAMPLES,
        "n_features": N_FEATURES, "n_query": N_QUERY,
        "seconds_per_operation": "", "accuracy": "", "max_abs_error": "",
        "oracle": "", "notes": "",
    })
    row.update(values)
    return row


def numpy_rows(details: dict[str, str]) -> tuple[
        list[dict[str, Any]], np.ndarray, float, np.ndarray, np.ndarray]:
    x, query = fixture()
    alpha, offset, iterations = fit_oracle(x)
    expected_scores, expected_labels = predict_oracle(x, query, alpha, offset)
    fit_started = time.perf_counter()
    for _ in range(ORACLE_REPETITIONS):
        fit_oracle(x)
    fit_seconds = (time.perf_counter() - fit_started) / ORACLE_REPETITIONS
    predict_started = time.perf_counter()
    for _ in range(ORACLE_REPETITIONS):
        predict_oracle(x, query, alpha, offset)
    predict_seconds = (time.perf_counter() - predict_started) / ORACLE_REPETITIONS
    rows = [
        make_row(details, phase="fit", backend="numpy_oracle", status="pass",
                 seconds_per_operation=fit_seconds, accuracy=1.0,
                 max_abs_error=0.0,
                 oracle="independent NumPy capped-simplex RBF dual oracle",
                 notes=f"dual mass/support constraints passed; iterations={iterations}"),
        make_row(details, phase="predict", backend="numpy_oracle", status="pass",
                 seconds_per_operation=predict_seconds, accuracy=1.0,
                 max_abs_error=0.0,
                 oracle="independent NumPy RBF score/label oracle",
                 notes="signed score and ±1 anomaly-label checksum"),
    ]
    return rows, alpha, offset, expected_scores, expected_labels


def parse_fortran(path: Path) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    weights = np.full(N_SAMPLES, np.nan, dtype=np.float64)
    offset = np.nan
    scores = np.full(N_QUERY, np.nan, dtype=np.float64)
    labels = np.full(N_QUERY, -999, dtype=np.int64)
    with path.open(newline="") as stream:
        for record in csv.DictReader(stream):
            quantity = record["quantity"]
            row = int(record["row"]) - 1
            value = float(record["value"])
            if quantity == "weight":
                weights[row] = value
            elif quantity == "offset":
                offset = value
            elif quantity == "score":
                scores[row] = value
            elif quantity == "prediction":
                labels[row] = int(value)
            else:
                raise RuntimeError(f"unknown FortML quantity {quantity!r}")
    if (np.isnan(weights).any() or not np.isfinite(offset) or
            np.isnan(scores).any() or np.any(labels == -999)):
        raise RuntimeError("FortML omitted a one-class SVM output")
    return weights, offset, scores, labels


def parse_timing(stdout: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in stdout.splitlines():
        fields = line.strip().split(",")
        if len(fields) == 5 and fields[0] in {
                "one_class_svm_fit", "one_class_svm_predict"}:
            values[fields[0]] = float(fields[-1])
    return values


def fortml_rows(fortml: Path, details: dict[str, str], expected_alpha: np.ndarray,
                expected_offset: float, expected_scores: np.ndarray,
                expected_labels: np.ndarray, no_build: bool) -> list[dict[str, Any]]:
    target = "fortml_bench_one_class_svm"
    source = fortml / "app" / f"{target}.f90"
    if not source.is_file():
        return [make_row(
            details, phase=phase, backend="fortml_cpu", device="cpu", status="unavailable",
            oracle="typed release-target contract",
            notes=f"release target source is absent: {source.name}",
        ) for phase in ("fit", "predict")] + [make_row(
            details, phase="predict", backend="fortml_cuda", device="cuda",
            status="unavailable", oracle="typed_device_contract",
            notes="no resident RBF one-class SVM kernel; FORTNUM_NOT_IMPLEMENTED",
        )]
    if not no_build:
        subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, check=True)
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"),
                        "OMP_NUM_THREADS": "1"})
    with tempfile.TemporaryDirectory(dir=fortml / "build") as directory:
        output = Path(directory) / "one_class_svm.csv"
        environment["FORTML_BENCH_ONE_CLASS_SVM_OUTPUT"] = str(output)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", target], cwd=fortml,
            env=environment, capture_output=True, text=True, check=True,
        )
        actual_alpha, actual_offset, actual_scores, actual_labels = parse_fortran(output)
    error = max(
        float(np.max(np.abs(actual_alpha - expected_alpha))),
        abs(actual_offset - expected_offset),
        float(np.max(np.abs(actual_scores - expected_scores))),
        float(np.max(actual_labels != expected_labels)),
    )
    if error > 5.0e-9:
        raise RuntimeError(f"FortML one-class SVM oracle mismatch {error:.3e}")
    timing = parse_timing(completed.stdout)
    if "one_class_svm_fit" not in timing or "one_class_svm_predict" not in timing:
        raise RuntimeError("FortML one-class SVM app omitted timing records")
    rows = [
        make_row(details, phase="fit", backend="fortml_cpu", device="cpu", status="pass",
                 seconds_per_operation=timing["one_class_svm_fit"], accuracy=1.0,
                 max_abs_error=error,
                 oracle="independent NumPy capped-simplex dual/offset oracle",
                 notes="complete support weights and KKT offset matched"),
        make_row(details, phase="predict", backend="fortml_cpu", device="cpu", status="pass",
                 seconds_per_operation=timing["one_class_svm_predict"], accuracy=1.0,
                 max_abs_error=error,
                 oracle="independent NumPy RBF score/label oracle",
                 notes="complete signed scores and ±1 anomaly labels matched"),
        make_row(details, phase="predict", backend="fortml_cuda", device="cuda",
                 status="unavailable", oracle="typed_device_contract",
                 notes="no resident RBF one-class SVM kernel; FORTNUM_NOT_IMPLEMENTED"),
    ]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/one_class_svm.csv"))
    parser.add_argument("--no-build", action="store_true",
                        help="reuse a previously built FortML release app")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    details = metadata(root, fortml, args.output)
    rows, expected_alpha, expected_offset, expected_scores, expected_labels = numpy_rows(details)
    rows.extend(fortml_rows(fortml, details, expected_alpha, expected_offset,
                            expected_scores, expected_labels, args.no_build))
    if expected_scores.shape != (N_QUERY,) or expected_labels.shape != (N_QUERY,):
        raise RuntimeError("one-class SVM oracle returned malformed query arrays")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
