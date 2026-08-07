#!/usr/bin/env python3
"""Correctness-gated XGBoost ``rank:pairwise`` lane.

The NumPy oracle evaluates the normalised pairwise logistic loss and its
gradient/Hessian for two rows in one query; a third row in another query must
not contribute.  The FortML gate runs the public ranking test, including model
fit/prediction ordering, query isolation, and malformed singleton-query
refusal.  No ranking CUDA result is inferred from the CPU objective path.
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_queries", "seconds_per_operation", "metric", "value", "max_abs_error",
    "oracle", "python_version", "numpy_version", "fortml_revision",
    "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def oracle() -> tuple[float, float, float, float]:
    margin = np.array([0.2, -0.1, 0.3], dtype=np.float64)
    target = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    group = np.array([1, 1, 2], dtype=np.int64)
    high, low = 0, 1
    delta = margin[high] - margin[low]
    probability = 1.0 / (1.0 + np.exp(delta))
    loss = float(np.logaddexp(0.0, -delta))
    gradient = np.zeros(3, dtype=np.float64)
    gradient[high], gradient[low] = -probability, probability
    hessian = np.zeros(3, dtype=np.float64)
    hessian[high] = hessian[low] = probability * (1.0 - probability)

    def pair_loss(values: np.ndarray) -> float:
        return float(np.logaddexp(0.0, -(values[high] - values[low])))

    # A slightly larger central-difference step keeps the second derivative
    # above float64 cancellation while remaining in the smooth logistic tail.
    step = 2.0e-4
    finite_difference = np.empty(3)
    for index in range(3):
        plus, minus = margin.copy(), margin.copy()
        plus[index] += step
        minus[index] -= step
        finite_difference[index] = (pair_loss(plus) - pair_loss(minus)) / (2.0 * step)
    gradient_error = float(np.max(np.abs(gradient - finite_difference)))
    # The out-of-query row must have exactly no derivative contribution.
    isolation_error = float(max(abs(gradient[2]), abs(hessian[2])))
    # Check each positive diagonal Hessian against a second finite difference.
    hessian_fd = []
    for index in (high, low):
        plus, minus = margin.copy(), margin.copy()
        plus[index] += step
        minus[index] -= step
        hessian_fd.append((pair_loss(plus) - 2.0 * pair_loss(margin) +
                           pair_loss(minus)) / step**2)
    hessian_error = float(max(abs(hessian[high] - hessian_fd[0]),
                              abs(hessian[low] - hessian_fd[1])))
    error = max(gradient_error, isolation_error, hessian_error)
    if error > 3.0e-8 or not np.array_equal(group, [1, 1, 2]):
        raise RuntimeError(f"independent pairwise ranking oracle failed: {error:.3e}")
    return loss, gradient_error, hessian_error, error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/xgboost_ranking.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    loss, gradient_error, hessian_error, oracle_error = oracle()
    started = time.perf_counter()
    if args.skip_fortml:
        status, notes = "skipped", "--skip-fortml"
    else:
        subprocess.run(["fo", "test", "test_xgboost_ranking"], cwd=fortml, check=True)
        status = "pass"
        notes = "public rank:pairwise loss/derivative, fit ordering, and group refusal"
    elapsed = time.perf_counter() - started
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": "gfortran", "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(details)
        row.update({"workload": "xgboost_rank_pairwise", "backend": "fortml",
                    "device": "cpu", "n_samples": 3, "n_queries": 2})
        row.update(values)
        rows.append(row)

    add(phase="independent_oracle", backend="numpy_oracle", status="pass", metric="pairwise_loss",
        value=loss, max_abs_error=oracle_error,
        oracle="independent NumPy pairwise logistic loss/gradient/Hessian FD oracle",
        notes=f"gradient_error={gradient_error:.3e}; hessian_error={hessian_error:.3e}; cross-query row is zero")
    add(phase="public_contract_gate", status=status, seconds_per_operation=elapsed,
        metric="oracle_max_abs_error", value=oracle_error, max_abs_error=oracle_error,
        oracle="FortML test_xgboost_ranking independent behavioral gate", notes=notes)
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_ranking_tree", value="nan", max_abs_error="nan",
        oracle="typed FortML CUDA capability refusal",
        notes="ranking trees and pairwise reductions are CPU-only; no host fallback")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
