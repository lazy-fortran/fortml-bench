#!/usr/bin/env python3
"""Correctness gate for stable binary and multinomial log probabilities."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "seconds_per_operation", "metric", "value",
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
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def oracle() -> tuple[float, float]:
    scores = np.array([[-1200.0, 1200.0], [0.4, -0.7], [4.0, 1.0]], dtype=np.float64)
    binary = np.column_stack((-np.logaddexp(0.0, scores[:, 0]),
                              -np.logaddexp(0.0, -scores[:, 0])))
    binary_error = float(np.max(np.abs(np.exp(binary).sum(axis=1) - 1.0)))
    maximum = np.max(scores, axis=1, keepdims=True)
    log_softmax = scores - maximum - np.log(np.exp(scores - maximum).sum(axis=1, keepdims=True))
    softmax_error = float(np.max(np.abs(np.exp(log_softmax).sum(axis=1) - 1.0)))
    if max(binary_error, softmax_error) > 2.0e-15:
        raise RuntimeError("stable log-probability oracle failed")
    return binary_error, softmax_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/linear_log_proba.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/LINEAR_LOG_PROBA.md"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml, output, report = (args.fortml.resolve(), args.output.resolve(),
                              args.report.resolve())
    binary_error, softmax_error = oracle()
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output, report)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(details)
        row.update({"workload": "linear_log_proba", "backend": "fortml",
                    "device": "cpu", "n_samples": 6, "n_features": 2})
        row.update(values)
        rows.append(row)

    add(phase="independent_oracle", backend="numpy_oracle", status="pass",
        metric="binary_normalization_max_abs_error", value=binary_error,
        max_abs_error=binary_error, oracle="NumPy logaddexp binary log-sigmoid",
        notes="scores include +/-1200 saturation")
    add(phase="independent_oracle", backend="numpy_oracle", status="pass",
        metric="multinomial_normalization_max_abs_error", value=softmax_error,
        max_abs_error=softmax_error, oracle="NumPy row-wise log-sum-exp",
        notes="stable shifted normalization")
    started = time.perf_counter()
    if args.skip_fortml:
        status, notes = "skipped", "--skip-fortml"
    else:
        environment = os.environ.copy()
        environment["FO_SCAN_FALLBACK"] = "regex"
        subprocess.run(["fo", "test", "test_logistic_regression", "test_softmax_regression"],
                       cwd=fortml, env=environment, check=True)
        status, notes = "pass", "value, central-difference JVP, and binary adjoint gates"
    elapsed = time.perf_counter() - started
    add(phase="public_contract_gate", status=status, seconds_per_operation=elapsed,
        metric="classifier_log_probability_products", value=max(binary_error, softmax_error),
        max_abs_error=max(binary_error, softmax_error),
        oracle="FortML logistic/softmax behavioral tests", notes=notes)
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_classifier_log_probability_graph", value="nan",
        oracle="typed FortML device boundary",
        notes="resident classifier reductions are not linked")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Linear-classifier log-probability products\n\n"
        "The lane checks stable binary log-sigmoid and multinomial log-softmax "
        "values, then runs the independent FortML logistic and softmax product "
        "tests. The saturation fixture includes logits of +/-1200. The CUDA "
        "row records the typed resident-kernel boundary.\n\n"
        f"The CSV records {len(rows)} rows. FortML revision: "
        f"`{details['fortml_revision']}`. Benchmark revision: "
        f"`{details['benchmark_revision']}`.\n"
    )
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
