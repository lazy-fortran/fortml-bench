#!/usr/bin/env python3
"""Correctness-gated deterministic linear-SGD regression/classification lane."""

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
    "workload", "phase", "backend", "device", "status", "metric", "value",
    "max_abs_error", "seconds", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
TOLERANCE = 3.0e-12


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
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
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.empty((8, 2), dtype=np.float64)
    y = np.empty(8, dtype=np.float64)
    labels = np.empty(8, dtype=np.int64)
    for i in range(8):
        x[i] = (float(i), float((i + 1) % 3))
        y[i] = 0.5 + 1.5*x[i, 0] - 0.25*x[i, 1]
        labels[i] = 1 if y[i] > 5.0 else -1
    return x, y, labels


def shuffle(order: list[int], state: int) -> int:
    for i in range(len(order), 1, -1):
        state = (48271*state) % 2147483647
        j = state % i
        order[i - 1], order[j] = order[j], order[i - 1]
    return state


def regression_oracle() -> float:
    x, y, _ = fixture()
    weight = np.zeros(2)
    intercept = 0.0
    average_weight = np.zeros(2)
    average_intercept = 0.0
    average_count = 0
    state = 41
    for _ in range(64):
        order = list(range(8))
        state = shuffle(order, state)
        for start in range(0, 8, 2):
            batch = order[start:start + 2]
            residual = intercept + x[batch] @ weight - y[batch]
            weight -= 0.03*np.mean(residual[:, None]*x[batch], axis=0)
            intercept -= 0.03*np.mean(residual)
            average_weight += weight
            average_intercept += intercept
            average_count += 1
    prediction = x @ (average_weight/average_count) + average_intercept/average_count
    return float(np.mean((prediction-y)**2))


def classifier_oracle() -> float:
    x, _, labels = fixture()
    weight = np.zeros(2)
    intercept = 0.0
    state = 41
    for _ in range(96):
        order = list(range(8))
        state = shuffle(order, state)
        for start in range(0, 8, 2):
            batch = order[start:start + 2]
            target = (labels[batch] == 1).astype(np.float64)
            logits = intercept + x[batch] @ weight
            probability = 1.0/(1.0 + np.exp(-logits))
            residual = probability-target
            weight -= 0.03*np.mean(residual[:, None]*x[batch], axis=0)
            intercept -= 0.03*np.mean(residual)
    probability = 1.0/(1.0 + np.exp(-(intercept+x@weight)))
    return float(np.mean(probability))


def run_app(fortml: Path) -> tuple[dict[str, float], float]:
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    started = time.perf_counter()
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_linear_sgd"], cwd=fortml,
        env=environment, check=True, capture_output=True, text=True,
    )
    values: dict[str, float] = {}
    elapsed = time.perf_counter() - started
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4 and fields[0].startswith("linear_sgd_"):
            values[fields[0]] = float(fields[3])
    if set(values) != {"linear_sgd_regression", "linear_sgd_classifier"}:
        raise RuntimeError(f"release app omitted rows: {sorted(values)}")
    return values, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/linear_sgd.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/LINEAR_SGD.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    expected_regression = regression_oracle()
    expected_classifier = classifier_oracle()
    values, seconds = run_app(fortml)
    regression_error = abs(values["linear_sgd_regression"] - expected_regression)
    classifier_error = abs(values["linear_sgd_classifier"] - expected_classifier)
    if regression_error > TOLERANCE or classifier_error > TOLERANCE:
        raise RuntimeError(f"NumPy recurrence mismatch: regression={regression_error:g}, classifier={classifier_error:g}")
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (args.output.resolve(), args.report.resolve())),
        "compiler": "gfortran", "flags": "-O3",
    }
    rows = [
        {**details, "workload": "linear_sgd", "phase": "regression", "backend": "fortml",
         "device": "cpu", "status": "pass", "metric": "mse", "value": values["linear_sgd_regression"],
         "max_abs_error": regression_error, "seconds": seconds,
         "oracle": "independent NumPy mini-batch/Polyak recurrence",
         "notes": "64 epochs; batch_size=2; seeded shuffle=41; average=true"},
        {**details, "workload": "linear_sgd", "phase": "classification", "backend": "fortml",
         "device": "cpu", "status": "pass", "metric": "mean_positive_probability",
         "value": values["linear_sgd_classifier"], "max_abs_error": classifier_error, "seconds": seconds,
         "oracle": "independent NumPy mini-batch logistic recurrence",
         "notes": "96 epochs; batch_size=2; seeded shuffle=41; labels=[-1,1]"},
        {**details, "workload": "linear_sgd", "phase": "capability_check", "backend": "fortml",
         "device": "cuda", "status": "unavailable", "metric": "status", "value": 3.0,
         "max_abs_error": "", "seconds": seconds, "oracle": "declared resident-device contract",
         "notes": "FORTNUM_NOT_IMPLEMENTED; no host fallback"},
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# Deterministic linear SGD\n\n"
        "The release application is checked against independent NumPy mini-batch recurrences for weighted linear regression and binary logistic classification. The CUDA row records the typed refusal; resident stochastic state and stochastic hypergradients are not claimed.\n\n"
        f"- FortML revision: `{details['fortml_revision']}`\n"
        f"- Benchmark revision: `{details['benchmark_revision']}`\n"
        f"- Compiler/flags: `{details['compiler']} {details['flags']}`\n"
        f"- Release wall time: `{seconds:.6g}` s\n"
        f"- Regression MSE error: `{regression_error:.6g}`\n"
        f"- Classifier probability error: `{classifier_error:.6g}`\n"
        f"- Raw record: [`{args.output}`]({args.output.name})\n",
    )
    print(f"wrote {args.output} ({len(rows)} rows)")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()

