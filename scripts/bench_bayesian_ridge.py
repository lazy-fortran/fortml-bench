#!/usr/bin/env python3
"""Correctness-gated weighted Bayesian-ridge posterior workload."""

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
TOLERANCE = 3.0e-11


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    ignored_names: set[str] = set()
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
    x = np.empty((64, 3), dtype=np.float64)
    y = np.empty((64, 2), dtype=np.float64)
    weights = np.empty(64, dtype=np.float64)
    for i in range(64):
        x[i, 0] = i / 16.0
        x[i, 1] = np.sin(x[i, 0])
        x[i, 2] = np.cos(0.5 * x[i, 0])
        y[i, 0] = 0.3 + 1.1*x[i, 0] - 0.2*x[i, 1] + 0.5*x[i, 2]
        y[i, 1] = -0.4 + 0.7*x[i, 0] + 0.4*x[i, 1] - 0.6*x[i, 2]
        weights[i] = 0.5 + (i + 1) % 5 / 5.0
    return x, y, weights


def oracle() -> tuple[np.ndarray, float]:
    x, y, weights = fixture()
    design = np.column_stack((np.ones(x.shape[0]), x))
    alpha, lam = 2.0, 0.8
    precision = lam*np.eye(design.shape[1]) + alpha*(design.T*weights)@design
    rhs = alpha*(design.T*weights)@y
    mean = np.linalg.solve(precision, rhs)
    sign, logdet = np.linalg.slogdet(precision)
    if sign <= 0:
        raise RuntimeError("oracle precision is not positive definite")
    residual = y - design@mean
    quad = alpha*np.sum(weights[:, None]*residual**2) + lam*np.sum(mean**2)
    logdet_likelihood = np.sum(np.log(alpha*weights))
    log_evidence = 0.5*y.shape[1]*(
        design.shape[1]*np.log(lam) - logdet + logdet_likelihood - quad
        - design.shape[0]*np.log(2*np.pi)
    )
    return mean, float(log_evidence)


def run_app(fortml: Path) -> tuple[dict[str, float | str], float]:
    env = os.environ.copy()
    env.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=env,
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    started = time.perf_counter()
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_bayesian_ridge"],
        cwd=fortml, env=env, check=True, capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - started
    values: dict[str, float | str] = {}
    for line in completed.stdout.splitlines():
        if line.strip() == "bayesian_ridge_cuda,unavailable":
            values["bayesian_ridge_cuda_status"] = "unavailable"
            continue
        if line.startswith("bayesian_ridge_") and "," in line:
            name, raw = line.split(",", 1)
            if name.startswith("bayesian_ridge_"):
                values[name.strip()] = float(raw.strip())
    required = {"bayesian_ridge_log_evidence", "bayesian_ridge_prediction_mean",
                "bayesian_ridge_precision_dimension", "bayesian_ridge_cuda_status"}
    missing = sorted(required - values.keys())
    if missing:
        raise RuntimeError(f"release app omitted {missing}")
    return values, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/bayesian_ridge.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/BAYESIAN_RIDGE.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    expected_mean, expected_evidence = oracle()
    expected_prediction_mean = float(np.mean(np.column_stack((np.ones(64), fixture()[0]))@expected_mean))
    values, seconds = run_app(fortml)
    evidence_error = abs(float(values["bayesian_ridge_log_evidence"]) - expected_evidence)
    prediction_error = abs(float(values["bayesian_ridge_prediction_mean"]) - expected_prediction_mean)
    if evidence_error > TOLERANCE or prediction_error > TOLERANCE:
        raise RuntimeError(f"NumPy posterior mismatch: evidence={evidence_error:g}, prediction={prediction_error:g}")
    if int(values["bayesian_ridge_precision_dimension"]) != 4:
        raise RuntimeError("unexpected posterior dimension")
    if values["bayesian_ridge_cuda_status"] != "unavailable":
        raise RuntimeError("Bayesian-ridge CUDA refusal changed")
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (args.output.resolve(), args.report.resolve())),
        "compiler": "gfortran", "flags": "-O3",
    }
    rows = [
        {**details, "workload": "bayesian_ridge", "phase": "posterior_evidence",
         "backend": "fortml", "device": "cpu", "status": "pass", "metric": "log_evidence",
         "value": float(values["bayesian_ridge_log_evidence"]), "max_abs_error": evidence_error,
         "seconds": seconds, "oracle": "independent NumPy weighted conjugate posterior",
         "notes": "alpha=2; lambda=0.8; dense 64x3, two outputs"},
        {**details, "workload": "bayesian_ridge", "phase": "prediction",
         "backend": "fortml", "device": "cpu", "status": "pass", "metric": "mean",
         "value": float(values["bayesian_ridge_prediction_mean"]), "max_abs_error": prediction_error,
         "seconds": seconds, "oracle": "independent NumPy weighted conjugate posterior",
         "notes": "posterior mean prediction"},
        {**details, "workload": "bayesian_ridge", "phase": "capability_check",
         "backend": "fortml", "device": "cuda", "status": "unavailable", "metric": "status",
         "value": 3.0, "max_abs_error": "", "seconds": seconds,
         "oracle": "declared resident-device contract", "notes": "FORTNUM_NOT_IMPLEMENTED; no host fallback"},
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# Weighted Bayesian ridge\n\n"
        "This lane compares the dense fixed-hyperparameter Bayesian-ridge posterior and prediction with an independent NumPy conjugate-Gaussian oracle. It records posterior evidence metadata and the typed CUDA refusal; evidence maximisation/ARD and resident GPU execution are not claimed.\n\n"
        f"- FortML revision: `{details['fortml_revision']}`\n"
        f"- Benchmark revision: `{details['benchmark_revision']}`\n"
        f"- Compiler/flags: `{details['compiler']} {details['flags']}`\n"
        f"- Release wall time: `{seconds:.6g}` s\n"
        f"- Evidence error: `{evidence_error:.6g}`\n"
        f"- Prediction-mean error: `{prediction_error:.6g}`\n"
        f"- Raw record: [`{args.output}`]({args.output.name})\n",
    )
    print(f"wrote {args.output} ({len(rows)} rows)")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
