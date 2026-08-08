#!/usr/bin/env python3
"""Correctness-gated benchmark for independent multilabel Laplace GPs.

The NumPy reference independently performs the weighted binary Laplace
Newton solve and the logistic predictive approximation for both label heads.
The FortML release probe is accepted only when every probability and input JVP
matches that reference; the CUDA row records the explicit typed refusal.
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


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_labels", "n_features", "n_parameters", "seconds_per_operation",
    "metric", "value", "max_abs_error", "oracle", "python_version",
    "numpy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
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


def fit_head(x: np.ndarray, labels: np.ndarray, weights: np.ndarray,
             variance: float, lengthscale: float) -> tuple[np.ndarray, np.ndarray]:
    distance = (x[:, None] - x[None, :]) ** 2
    kernel = variance * np.exp(-0.5 * distance / lengthscale**2)
    kernel[np.diag_indices_from(kernel)] += 1.0e-7
    signed = 2.0 * labels.astype(np.float64) - 1.0
    mode = np.zeros(x.size, dtype=np.float64)
    for _ in range(100):
        margin = signed * mode
        probability = 1.0 / (1.0 + np.exp(-margin))
        gradient = 1.0 - probability
        curvature = np.maximum(probability * (1.0 - probability), 1.0e-12)
        sqrt_w = np.sqrt(np.maximum(weights * curvature, 1.0e-12))
        b = weights * curvature * mode + signed * weights * gradient
        system = np.eye(x.size) + sqrt_w[:, None] * kernel * sqrt_w[None, :]
        rhs = np.linalg.solve(system, sqrt_w * (kernel @ b))
        new_mode = kernel @ (b - sqrt_w * rhs)
        scale = max(1.0, float(np.max(np.abs(mode))))
        if np.max(np.abs(new_mode - mode)) / scale <= 1.0e-9:
            mode = new_mode
            break
        mode = new_mode
    margin = signed * mode
    probability = 1.0 / (1.0 + np.exp(-margin))
    curvature = np.maximum(probability * (1.0 - probability), 1.0e-12)
    sqrt_w = np.sqrt(np.maximum(weights * curvature, 1.0e-12))
    alpha = np.linalg.solve(kernel, mode)
    return alpha, sqrt_w


def predict_head(x: np.ndarray, query: np.ndarray, alpha: np.ndarray,
                 sqrt_w: np.ndarray, variance: float,
                 lengthscale: float) -> np.ndarray:
    cross = variance * np.exp(-0.5 * (x[:, None] - query[None, :])**2 / lengthscale**2)
    mean = cross.T @ alpha
    train_kernel = variance * np.exp(-0.5 * (x[:, None] - x[None, :])**2 / lengthscale**2)
    train_kernel[np.diag_indices_from(train_kernel)] += 1.0e-7
    system = np.eye(x.size) + sqrt_w[:, None] * train_kernel * sqrt_w[None, :]
    work = np.linalg.solve(system, sqrt_w[:, None] * cross)
    prior = np.full(query.size, variance)
    posterior_variance = np.maximum(prior - np.sum(work * work, axis=0), 0.0)
    scale = np.sqrt(1.0 + np.pi * posterior_variance / 8.0)
    positive = 1.0 / (1.0 + np.exp(-mean / scale))
    return positive


def oracle() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.array([-2.0, -1.5, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 1.5, 2.0])
    labels = np.array([[0, 1], [0, 1], [0, 1], [0, 0], [0, 0],
                       [1, 0], [1, 0], [1, 1], [1, 1], [1, 1]], dtype=np.int64)
    query = np.array([-1.7, -0.4, 0.0, 0.7, 1.7])
    query_dot = np.array([0.2, -0.3, 0.1, 0.4, -0.2])
    weights = np.array([1.0, 0.9, 1.1, 1.0, 0.8, 1.2, 1.0, 1.1, 0.9, 1.0])
    positive = np.empty((query.size, labels.shape[1]))
    for index in range(labels.shape[1]):
        alpha, sqrt_w = fit_head(x, labels[:, index], weights, 1.3, 0.75)
        positive[:, index] = predict_head(x, query, alpha, sqrt_w, 1.3, 0.75)
    step = 2.0e-6
    plus = np.empty_like(positive)
    minus = np.empty_like(positive)
    for index in range(labels.shape[1]):
        alpha, sqrt_w = fit_head(x, labels[:, index], weights, 1.3, 0.75)
        plus[:, index] = predict_head(x, query + step * query_dot, alpha, sqrt_w, 1.3, 0.75)
        minus[:, index] = predict_head(x, query - step * query_dot, alpha, sqrt_w, 1.3, 0.75)
    jvp = (plus - minus) / (2.0 * step)
    return positive, jvp, (positive > np.array([0.5, 0.6])).astype(np.int64), float(np.max(np.abs(jvp)))


def parse_probe(output: str) -> dict[str, list[list[str]]]:
    records: dict[str, list[list[str]]] = {}
    for line in output.splitlines():
        if not line.startswith("gp_multilabel_"):
            continue
        fields = next(csv.reader([line]))
        records.setdefault(fields[0], []).append(fields[1:])
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/gp_multilabel.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml, output = args.fortml.resolve(), args.output.resolve()
    expected, expected_jvp, expected_labels, oracle_jvp = oracle()
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)), "compiler": "gfortran",
        "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(details)
        row.update({"workload": "gp_multilabel", "backend": "fortml", "device": "cpu",
                    "n_samples": 10, "n_labels": 2, "n_features": 1,
                    "n_parameters": 4})
        row.update(values)
        rows.append(row)

    add(phase="independent_oracle", backend="numpy_oracle", status="pass",
        metric="input_jvp_max_abs", value=oracle_jvp, max_abs_error=0.0,
        oracle="independent NumPy weighted binary Laplace Newton and predictive finite difference",
        notes="two independent Bernoulli heads; probabilities are not simplex-normalized")
    started = time.perf_counter()
    if args.skip_fortml:
        status, records, notes = "skipped", {}, "--skip-fortml"
    else:
        environment = os.environ.copy()
        environment["FO_SCAN_FALLBACK"] = "regex"
        completed = subprocess.run(
            ["fo", "exec", "--no-build", "fortml_bench_gp_multilabel"],
            cwd=fortml, env=environment, check=True, capture_output=True, text=True,
        )
        status, records, notes = "pass", parse_probe(completed.stdout), "release probe"
    elapsed = time.perf_counter() - started
    if status == "pass":
        observed = np.zeros_like(expected)
        observed_jvp = np.zeros_like(expected_jvp)
        observed_labels = np.zeros_like(expected_labels)
        for values in records["gp_multilabel_probability"]:
            observed[int(values[0]) - 1, int(values[1]) - 1] = float(values[2])
        for values in records["gp_multilabel_probability_jvp"]:
            observed_jvp[int(values[0]) - 1, int(values[1]) - 1] = float(values[2])
        for values in records["gp_multilabel_prediction"]:
            observed_labels[int(values[0]) - 1, int(values[1]) - 1] = int(values[2])
        error = float(max(np.max(np.abs(observed - expected)),
                          np.max(np.abs(observed_jvp - expected_jvp)),
                          np.max(np.abs(observed_labels - expected_labels))))
        if error > 4.0e-6:
            raise RuntimeError(f"multilabel GP oracle mismatch: {error:.3e}")
        fit_seconds = float(records["gp_multilabel_fit_seconds"][0][0])
        predict_seconds = float(records["gp_multilabel_predict_seconds"][0][0])
    else:
        error, fit_seconds, predict_seconds = float("nan"), float("nan"), float("nan")
    add(phase="fit", status=status, seconds_per_operation=fit_seconds,
        metric="fit_seconds", value=fit_seconds, max_abs_error=error,
        oracle="FortML multilabel Laplace-GP release probe", notes=notes)
    add(phase="predict_derivatives", status=status, seconds_per_operation=predict_seconds,
        metric="probability_and_input_jvp_max_abs", value=error, max_abs_error=error,
        oracle="NumPy multilabel GP posterior and finite-difference JVP", notes=notes)
    cuda_status = "unavailable"
    cuda_code = 3
    if status == "pass":
        cuda_code = int(records["gp_multilabel_cuda"][0][0])
        if cuda_code != 3:
            raise RuntimeError(f"unexpected CUDA status code {cuda_code}")
    add(phase="device_contract", device="cuda", status=cuda_status,
        metric="resident_multilabel_laplace_graph", value="nan", max_abs_error="nan",
        oracle="typed FORTNUM_NOT_IMPLEMENTED refusal", notes=f"status_code={cuda_code}; no host fallback")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
