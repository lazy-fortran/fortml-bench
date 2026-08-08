#!/usr/bin/env python3
"""Correctness-gated fixed-input multiclass MLP parameter-product benchmark."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_hidden", "n_classes", "n_parameters",
    "seconds_per_operation", "metric", "value", "expected_value",
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


def parse_oracle(path: Path) -> dict[tuple[str, int, int], float]:
    values: dict[tuple[str, int, int], float] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["quantity"], int(row["row"]), int(row["column"]))
            values[key] = float(row["value"])
    return values


def fixture() -> np.ndarray:
    return np.array([
        -2.0, -1.0, -1.0, -2.0, 0.0, 2.0,
        0.0, 1.0, 1.0, 0.0, 2.0, 1.0,
        2.0, 0.0, 1.0, 2.0, 2.0, 1.0,
    ], dtype=np.float64).reshape((9, 2), order="F")


def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    position = 0
    weight_1 = theta[position:position + 6].reshape((2, 3), order="F")
    position += 6
    bias_1 = theta[position:position + 3]
    position += 3
    weight_2 = theta[position:position + 9].reshape((3, 3), order="F")
    position += 9
    bias_2 = theta[position:position + 3]
    return weight_1, bias_1, weight_2, bias_2


def probabilities(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    weight_1, bias_1, weight_2, bias_2 = unpack(theta)
    hidden = np.tanh(x @ weight_1 + bias_1)
    logits = hidden @ weight_2 + bias_2
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def oracle(values: dict[tuple[str, int, int], float]) -> tuple[float, float, float, float]:
    theta = np.array([values[("parameter", i, 1)] for i in range(1, 22)], dtype=np.float64)
    direction = np.array([values[("parameter_tangent", i, 1)] for i in range(1, 22)], dtype=np.float64)
    cotangent = np.array([
        values[("probability_cotangent", i, j)]
        for i in range(1, 10) for j in range(1, 4)
    ], dtype=np.float64).reshape((9, 3))
    x = fixture()
    h = 1.0e-6
    plus = probabilities(theta + h*direction, x)
    minus = probabilities(theta - h*direction, x)
    expected_jvp = (plus - minus)/(2.0*h)
    observed_jvp = np.array([
        values[("probability_tangent", i, j)]
        for i in range(1, 10) for j in range(1, 4)
    ], dtype=np.float64).reshape((9, 3))
    jvp_error = float(np.max(np.abs(observed_jvp - expected_jvp)))
    expected_vjp = np.empty(theta.size, dtype=np.float64)
    for index in range(theta.size):
        plus_theta = theta.copy()
        minus_theta = theta.copy()
        plus_theta[index] += h
        minus_theta[index] -= h
        expected_vjp[index] = (
            np.sum(cotangent*probabilities(plus_theta, x))
            - np.sum(cotangent*probabilities(minus_theta, x))
        )/(2.0*h)
    observed_vjp = np.array([
        values[("parameter_bar", i, 1)] for i in range(1, 22)
    ], dtype=np.float64)
    vjp_error = float(np.max(np.abs(observed_vjp - expected_vjp)))
    observed_probabilities = np.array([
        values[("probability", i, j)]
        for i in range(1, 10) for j in range(1, 4)
    ], dtype=np.float64).reshape((9, 3))
    simplex_error = float(np.max(np.abs(np.sum(observed_probabilities, axis=1) - 1.0)))
    duality_error = abs(float(np.sum(cotangent*observed_jvp)) -
                        float(np.dot(observed_vjp, direction)))
    return jvp_error, vjp_error, simplex_error, duality_error


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def timing(stdout: str, phase: str) -> float:
    match = re.search(
        rf"^mlp_classifier_parameter_products_{phase},tanh,\s*([0-9Ee+.-]+)$",
        stdout, re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"release app omitted {phase} timing")
    return float(match.group(1))


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
        "oracle": "independent NumPy tanh MLP replay and central finite differences",
    }
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                   env=environment, check=True, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE, text=True)
    with tempfile.TemporaryDirectory(prefix="fortml-mlp-param-products-", dir="/mnt/storage") as directory:
        oracle_path = Path(directory) / "oracle.csv"
        environment["FORTML_BENCH_MLP_CLASSIFIER_PARAMETER_PRODUCTS_ORACLE"] = str(oracle_path)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", args.target], cwd=fortml,
            env=environment, check=True, capture_output=True, text=True,
        )
        values = parse_oracle(oracle_path)
    jvp_error, vjp_error, simplex_error, duality_error = oracle(values)
    tolerance = 5.0e-5
    if jvp_error > tolerance or vjp_error > 5.0e-6 or simplex_error > 5.0e-13 or duality_error > 5.0e-10:
        raise RuntimeError(
            f"parameter-product oracle mismatch: jvp={jvp_error:.3e}, "
            f"vjp={vjp_error:.3e}, simplex={simplex_error:.3e}, duality={duality_error:.3e}"
        )
    common = dict(n_samples=9, n_features=2, n_hidden=3, n_classes=3, n_parameters=21)
    rows = [
        row(details, workload="mlp_classifier_parameter_products", phase="fit",
            backend="fortml", device="cpu", status="pass",
            seconds_per_operation=timing(completed.stdout, "fit"), metric="fit_seconds",
            value=timing(completed.stdout, "fit"), expected_value="",
            max_abs_error="", notes="deterministic 2-3-3 tanh classifier"),
        row(details, workload="mlp_classifier_parameter_products", phase="predict",
            backend="fortml", device="cpu", status="pass",
            seconds_per_operation=timing(completed.stdout, "predict"), metric="predict_seconds",
            value=timing(completed.stdout, "predict"), notes="softmax probability baseline"),
        row(details, workload="mlp_classifier_parameter_products", phase="parameter_jvp",
            backend="fortml", device="cpu", status="pass",
            seconds_per_operation=timing(completed.stdout, "jvp"), metric="max_abs_error",
            value=jvp_error, expected_value=0.0, max_abs_error=jvp_error,
            notes="fixed-input packed-parameter JVP"),
        row(details, workload="mlp_classifier_parameter_products", phase="parameter_vjp",
            backend="fortml", device="cpu", status="pass",
            seconds_per_operation=timing(completed.stdout, "vjp"), metric="max_abs_error",
            value=vjp_error, expected_value=0.0, max_abs_error=vjp_error,
            notes="fixed-input packed-parameter VJP"),
        row(details, workload="mlp_classifier_parameter_products", phase="duality",
            backend="fortml", device="cpu", status="pass", metric="jvp_vjp_duality_error",
            value=duality_error, expected_value=0.0, max_abs_error=duality_error,
            notes="Euclidean cotangent/tangent contraction"),
        row(details, workload="mlp_classifier_parameter_products", phase="device_contract",
            backend="fortml", device="cuda", status="unavailable", metric="parameter_jvp",
            value="unavailable", expected_value="", max_abs_error="",
            oracle="FortML typed device capability contract",
            notes="resident multiclass MLP CUDA graph is not linked; no host fallback"),
    ]
    for item in rows:
        item.update(common)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/mlp_classifier_parameter_products.csv"))
    parser.add_argument("--target", default="fortml_bench_mlp_classifier_parameter_products")
    args = parser.parse_args()
    records = run(args)
    with args.output.resolve().open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} rows to {args.output.resolve()}")


if __name__ == "__main__":
    main()
