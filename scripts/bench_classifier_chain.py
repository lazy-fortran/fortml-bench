#!/usr/bin/env python3
"""Correctness-gated classifier-chain benchmark.

The release app writes fitted packed parameters, probabilities, and hard
labels.  The NumPy oracle below replays the same sequential sigmoid heads from
those parameters, so the timing rows are accepted only after an independent
forward and thresholding check.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


N_SAMPLES = 192
N_FEATURES = 4
N_OUTPUTS = 3
N_PARAMETERS = N_OUTPUTS * N_FEATURES + N_OUTPUTS * (N_OUTPUTS + 1) // 2
FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_outputs", "seconds_per_operation", "metric", "value",
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
            continue
    dirty = any(
        line[3:].split(" -> ")[-1].strip() not in ignored_names
        for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True,
        ).splitlines()
    )
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray]:
    x = np.empty((N_SAMPLES, N_FEATURES), dtype=np.float64)
    labels = np.empty((N_SAMPLES, N_OUTPUTS), dtype=np.int64)
    for i in range(N_SAMPLES):
        phase = float(i + 1)
        for j in range(N_FEATURES):
            x[i, j] = np.sin(0.021 * phase + 0.083 * (j + 1)) + \
                0.15 * np.cos(0.011 * phase * (j + 1))
        score = 0.8 * x[i, 0] - 0.45 * x[i, 1] + np.sin(0.13 * phase) * 0.15
        labels[i, 0] = int(score > 0.0)
        score = -0.35 * x[i, 0] + 0.7 * x[i, 2] - \
            0.1 * np.cos(0.09 * phase + 0.3)
        labels[i, 1] = int(score > 0.0)
        score = 0.3 * x[i, 1] + 0.55 * x[i, 3] + \
            0.2 * np.sin(0.07 * phase + 0.5)
        labels[i, 2] = int(score > 0.0)
    return x, labels


def oracle(x: np.ndarray, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.empty((x.shape[0], N_OUTPUTS), dtype=np.float64)
    augmented = np.zeros((x.shape[0], N_FEATURES + N_OUTPUTS - 1), dtype=np.float64)
    augmented[:, :N_FEATURES] = x
    position = 0
    for output in range(N_OUTPUTS):
        feature_count = N_FEATURES + output
        count = feature_count + 1
        head = parameters[position:position + count]
        position += count
        logits = augmented[:, :feature_count] @ head[:feature_count] + head[-1]
        probabilities[:, output] = 1.0 / (1.0 + np.exp(-logits))
        if output + 1 < N_OUTPUTS:
            augmented[:, N_FEATURES + output] = probabilities[:, output]
    predicted = (probabilities >= 0.5).astype(np.int64)
    return probabilities, predicted


def directions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_dot = np.empty((N_SAMPLES, N_FEATURES), dtype=np.float64)
    probabilities_bar = np.empty((N_SAMPLES, N_OUTPUTS), dtype=np.float64)
    theta_dot = np.empty(N_PARAMETERS, dtype=np.float64)
    for i in range(N_SAMPLES):
        for j in range(N_FEATURES):
            x_dot[i, j] = 0.003 * np.cos(0.017 * (i + 1) * (j + 1))
        for j in range(N_OUTPUTS):
            probabilities_bar[i, j] = 0.2 * np.sin(
                0.03 * (i + 1) + 0.4 * (j + 1)
            )
    for i in range(N_PARAMETERS):
        theta_dot[i] = 0.01 * np.sin(0.13 * (i + 1))
    return x_dot, probabilities_bar, theta_dot


def vjp(x: np.ndarray, parameters: np.ndarray,
        cotangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.empty((x.shape[0], N_OUTPUTS), dtype=np.float64)
    augmented = np.zeros((x.shape[0], N_FEATURES + N_OUTPUTS - 1), dtype=np.float64)
    augmented[:, :N_FEATURES] = x
    heads: list[tuple[np.ndarray, int]] = []
    position = 0
    for output in range(N_OUTPUTS):
        feature_count = N_FEATURES + output
        count = feature_count + 1
        head = parameters[position:position + count]
        position += count
        heads.append((head, feature_count))
        logits = augmented[:, :feature_count] @ head[:feature_count] + head[-1]
        probabilities[:, output] = 1.0 / (1.0 + np.exp(-logits))
        if output + 1 < N_OUTPUTS:
            augmented[:, N_FEATURES + output] = probabilities[:, output]
    cotangent_work = cotangent.copy()
    parameter_bar = np.zeros_like(parameters)
    x_bar = np.zeros_like(x)
    for output in range(N_OUTPUTS - 1, -1, -1):
        head, feature_count = heads[output]
        p = probabilities[:, output]
        q = cotangent_work[:, output] * p * (1.0 - p)
        first = sum(N_FEATURES + k + 1 for k in range(output))
        parameter_bar[first:first + feature_count] = q @ augmented[:, :feature_count]
        parameter_bar[first + feature_count] = np.sum(q)
        augmented_bar = q[:, None] * head[:feature_count][None, :]
        x_bar += augmented_bar[:, :N_FEATURES]
        if output > 0:
            cotangent_work[:, :output] += augmented_bar[:, N_FEATURES:]
    return parameter_bar, x_bar


def parse(stdout: str, path: Path) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    timing: dict[str, float] = {}
    for line in stdout.splitlines():
        fields = line.strip().split(",")
        if len(fields) == 5 and fields[0].startswith("classifier_chain_"):
            timing[fields[0]] = float(fields[-1])
    parameters = np.zeros(sum(N_FEATURES + output + 1 for output in range(N_OUTPUTS)))
    probabilities = np.zeros((N_SAMPLES, N_OUTPUTS))
    predicted = np.zeros((N_SAMPLES, N_OUTPUTS), dtype=np.int64)
    theta_hvp = np.zeros(N_PARAMETERS)
    x_hvp = np.zeros((N_SAMPLES, N_FEATURES))
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            quantity = row["quantity"]
            index = int(row["row"]) - 1
            column = int(row["column"]) - 1
            if quantity == "parameter":
                parameters[index] = float(row["value"])
            elif quantity == "probability":
                probabilities[index, column] = float(row["value"])
            elif quantity == "prediction":
                predicted[index, column] = int(float(row["value"]))
            elif quantity == "theta_hvp":
                theta_hvp[index] = float(row["value"])
            elif quantity == "x_hvp":
                x_hvp[index, column] = float(row["value"])
    if set(timing) != {"classifier_chain_fit", "classifier_chain_predict", "classifier_chain_hvp"}:
        raise RuntimeError("release app omitted classifier-chain timings")
    return timing, parameters, probabilities, predicted, theta_hvp, x_hvp


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/classifier_chain.csv"))
    parser.add_argument("--target", default="fortml_bench_classifier_chain")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    observed_path = args.output.with_name(f".{args.output.stem}_observed.csv").resolve()
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (args.output.resolve(), observed_path)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1",
                        "FORTML_BENCH_CLASSIFIER_CHAIN_ORACLE": str(observed_path)})
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    completed = subprocess.run(["fo", "exec", "--no-build", args.target], cwd=fortml,
                               env=environment, check=True, capture_output=True, text=True)
    timing, parameters, observed_probabilities, observed_predicted, \
        observed_theta_hvp, observed_x_hvp = parse(completed.stdout, observed_path)
    x, labels = fixture()
    expected_probabilities, expected_predicted = oracle(x, parameters)
    probability_error = float(np.max(np.abs(observed_probabilities - expected_probabilities)))
    prediction_error = float(np.max(np.abs(observed_predicted - expected_predicted)))
    if probability_error > 5e-14 or prediction_error != 0.0:
        raise RuntimeError(
            f"classifier-chain oracle mismatch: probability={probability_error:.3e}, "
            f"prediction={prediction_error:.3e}"
        )
    x_dot, probabilities_bar, theta_dot = directions()
    h = 1.0e-5
    theta_bar_plus, x_bar_plus = vjp(
        x + h * x_dot, parameters + h * theta_dot, probabilities_bar
    )
    theta_bar_minus, x_bar_minus = vjp(
        x - h * x_dot, parameters - h * theta_dot, probabilities_bar
    )
    expected_theta_hvp = (theta_bar_plus - theta_bar_minus) / (2.0 * h)
    expected_x_hvp = (x_bar_plus - x_bar_minus) / (2.0 * h)
    theta_hvp_error = float(np.max(np.abs(observed_theta_hvp - expected_theta_hvp)))
    x_hvp_error = float(np.max(np.abs(observed_x_hvp - expected_x_hvp)))
    if theta_hvp_error > 5e-7 or x_hvp_error > 5e-7:
        raise RuntimeError(
            f"classifier-chain HVP oracle mismatch: theta={theta_hvp_error:.3e}, "
            f"x={x_hvp_error:.3e}"
        )
    records = [
        row(details, workload="classifier_chain", phase="fit", backend="fortml",
            device="cpu", status="pass", n_samples=N_SAMPLES, n_features=N_FEATURES,
            n_outputs=N_OUTPUTS, seconds_per_operation=timing["classifier_chain_fit"],
            metric="oracle_max_abs_error", value=probability_error,
            max_abs_error=probability_error, oracle="independent NumPy sigmoid-chain replay",
            notes="packed fitted heads replay exactly"),
        row(details, workload="classifier_chain", phase="predict", backend="fortml",
            device="cpu", status="pass", n_samples=N_SAMPLES, n_features=N_FEATURES,
            n_outputs=N_OUTPUTS, seconds_per_operation=timing["classifier_chain_predict"],
            metric="hard_label_max_abs_error", value=prediction_error,
            max_abs_error=prediction_error, oracle="independent NumPy threshold replay",
            notes="probabilities and integer labels match"),
        row(details, workload="classifier_chain", phase="hvp", backend="fortml",
            device="cpu", status="pass", n_samples=N_SAMPLES, n_features=N_FEATURES,
            n_outputs=N_OUTPUTS, seconds_per_operation=timing["classifier_chain_hvp"],
            metric="theta_hvp_max_abs_error", value=theta_hvp_error,
            max_abs_error=theta_hvp_error,
            oracle="independent NumPy finite-difference reverse oracle",
            notes=f"input_hvp_max_abs_error={x_hvp_error:.3e}"),
        row(details, workload="classifier_chain", phase="hvp_input", backend="fortml",
            device="cpu", status="pass", n_samples=N_SAMPLES, n_features=N_FEATURES,
            n_outputs=N_OUTPUTS, metric="x_hvp_max_abs_error", value=x_hvp_error,
            max_abs_error=x_hvp_error,
            oracle="independent NumPy finite-difference reverse oracle",
            notes="joint parameter/input direction"),
        row(details, workload="classifier_chain", phase="device_contract", backend="fortml",
            device="cuda", status="unavailable", n_samples=N_SAMPLES, n_features=N_FEATURES,
            n_outputs=N_OUTPUTS, metric="api_surface", value="unavailable", max_abs_error=0.0,
            oracle="device capability contract",
            notes="classifier-chain CUDA path is a typed refusal"),
        row(details, workload="classifier_chain", phase="hvp_device_contract", backend="fortml",
            device="cuda", status="unavailable", n_samples=N_SAMPLES,
            n_features=N_FEATURES, n_outputs=N_OUTPUTS, metric="api_surface",
            value="unavailable", max_abs_error=0.0,
            oracle="device capability contract",
            notes="classifier-chain HVP CUDA path is a typed refusal"),
    ]
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    observed_path.unlink(missing_ok=True)
    print(f"wrote {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()
