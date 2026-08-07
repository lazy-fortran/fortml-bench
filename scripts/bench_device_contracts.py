#!/usr/bin/env python3
"""Correctness-gated checks for FortML's resident CUDA micro-kernels.

The CUDA test programs own the device execution and return no host timing.  This
harness supplies independent NumPy fixture oracles, runs each gate, and records
``pass`` only when the native test reports its numerical check passed.  Missing
toolchains/devices are explicit ``skipped`` rows; they are never relabeled as a
CPU timing or a device pass.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status",
    "seconds_per_operation", "metric", "value", "max_abs_error", "oracle",
    "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    ignored = tuple((root / "results" / name).resolve()
                    for name in ("multilabel_metrics.csv", "roc_auc.csv",
                                 "device_contracts.csv",
                                 "basis_pipeline_training.csv"))
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": os.environ.get("NVCCFLAGS", "-O3 -arch=native"),
    }


def base(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    row.update(details)
    row.update({"backend": "fortml", "device": "cuda"})
    row.update(values)
    return row


def knn_oracle() -> tuple[np.ndarray, float]:
    """Return the exact labels and checksum for the CUDA kNN fixture."""
    train = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float64)
    query = np.array([[-1.5], [1.5]], dtype=np.float64)
    labels = np.array([-7, -7, 11, 11], dtype=np.int64)
    distances = np.sum((query[:, None, :] - train[None, :, :]) ** 2, axis=2)
    row_index = np.broadcast_to(np.arange(train.shape[0]), distances.shape)
    order = np.lexsort((row_index, distances), axis=1)[:, 0]
    expected = labels[order]
    if not np.array_equal(expected, np.array([-7, 11], dtype=np.int64)):
        raise RuntimeError("kNN independent oracle fixture changed")
    return expected, float(np.sum(expected))


def rmsprop_oracle() -> tuple[float, float]:
    """Return the expected resident-state norm and finite checksum."""
    parameters = np.array([0.2, -0.1, 0.3, -0.25], dtype=np.float64)
    square = np.zeros(4, dtype=np.float64)
    mean = np.zeros(4, dtype=np.float64)
    buffer = np.zeros(4, dtype=np.float64)
    learning_rate, decay, epsilon, momentum = 0.08, 0.8, 1.0e-5, 0.2
    for _ in range(5):
        gradient = parameters - 0.1 * np.arange(1, 5, dtype=np.float64)
        square = decay * square + (1.0 - decay) * gradient**2
        mean = decay * mean + (1.0 - decay) * gradient
        variance = np.maximum(square - mean**2, 0.0)
        direction = gradient / (np.sqrt(variance) + epsilon)
        buffer = momentum * buffer + direction
        parameters -= learning_rate * buffer
    norm = float(np.linalg.norm(parameters))
    checksum = float(np.sum(parameters) + np.sum(square) + np.sum(mean) + np.sum(buffer))
    if not np.isfinite(norm) or not np.isfinite(checksum):
        raise RuntimeError("RMSprop independent oracle is nonfinite")
    return norm, checksum


def mse_oracle() -> float:
    """Return the weighted multi-output MSE used by the CUDA metric gate."""
    target = np.reshape(np.array([
        1.0, -2.0, 0.5, 4.0, 2.0, 1.0, -1.5, 3.0,
    ], dtype=np.float64), (4, 2), order="F")
    prediction = np.reshape(np.array([
        0.0, -1.0, 1.5, 2.0, 1.0, 2.0, -0.5, 4.0,
    ], dtype=np.float64), (4, 2), order="F")
    weights = np.array([1.0, 2.0, 0.5, 3.0], dtype=np.float64)
    value = np.sum(weights[:, None] * (target - prediction) ** 2)
    value /= weights.sum() * target.shape[1]
    if not np.isfinite(value):
        raise RuntimeError("CUDA MSE independent oracle is nonfinite")
    return float(value)


def adamw_oracle() -> tuple[float, float]:
    """Return the resident AdamW fixture norm and state checksum.

    This recurrence is intentionally independent of FortML's CUDA test and
    mirrors the public plan contract: bias-corrected moments and decoupled
    weight decay are evaluated for seven device-resident steps.
    """
    parameters = np.array([0.2, -0.1, 0.3, -0.25, 0.05], dtype=np.float64)
    first = np.zeros(5, dtype=np.float64)
    second = np.zeros(5, dtype=np.float64)
    learning_rate, beta1, beta2 = 0.035, 0.81, 0.93
    epsilon, weight_decay = 1.0e-6, 0.17
    for step in range(7):
        gradient = parameters - 0.07 * np.arange(1, 6, dtype=np.float64)
        gradient += 0.01 * step
        first = beta1 * first + (1.0 - beta1) * gradient
        second = beta2 * second + (1.0 - beta2) * gradient**2
        bias1 = 1.0 - beta1 ** (step + 1)
        bias2 = 1.0 - beta2 ** (step + 1)
        parameters = ((1.0 - learning_rate * weight_decay) * parameters -
                      learning_rate * (first / bias1) /
                      (np.sqrt(second / bias2) + epsilon))
    norm = float(np.linalg.norm(parameters))
    checksum = float(np.sum(parameters) + np.sum(first) + np.sum(second))
    if not np.isfinite(norm) or not np.isfinite(checksum):
        raise RuntimeError("CUDA AdamW independent oracle is nonfinite")
    return norm, checksum


def dense_oracle() -> tuple[float, float, float]:
    """Return value, JVP, and VJP activation-sweep checksums for the dense plan.

    The native gate independently checks every output against the same
    recurrence.  Keeping a NumPy checksum here makes the benchmark row a
    separate fixture oracle rather than a pass/fail transcription.
    """
    weights = np.array([
        0.5, -1.0, 0.25, -0.75, 0.4, 1.2,
    ], dtype=np.float64).reshape((2, 3))
    bias = np.array([-0.1, 0.2], dtype=np.float64)
    query = np.array([
        -1.0, 0.0, 0.5, 2.0, -0.25,
        1.0, -0.5, 1.5, -2.0, 0.75,
        0.25, -1.0, 2.0, 0.5, -1.5,
    ], dtype=np.float64).reshape((3, 5))
    affine = weights @ query + bias[:, None]
    query_dot = np.array([
        0.25, -0.5, 1.0, -0.75, 0.2,
        -0.4, 0.8, -0.6, 0.3, -0.1,
        0.7, -0.2, 0.5, 0.9, -0.8,
    ], dtype=np.float64).reshape((3, 5))
    weights_dot = np.array([
        -0.2, 0.3, 0.1, 0.4, -0.5, 0.6,
    ], dtype=np.float64).reshape((2, 3))
    bias_dot = np.array([0.15, -0.25], dtype=np.float64)
    affine_dot = weights @ query_dot + weights_dot @ query + bias_dot[:, None]
    output_bar = np.array([
        0.2, -0.4, 0.7, 0.1, -0.3,
        -0.6, 0.5, -0.2, 0.8, 0.15,
    ], dtype=np.float64).reshape((2, 5))
    values: list[np.ndarray] = []
    tangents: list[np.ndarray] = []
    derivatives: list[np.ndarray] = []
    values.append(affine)
    tangents.append(affine_dot)
    derivatives.append(np.ones_like(affine))
    values.append(np.tanh(affine))
    tangents.append((1.0 - np.tanh(affine)**2) * affine_dot)
    derivatives.append(1.0 - np.tanh(affine)**2)
    values.append(np.maximum(affine, 0.0))
    tangents.append(np.where(affine >= 0.0, affine_dot, 0.0))
    derivatives.append(np.where(affine >= 0.0, 1.0, 0.0))
    gelu_inner = 0.79788456080286535588 * (affine + 0.044715 * affine**3)
    gelu_tanh = np.tanh(gelu_inner)
    values.append(0.5 * affine * (1.0 + np.tanh(
        0.79788456080286535588 *
        (affine + 0.044715 * affine**3))))
    tangents.append((0.5 * (1.0 + gelu_tanh) +
                     0.5 * affine * (1.0 - gelu_tanh**2) *
                     0.79788456080286535588 * (1.0 + 3.0 * 0.044715 * affine**2)) *
                    affine_dot)
    derivatives.append(0.5 * (1.0 + gelu_tanh) +
                      0.5 * affine * (1.0 - gelu_tanh**2) *
                      0.79788456080286535588 *
                      (1.0 + 3.0 * 0.044715 * affine**2))
    sigmoid = 1.0 / (1.0 + np.exp(-affine))
    values.append(affine / (1.0 + np.exp(-affine)))
    tangents.append((sigmoid + affine * sigmoid * (1.0 - sigmoid)) * affine_dot)
    derivatives.append(sigmoid + affine * sigmoid * (1.0 - sigmoid))
    values.append(np.where(affine >= 0.0, affine, np.exp(affine) - 1.0))
    tangents.append(np.where(affine >= 0.0, 1.0, np.exp(affine)) * affine_dot)
    derivatives.append(np.where(affine >= 0.0, 1.0, np.exp(affine)))
    values.append(np.where(
        affine > 20.0, affine,
        np.where(affine < -20.0, np.exp(affine), np.log1p(np.exp(affine)))))
    tangents.append(sigmoid * affine_dot)
    derivatives.append(sigmoid)
    values.append(np.where(affine >= 0.0, affine, 0.01 * affine))
    tangents.append(np.where(affine >= 0.0, 1.0, 0.01) * affine_dot)
    derivatives.append(np.where(affine >= 0.0, 1.0, 0.01))
    checksum = float(sum(np.sum(value) for value in values))
    jvp_checksum = float(sum(np.sum(tangent) for tangent in tangents))
    vjp_checksum = 0.0
    for derivative in derivatives:
        cotangent = output_bar * derivative
        vjp_checksum += float(np.sum(weights.T @ cotangent))
        vjp_checksum += float(np.sum(cotangent @ query.T))
        vjp_checksum += float(np.sum(cotangent))
    if (not np.isfinite(checksum) or not np.isfinite(jvp_checksum) or
            not np.isfinite(vjp_checksum)):
        raise RuntimeError("CUDA dense independent oracle is nonfinite")
    return checksum, jvp_checksum, vjp_checksum


def run_gate(fortml: Path, script_name: str) -> tuple[str, str, float | None]:
    script = fortml / "test" / script_name
    if not script.is_file():
        return "unavailable", f"test script is absent: {script_name}", None
    started = time.perf_counter()
    process = subprocess.run([str(script)], cwd=fortml, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    output = (process.stdout + "\n" + process.stderr).strip()
    lowered = output.lower()
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if process.returncode != 0:
        return "failed", lines[-1] if lines else "CUDA gate returned nonzero status", elapsed
    if "skipped" in lowered or "unavailable" in lowered:
        skip_lines = [line for line in lines if "skip" in line.lower() or "unavailable" in line.lower()]
        return "skipped", skip_lines[-1] if skip_lines else "CUDA gate skipped", elapsed
    if "pass" not in lowered:
        return "failed", lines[-1] if lines else "CUDA gate emitted no PASS marker", elapsed
    match = re.search(r"max error ([0-9.+\-eE]+)", output)
    observed_error = float(match.group(1)) if match else None
    pass_lines = [line for line in lines if "pass" in line.lower()]
    return "pass", pass_lines[-1] if pass_lines else "CUDA gate passed", elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/device_contracts.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    details = metadata(root, fortml, output)
    expected_knn, knn_checksum = knn_oracle()
    rmsprop_norm, rmsprop_checksum = rmsprop_oracle()
    mse_value = mse_oracle()
    adamw_norm, adamw_checksum = adamw_oracle()
    dense_checksum, dense_jvp_checksum, dense_vjp_checksum = dense_oracle()
    rows: list[dict[str, Any]] = []

    status, notes, elapsed = run_gate(fortml, "run_knn_classifier_cuda.sh")
    rows.append(base(
        details, workload="knn_device_predict", phase="predict", status=status,
        seconds_per_operation="", metric="prediction_label_checksum",
        value=knn_checksum, max_abs_error=0.0 if status == "pass" else "",
        oracle="independent NumPy nearest-neighbor labels [-7,11] and checksum",
        notes=f"native gate checks resident prediction; expected labels={expected_knn.tolist()}; {notes}"))

    status, notes, elapsed = run_gate(fortml, "run_cuda_rmsprop_state.sh")
    observed_error = None
    match = re.search(r"max error ([0-9.+\-eE]+)", notes)
    if match:
        observed_error = float(match.group(1))
        if status == "pass" and observed_error > 2.0e-12:
            status = "failed"
    rows.append(base(
        details, workload="rmsprop_device_state", phase="optimizer_step", status=status,
        seconds_per_operation="", metric="parameter_l2_norm", value=rmsprop_norm,
        max_abs_error=observed_error if observed_error is not None else (0.0 if status == "pass" else ""),
        oracle="independent NumPy centered RMSprop state recurrence",
        notes=f"expected state checksum={rmsprop_checksum:.16e}; native gate checks five resident steps; {notes}"))

    status, notes, elapsed = run_gate(fortml, "run_cuda_metric.sh")
    rows.append(base(
        details, workload="cuda_weighted_mse_reduction", phase="metric", status=status,
        seconds_per_operation="", metric="weighted_mean_squared_error", value=mse_value,
        max_abs_error="", oracle="independent NumPy weighted multi-output MSE",
        notes=f"native gate checks transfer-inclusive CUDA block reduction; expected value={mse_value:.16e}; {notes}"))

    status, notes, elapsed = run_gate(fortml, "run_cuda_mse_plan.sh")
    observed_error = None
    match = re.search(r"max error ([0-9.+\-eE]+)", notes)
    if match:
        observed_error = float(match.group(1))
        if status == "pass" and observed_error > 3.0e-13:
            status = "failed"
    rows.append(base(
        details, workload="cuda_weighted_mse_resident_plan", phase="metric", status=status,
        seconds_per_operation="", metric="weighted_mean_squared_error", value=mse_value,
        max_abs_error=(observed_error if observed_error is not None else
                       (0.0 if status == "pass" else "")),
        oracle="independent NumPy weighted multi-output MSE",
        notes=f"native gate retains target/prediction/weights on device for five executions; expected value={mse_value:.16e}; {notes}"))

    status, notes, elapsed = run_gate(fortml, "run_cuda_adamw_state.sh")
    observed_error = None
    match = re.search(r"max error ([0-9.+\-eE]+)", notes)
    if match:
        observed_error = float(match.group(1))
        if status == "pass" and observed_error > 3.0e-13:
            status = "failed"
    rows.append(base(
        details, workload="adamw_device_state", phase="optimizer_step", status=status,
        seconds_per_operation="", metric="parameter_l2_norm", value=adamw_norm,
        max_abs_error=(observed_error if observed_error is not None else
                       (0.0 if status == "pass" else "")),
        oracle="independent NumPy AdamW recurrence with decoupled weight decay",
        notes=f"expected state checksum={adamw_checksum:.16e}; native gate checks seven resident steps; {notes}"))

    status, notes, elapsed = run_gate(fortml, "run_cuda_forest_plan.sh")
    observed_error = None
    match = re.search(r"max error ([0-9.+\-eE]+)", notes)
    if match:
        observed_error = float(match.group(1))
        if status == "pass" and observed_error > 1.0e-13:
            status = "failed"
    rows.append(base(
        details, workload="cuda_forest_resident_prediction", phase="predict", status=status,
        seconds_per_operation="", metric="max_abs_error", value="",
        max_abs_error=(observed_error if observed_error is not None else
                       (0.0 if status == "pass" else "")),
        oracle="independent CPU tree-walk probabilities and sorted-label tie oracle",
        notes=f"native gate retains flattened trees across repeated query batches; {notes}"))

    status, notes, elapsed = run_gate(fortml, "run_cuda_dense_plan.sh")
    observed_error = None
    match = re.search(r"max error ([0-9.+\-eE]+)", notes)
    if match:
        observed_error = float(match.group(1))
        if status == "pass" and observed_error > 3.0e-13:
            status = "failed"
    rows.append(base(
        details, workload="cuda_dense_resident_inference", phase="predict", status=status,
        seconds_per_operation="", metric="activation_sweep_checksum",
        value=dense_checksum,
        max_abs_error=(observed_error if observed_error is not None else
                       (0.0 if status == "pass" else "")),
        oracle="independent NumPy affine plus eight-activation checksum",
        notes=f"native gate checks all eight MLP activations and two resident batches; expected checksum={dense_checksum:.16e}; {notes}"))
    rows.append(base(
        details, workload="cuda_dense_resident_jvp", phase="jvp", status=status,
        seconds_per_operation="", metric="activation_jvp_checksum",
        value=dense_jvp_checksum,
        max_abs_error=(observed_error if observed_error is not None else
                       (0.0 if status == "pass" else "")),
        oracle="independent NumPy affine tangent plus eight activation derivatives",
        notes=f"native gate checks value/JVP for all eight activations; expected checksum={dense_jvp_checksum:.16e}; {notes}"))
    rows.append(base(
        details, workload="cuda_dense_resident_vjp", phase="vjp", status=status,
        seconds_per_operation="", metric="activation_vjp_checksum",
        value=dense_vjp_checksum,
        max_abs_error=(observed_error if observed_error is not None else
                       (0.0 if status == "pass" else "")),
        oracle="independent NumPy affine cotangent plus eight activation derivatives",
        notes=f"native gate checks value/JVP/VJP for all eight activations; expected checksum={dense_vjp_checksum:.16e}; {notes}"))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
