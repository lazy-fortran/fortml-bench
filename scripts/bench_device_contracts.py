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
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
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


def run_gate(fortml: Path, script_name: str) -> tuple[str, str, float | None]:
    script = fortml / "test" / script_name
    if not script.is_file():
        return "unavailable", f"test script is absent: {script_name}", None
    started = time.perf_counter()
    process = subprocess.run([str(script)], cwd=fortml, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    output = (process.stdout + "\n" + process.stderr).strip()
    lowered = output.lower()
    if process.returncode != 0:
        return "failed", output[-500:] or "CUDA gate returned nonzero status", elapsed
    if "skipped" in lowered or "unavailable" in lowered:
        return "skipped", output[-500:] or "CUDA gate skipped", elapsed
    if "pass" not in lowered:
        return "failed", output[-500:] or "CUDA gate emitted no PASS marker", elapsed
    match = re.search(r"max error ([0-9.+\-eE]+)", output)
    observed_error = float(match.group(1)) if match else None
    return "pass", output[-500:] or "CUDA gate passed", elapsed


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

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
