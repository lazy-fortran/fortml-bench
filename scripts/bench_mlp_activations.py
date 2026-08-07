#!/usr/bin/env python3
"""Benchmark dense MLP activation kernels against an independent NumPy oracle.

The fixture exercises the same packed dense network with each supported smooth
or piecewise activation.  FortML's checksum is accepted only when it agrees
with the NumPy forward calculation; the CUDA rows remain explicit capability
records until the complete MLP forward path has resident kernels.
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


N_SAMPLES = 2048
N_FEATURES = 8
N_HIDDEN = 32
N_OUTPUTS = 4
REPETITIONS = 32
KINDS = ("linear", "tanh", "relu", "gelu", "silu", "elu", "softplus", "leaky_relu")
FIELDS = (
    "activation", "backend", "device", "status", "n_samples", "n_features",
    "n_hidden", "n_outputs", "repetitions", "seconds_per_operation", "checksum",
    "expected_checksum", "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines()
    dirty = [
        line for line in dirty
        if (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        not in ignored_paths
    ]
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(1, N_SAMPLES + 1, dtype=np.float64)[:, None]
    columns = np.arange(1, N_FEATURES + 1, dtype=np.float64)[None, :]
    x = np.sin(0.017 * rows + 0.13 * columns)
    x += 0.15 * np.cos(0.009 * rows * columns)
    parameters = 0.07 * np.sin(0.37 * np.arange(1, parameter_count() + 1))
    return x, parameters


def parameter_count() -> int:
    return N_FEATURES * N_HIDDEN + N_HIDDEN + N_HIDDEN * N_OUTPUTS + N_OUTPUTS


def unpack(parameters: np.ndarray) -> tuple[np.ndarray, ...]:
    position = 0
    count = N_FEATURES * N_HIDDEN
    weight_1 = parameters[position : position + count].reshape(
        (N_FEATURES, N_HIDDEN), order="F"
    )
    position += count
    bias_1 = parameters[position : position + N_HIDDEN]
    position += N_HIDDEN
    count = N_HIDDEN * N_OUTPUTS
    weight_2 = parameters[position : position + count].reshape(
        (N_HIDDEN, N_OUTPUTS), order="F"
    )
    position += count
    bias_2 = parameters[position : position + N_OUTPUTS]
    return weight_1, bias_1, weight_2, bias_2


def activation(values: np.ndarray, name: str) -> np.ndarray:
    if name == "linear":
        return values
    if name == "tanh":
        return np.tanh(values)
    if name == "relu":
        return np.maximum(values, 0.0)
    if name == "gelu":
        u = np.sqrt(2.0 / np.pi) * (values + 0.044715 * values**3)
        return 0.5 * values * (1.0 + np.tanh(u))
    if name == "silu":
        return values / (1.0 + np.exp(-values))
    if name == "elu":
        return np.where(values > 0.0, values, np.exp(values) - 1.0)
    if name == "softplus":
        return np.where(values > 0.0, values + np.log1p(np.exp(-values)),
                        np.log1p(np.exp(values)))
    if name == "leaky_relu":
        return np.where(values >= 0.0, values, 0.01 * values)
    raise ValueError(f"unknown activation {name}")


def oracle(name: str) -> float:
    x, parameters = fixture()
    weight_1, bias_1, weight_2, bias_2 = unpack(parameters)
    hidden = activation(x @ weight_1 + bias_1, name)
    prediction = hidden @ weight_2 + bias_2
    return float(np.sum(prediction))


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        **details,
        "activation": "", "backend": "", "device": "cpu", "status": "",
        "n_samples": N_SAMPLES, "n_features": N_FEATURES, "n_hidden": N_HIDDEN,
        "n_outputs": N_OUTPUTS, "repetitions": REPETITIONS,
        "seconds_per_operation": "", "checksum": "", "expected_checksum": "",
        "max_abs_error": "", "oracle": "", "notes": "",
    }
    result.update(values)
    return result


def parse_rows(stdout: str) -> dict[str, tuple[float, float]]:
    parsed: dict[str, tuple[float, float]] = {}
    pattern = re.compile(r"^([a-z_]+),(\d+),(\d+),(\d+),(\d+),(\d+),([^,]+),([^,]+)$")
    for line in stdout.splitlines():
        match = pattern.match(line.strip())
        if match:
            parsed[match.group(1)] = (float(match.group(7)), float(match.group(8)))
    return parsed


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = Path(__file__).resolve().parents[1]
    fortml = Path(args.fortml).resolve()
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, check=True)
    completed = subprocess.run(
        ["fo", "exec", "--no-build", args.target], cwd=fortml,
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    measured = parse_rows(completed.stdout)
    records: list[dict[str, Any]] = []
    for name in KINDS:
        expected = oracle(name)
        if name not in measured:
            records.append(row(details, activation=name, backend="fortml", status="unavailable",
                               expected_checksum=expected,
                               oracle="independent NumPy dense MLP activation checksum",
                               notes="release app emitted no activation row"))
            continue
        checksum, seconds = measured[name]
        error = abs(checksum - expected)
        records.append(row(details, activation=name, backend="numpy_oracle", status="pass",
                           repetitions=1, checksum=expected, expected_checksum=expected,
                           max_abs_error=0.0,
                           oracle="independent NumPy dense MLP activation checksum"))
        records.append(row(details, activation=name, backend="fortml", status="pass" if error <= 2.0e-11 else "fail",
                           checksum=checksum, expected_checksum=expected,
                           max_abs_error=error, seconds_per_operation=seconds,
                           oracle="independent NumPy dense MLP activation checksum",
                           notes="host resident MLP forward timing"))
        if error > 2.0e-11:
            raise RuntimeError(f"{name} checksum error {error} exceeds tolerance")
        records.append(row(details, activation=name, backend="fortml", device="cuda",
                           status="unavailable", expected_checksum=expected,
                           oracle="FortML device capability boundary",
                           notes="MLP activation forward path is host-only; no CPU timing relabeled as CUDA"))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", default="../fortml", type=Path)
    parser.add_argument("--target", default="fortml_bench_mlp_activations")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()
