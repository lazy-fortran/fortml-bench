#!/usr/bin/env python3
"""Correctness-gated benchmark for FortML pairwise contrastive products.

The NumPy implementation below is deliberately independent of FortML.  It
checks the release app's value, JVP, VJP, and HVP checksums before retaining
timings and records the typed CUDA capability boundary.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "dimensions",
    "repetitions", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
N, D, REPETITIONS, MARGIN = 128, 16, 512, 1.25
EPS = 2.0e-12


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    ignored_names = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, ...]:
    indices = np.arange(1, N + 1, dtype=np.float64)[:, None]
    coordinates = np.arange(1, D + 1, dtype=np.float64)[None, :]
    a = np.sin(0.017 * indices * coordinates)
    b = np.cos(0.013 * (indices * coordinates + 2.0))
    a_dot = 0.03 * np.sin(0.011 * (indices + 2.0 * coordinates))
    b_dot = -0.02 * np.cos(0.009 * (2.0 * indices + coordinates))
    labels = (indices[:, 0].astype(int) % 3 == 0).astype(int)
    weights = 0.5 + (np.arange(1, N + 1) % 7) / 7.0
    return a, b, a_dot, b_dot, labels, weights


def oracle() -> dict[str, float]:
    a, b, a_dot, b_dot, labels, weights = fixture()
    difference = a - b
    distance = np.linalg.norm(difference, axis=1)
    pair_value = np.where(
        labels == 1, 0.5 * distance**2,
        0.5 * np.maximum(0.0, MARGIN - distance)**2,
    )
    value = float(np.dot(weights, pair_value) / weights.sum())
    distance_dot = np.sum(difference * (a_dot - b_dot), axis=1) / distance
    radial_derivative = np.where(
        labels == 1, distance,
        np.where(distance < MARGIN, distance - MARGIN, 0.0),
    )
    value_dot = float(np.dot(weights, radial_derivative * distance_dot) / weights.sum())
    gradient = (weights * radial_derivative / distance / weights.sum())[:, None] * difference
    vjp_sum = float(0.7 * gradient.sum())
    radial_curvature = np.where(
        labels == 1, 0.0,
        np.where(distance < MARGIN, MARGIN / distance**2, 0.0),
    )
    a_hvp = (weights / weights.sum())[:, None] * (
        (radial_derivative / distance)[:, None] * (a_dot - b_dot)
        + radial_curvature[:, None] * difference * distance_dot[:, None]
    )
    b_hvp = -a_hvp
    return {
        "value": value,
        "jvp": value_dot,
        "vjp": vjp_sum,
        "hvp": float((a_hvp + b_hvp).sum()),
    }


def row(details: dict[str, str], **updates: object) -> dict[str, str]:
    output = {field: "" for field in FIELDS}
    output.update(details)
    output.update({key: str(value) for key, value in updates.items()})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/contrastive_loss.csv"))
    parser.add_argument("--target", default="fortml_bench_contrastive_loss")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    expected = oracle()
    completed = subprocess.run(
        ["fo", "exec", "--no-build", args.target], cwd=fortml,
        check=True, capture_output=True, text=True,
    )
    actual: dict[str, tuple[float, float]] = {}
    cuda_status = None
    for line in completed.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) == 5 and fields[0] == "contrastive_loss":
            if fields[3] == "refused":
                cuda_status = fields[4]
            else:
                actual[fields[1]] = (float(fields[3]), float(fields[4]))
    if set(actual) != set(expected):
        raise RuntimeError(f"missing contrastive release rows: {sorted(actual)}")
    errors = {name: abs(actual[name][1] - expected[name]) for name in expected}
    if max(errors.values()) > EPS:
        raise RuntimeError(f"contrastive checksum mismatch: {max(errors.values()):.3e}")
    if cuda_status is None:
        raise RuntimeError("missing contrastive CUDA capability row")

    details = {
        "workload": "contrastive_loss", "backend": "fortml", "device": "cpu",
        "status": "pass", "dimensions": f"{N}x{D} paired embeddings",
        "repetitions": str(REPETITIONS),
        "oracle": "independent NumPy Euclidean contrastive formula",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (args.output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
        "notes": "weighted pair value/JVP/VJP/HVP; zero-distance and margin-kink products are typed refusals",
    }
    rows = []
    for phase, (seconds, checksum) in actual.items():
        rows.append(row(details, phase=phase,
                        seconds_per_operation=f"{seconds:.17e}",
                        metric="seconds_per_operation", value=f"{checksum:.17e}",
                        max_abs_error=f"{errors[phase]:.17e}"))
    rows.append(row(details, phase="device_capability", device="cuda",
                    status="unavailable", repetitions="", seconds_per_operation="",
                    metric="", value="", max_abs_error="",
                    oracle="typed_device_contract",
                    notes=f"contrastive value CUDA refusal status {cuda_status}; no host fallback"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
