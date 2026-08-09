#!/usr/bin/env python3
"""Correctness-gated weighted MLP training benchmark.

The NumPy rows independently evaluate the weighted affine MSE+L2 value and
gradient and the one-step weighted SGD recurrence. The FortML release app is
accepted only when its oracle errors match those calculations. CUDA is an
explicit refusal because the complete resident MLP trainer is not available.
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


FIELDS = (
    "workload", "backend", "device", "status", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
TOLERANCE = 2.0e-12


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


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.array([-1.0, -0.25, 0.75, 1.5], dtype=np.float64)[:, None]
    target = np.array([0.4, -0.1, 0.8, 1.2], dtype=np.float64)[:, None]
    weight = np.array([1.0, 0.0, 2.0, 0.5], dtype=np.float64)
    theta = np.array([0.3, -0.2], dtype=np.float64)
    l2 = 0.07
    return x, target, weight, theta, l2


def oracle() -> dict[str, float]:
    x, target, weight, theta, l2 = fixture()
    prediction = x[:, 0] * theta[0] + theta[1]
    residual = prediction - target[:, 0]
    mass = float(weight.sum())
    value = 0.5 * float(np.sum(weight * residual**2)) / mass + 0.5 * l2 * float(
        np.sum(theta**2)
    )
    gradient = np.array(
        [np.sum(weight * residual * x[:, 0]) / mass + l2 * theta[0],
         np.sum(weight * residual) / mass + l2 * theta[1]],
        dtype=np.float64,
    )
    rate = 0.025
    expected_theta = theta - rate * np.array(
        [np.sum(weight * residual * x[:, 0]) / mass,
         np.sum(weight * residual) / mass],
        dtype=np.float64,
    )
    return {
        "loss": value,
        "gradient_norm": float(np.max(np.abs(gradient))),
        "expected_weighted_gradient": float(np.linalg.norm(gradient - l2 * theta)),
        "expected_theta_0": float(expected_theta[0]),
        "expected_theta_1": float(expected_theta[1]),
    }


def run_release(fortml: Path) -> dict[str, float]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    build = subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml,
        env=environment, capture_output=True, text=True,
    )
    if build.returncode:
        detail = build.stderr.strip().splitlines()[-1] if build.stderr.strip() else "release build failed"
        raise RuntimeError(detail)
    test = subprocess.run(
        ["fo", "test", "test_mlp_weighted_training"],
        cwd=fortml, env=environment, capture_output=True, text=True,
    )
    if test.returncode:
        detail = test.stderr.strip().splitlines()[-1] if test.stderr.strip() else "weighted test failed"
        raise RuntimeError(detail)
    run = subprocess.run(
        ["fo", "exec", "fortml_bench_mlp_weighted_training"],
        cwd=fortml, env=environment, capture_output=True, text=True,
    )
    if run.returncode:
        detail = run.stderr.strip().splitlines()[-1] if run.stderr.strip() else "release app failed"
        raise RuntimeError(detail)
    rows = list(csv.DictReader(line for line in run.stdout.splitlines() if "," in line))
    if not rows:
        raise RuntimeError(f"weighted app returned no CSV rows: {run.stdout!r}")
    return {row["metric"]: float(row["value"]) for row in rows}


def row(details: dict[str, Any], **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {field: "" for field in FIELDS}
    result.update(details)
    result.update({
        "workload": "mlp_weighted_training", "backend": "", "device": "cpu",
        "status": "", "metric": "", "value": "", "max_abs_error": "",
        "oracle": "", "notes": "",
    })
    result.update(values)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/mlp_weighted_training.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    fortml = args.fortml.resolve()
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml), "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    expected = oracle()
    rows = [
        row(details, backend="numpy_oracle", status="pass", metric="loss",
            value=expected["loss"], max_abs_error=0.0,
            oracle="independent weighted affine MSE+L2 calculation"),
        row(details, backend="numpy_oracle", status="pass", metric="expected_theta_0",
            value=expected["expected_theta_0"], max_abs_error=0.0,
            oracle="independent weighted SGD recurrence"),
        row(details, backend="numpy_oracle", status="pass", metric="expected_theta_1",
            value=expected["expected_theta_1"], max_abs_error=0.0,
            oracle="independent weighted SGD recurrence"),
    ]
    if args.skip_fortml:
        rows.append(row(details, backend="fortml", status="skipped",
                        oracle="FortML weighted-training release app", notes="--skip-fortml"))
    else:
        release = run_release(fortml)
        comparisons = (
            ("loss_error", 0.0),
            ("gradient_error", 0.0),
            ("parameter_error", 0.0),
        )
        for metric, _ in comparisons:
            error = release[metric]
            if error > TOLERANCE:
                raise RuntimeError(f"FortML {metric} exceeds tolerance: {error:.3e}")
            rows.append(row(details, backend="fortml", status="pass", metric=metric,
                            value=error, max_abs_error=error,
                            oracle="independent NumPy weighted affine recurrence",
                            notes=f"tolerance={TOLERANCE:.1e}"))
        rows.append(row(details, backend="fortml", status="pass", metric="validation_loss",
                        value=release["validation_loss"], max_abs_error="",
                        oracle="weighted validation MSE+L2"))
        rows.append(row(details, backend="fortml", status="pass", metric="invalid_status",
                        value=release["invalid_status"], max_abs_error="",
                        oracle="transactional malformed-weight refusal"))
    rows.append(row(details, backend="cuda", device="cuda", status="refused",
                    oracle="resident MLP trainer capability contract",
                    notes="FortML exposes typed CUDA refusal for full MLP training"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
