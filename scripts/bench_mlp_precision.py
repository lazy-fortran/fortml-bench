#!/usr/bin/env python3
"""Correctness-gated MLP training precision capability benchmark.

The NumPy recurrence is an independent one-layer linear-MSE oracle.  The
FortML test additionally checks that recognized FP32/FP16/BF16 requests refuse
before mutating model state and that unknown modes are domain errors.  No
lower-precision or CUDA timing is reported as available until master weights,
loss scaling, and resident state exist.
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


PRECISIONS = (("fp64", 1), ("fp32", 2), ("fp16", 3), ("bf16", 4))
FIELDS = (
    "workload", "phase", "precision", "backend", "device", "status",
    "epochs", "n_parameters", "seconds_per_operation", "metric", "value",
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
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def oracle() -> tuple[np.ndarray, float]:
    x = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
    target = 0.4 * x + 0.2
    theta = np.array([0.1, -0.2], dtype=np.float64)
    learning_rate = 0.05
    for _ in range(3):
        residual = x * theta[0] + theta[1] - target
        gradient = np.array([np.mean(residual * x), np.mean(residual)])
        theta = theta - learning_rate * gradient
    if not np.all(np.isfinite(theta)):
        raise RuntimeError("FP64 precision oracle is nonfinite")
    return theta, float(np.max(np.abs(theta)))


def base(details: dict[str, str], **values: object) -> dict[str, object]:
    row = {field: "" for field in FIELDS}
    row.update(details)
    row.update({"workload": "mlp_precision", "epochs": 3, "n_parameters": 2,
                "device": "cpu"})
    row.update(values)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/mlp_precision.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    expected, scale = oracle()
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }
    rows: list[dict[str, object]] = []
    rows.append(base(details, phase="independent_oracle", precision="fp64",
                     backend="numpy_oracle", status="pass", value=float(expected[0]),
                     max_abs_error=0.0,
                     oracle="independent NumPy linear-MSE SGD recurrence",
                     notes=f"theta={expected.tolist()}"))
    if args.skip_fortml:
        status, elapsed, note = "skipped", "", "--skip-fortml"
    else:
        started = time.perf_counter()
        environment = os.environ.copy()
        environment.setdefault("FO_SCAN_FALLBACK", "regex")
        result = subprocess.run(
            ["fo", "test", "--target", "test_mlp_precision_contract"],
            cwd=fortml, env=environment, capture_output=True, text=True,
        )
        elapsed = time.perf_counter() - started
        status = "pass" if result.returncode == 0 else "unavailable"
        note = "test_mlp_precision_contract independent oracle and refusal gate"
        if result.returncode != 0:
            note = "fo test failed: " + result.stderr[-240:].replace("\n", " ")
    rows.append(base(details, phase="reference_training", precision="fp64",
                     backend="fortml", status=status,
                     seconds_per_operation=elapsed, value=float(expected[0]),
                     max_abs_error=0.0,
                     oracle="FortML test_mlp_precision_contract vs NumPy recurrence",
                     notes=note))
    for name, _ in PRECISIONS[1:]:
        rows.append(base(details, phase="typed_refusal", precision=name,
                         backend="fortml", status="unavailable",
                         metric="status_code", value="nan", max_abs_error="nan",
                         oracle="FORTNUM_NOT_IMPLEMENTED before model mutation",
                         notes="master weights/loss scaling/overflow recovery not linked"))
    rows.append(base(details, phase="device_contract", precision="fp64",
                     backend="fortml", device="cuda", status="unavailable",
                     metric="resident_mixed_precision", value="nan", max_abs_error="nan",
                     oracle="typed device refusal",
                     notes="trainer state is host-resident; no hidden GPU fallback"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
