#!/usr/bin/env python3
"""Correctness-gated Huber and quantile XGBoost-style benchmarks."""

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
    "workload", "objective", "phase", "backend", "device", "status",
    "n_samples", "n_features", "seconds_per_operation", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray]:
    return (np.arange(4, dtype=np.float64)[:, None],
            np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float64))


def row(details: dict[str, str], **values: object) -> dict[str, object]:
    output: dict[str, object] = {field: "" for field in FIELDS}
    output.update(details)
    output.update(values)
    return output


def oracle() -> dict[str, np.ndarray]:
    # One depth-one Newton tree, L2=1, learning rate=1.  Huber delta=1 has
    # base mean 5 and leaf corrections -2/+2; alpha=.5 uses the lower
    # weighted median 0 and subgradient corrections -1/+1.
    return {
        "huber": np.array([3.0, 3.0, 7.0, 7.0]),
        "quantile": np.array([-1.0, -1.0, 1.0, 1.0]),
    }


def parse_output(stdout: str) -> dict[str, float | str]:
    values: dict[str, float | str] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2:
            key, value = fields
            if key == "xgb_robust_cuda":
                values[key] = value
            elif key.startswith("xgb_"):
                values[key] = float(value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/xgboost_robust.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (args.output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }
    x, target = fixture()
    expected = oracle()
    rows: list[dict[str, object]] = []
    for objective, prediction in expected.items():
        rows.append(row(details, workload="xgboost_robust", objective=objective,
                        phase="predict", backend="numpy_oracle", device="cpu",
                        status="pass", n_samples=x.shape[0], n_features=x.shape[1],
                        value=float(np.mean(prediction)), max_abs_error=0.0,
                        oracle="independent one-tree Newton Huber/pinball formula"))
    started = time.perf_counter()
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_xgboost_robust"],
        cwd=fortml, check=True, capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - started
    values = parse_output(completed.stdout)
    for objective in ("huber", "quantile"):
        error = float(values[f"xgb_{objective}_max_error"])
        if error > 1.0e-10:
            raise RuntimeError(f"{objective} oracle mismatch: {error:.3e}")
        rows.extend((
            row(details, workload="xgboost_robust", objective=objective,
                phase="fit", backend="fortml_cpu", device="cpu", status="pass",
                n_samples=x.shape[0], n_features=x.shape[1],
                seconds_per_operation=values[f"xgb_{objective}_fit_seconds"],
                value=float(np.mean(expected[objective])), max_abs_error=error,
                oracle="independent one-tree Newton Huber/pinball formula",
                notes="exact CPU tree fit; full subprocess elapsed %.6g s" % elapsed),
            row(details, workload="xgboost_robust", objective=objective,
                phase="predict", backend="fortml_cpu", device="cpu", status="pass",
                n_samples=x.shape[0], n_features=x.shape[1],
                seconds_per_operation=values[f"xgb_{objective}_predict_seconds"],
                value=float(np.mean(expected[objective])), max_abs_error=error,
                oracle="independent one-tree Newton Huber/pinball formula"),
        ))
    if values.get("xgb_robust_cuda") != "unavailable":
        raise RuntimeError("robust XGBoost CUDA refusal row missing")
    rows.append(row(details, workload="xgboost_robust", objective="all",
                    phase="predict", backend="fortml_cuda", device="cuda",
                    status="unavailable", n_samples=x.shape[0], n_features=x.shape[1],
                    oracle="typed_device_contract",
                    notes="no resident robust-tree CUDA kernel; no host fallback"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
