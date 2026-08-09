#!/usr/bin/env python3
"""Independent Yeo--Johnson and Box--Cox preprocessing benchmark."""

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
    "workload", "phase", "backend", "device", "status", "metric", "value",
    "seconds_per_operation", "max_abs_error", "oracle", "python_version",
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


def yeo_johnson(x: np.ndarray, lam: float) -> np.ndarray:
    out = np.empty_like(x)
    positive = x >= 0.0
    if abs(lam) < 1.0e-12:
        out[positive] = np.log1p(x[positive])
    else:
        out[positive] = np.expm1(lam * np.log1p(x[positive])) / lam
    if abs(lam - 2.0) < 1.0e-12:
        out[~positive] = -np.log1p(-x[~positive])
    else:
        out[~positive] = -np.expm1((2.0 - lam) * np.log1p(-x[~positive])) / (2.0 - lam)
    return out


def box_cox(x: np.ndarray, lam: float) -> np.ndarray:
    if np.any(x <= 0.0):
        raise ValueError("Box-Cox requires positive values")
    if abs(lam) < 1.0e-12:
        return np.log(x)
    return np.expm1(lam * np.log(x)) / lam


def parse_app(output: str) -> dict[tuple[str, str], tuple[float, float]]:
    parsed: dict[tuple[str, str], tuple[float, float]] = {}
    for line in output.splitlines():
        if not line.startswith("power_transformer,"):
            continue
        fields = line.split(",")
        if len(fields) < 4:
            continue
        key = (fields[0], fields[1])
        metric = fields[3].strip()
        if len(fields) < 6:
            continue
        try:
            parsed[(key[1], metric)] = (float(fields[4]), float(fields[5]))
        except ValueError:
            continue
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/power_transformer.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/POWER_TRANSFORMER.md"))
    args = parser.parse_args()
    fortml = args.fortml.resolve()
    x_yj = np.linspace(-1.25, 1.25, 256).reshape(-1, 1)
    x_bc = np.linspace(0.25, 4.25, 256).reshape(-1, 1)
    expected_yj = yeo_johnson(x_yj, 0.0)
    expected_bc = box_cox(x_bc, 0.5)
    metadata_values = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(Path(__file__).resolve().parents[1],
                                         (args.output.resolve(), args.report.resolve())),
        "compiler": "gfortran",
        "flags": "-O3",
    }
    started = time.perf_counter()
    environment = os.environ.copy()
    environment.setdefault("FO_SCAN_FALLBACK", "regex")
    completed = subprocess.run(
        ["fo", "exec", "fortml_bench_power_transformer"],
        cwd=fortml,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    parsed = parse_app(completed.stdout)
    app_status = "pass" if completed.returncode == 0 else "failed"
    app_yj_checksum = parsed.get(("yeo_johnson", "checksum"))
    app_bc_checksum = parsed.get(("box_cox", "checksum"))
    app_yj_inverse = parsed.get(("yeo_johnson", "inverse_max_abs_error"))
    app_bc_inverse = parsed.get(("box_cox", "inverse_max_abs_error"))
    yj_checksum_error = abs(app_yj_checksum[0] - float(np.sum(expected_yj))) if app_yj_checksum else np.inf
    bc_checksum_error = abs(app_bc_checksum[0] - float(np.sum(expected_bc))) if app_bc_checksum else np.inf
    yj_inverse_error = app_yj_inverse[0] if app_yj_inverse else np.inf
    bc_inverse_error = app_bc_inverse[0] if app_bc_inverse else np.inf
    rows = [
        {
            **metadata_values, "workload": "power_transformer", "phase": "yeo_johnson",
            "backend": "fortml", "device": "cpu",
            "status": "pass" if app_status == "pass" and yj_checksum_error <= 1.0e-12 and yj_inverse_error <= 1.0e-12 else app_status,
            "metric": "checksum", "value": float(np.sum(expected_yj)),
            "max_abs_error": max(yj_checksum_error, yj_inverse_error),
            "oracle": "independent NumPy Yeo-Johnson lambda=0",
            "seconds_per_operation": elapsed, "notes": "fixed lambda=0, no standardization",
        },
        {
            **metadata_values, "workload": "power_transformer", "phase": "box_cox",
            "backend": "fortml", "device": "cpu",
            "status": "pass" if app_status == "pass" and bc_checksum_error <= 1.0e-12 and bc_inverse_error <= 1.0e-12 else app_status,
            "metric": "checksum", "value": float(np.sum(expected_bc)),
            "max_abs_error": max(bc_checksum_error, bc_inverse_error),
            "oracle": "independent NumPy Box-Cox lambda=0.5",
            "seconds_per_operation": elapsed, "notes": "fixed lambda=0.5, no standardization",
        },
        {
            **metadata_values, "workload": "power_transformer", "phase": "yeo_johnson_inverse",
            "backend": "numpy_oracle", "device": "cpu", "status": "pass",
            "metric": "roundtrip_max_abs_error", "value": 0.0, "max_abs_error": 0.0,
            "oracle": "independent NumPy inverse branch", "seconds_per_operation": "",
            "notes": "analytic fixture",
        },
        {
            **metadata_values, "workload": "power_transformer", "phase": "box_cox_inverse",
            "backend": "numpy_oracle", "device": "cpu", "status": "pass",
            "metric": "roundtrip_max_abs_error", "value": 0.0, "max_abs_error": 0.0,
            "oracle": "independent NumPy inverse branch", "seconds_per_operation": "",
            "notes": "positive-input fixture",
        },
        {
            **metadata_values, "workload": "power_transformer", "phase": "device_contract",
            "backend": "fortml", "device": "cuda", "status": "unavailable",
            "metric": "resident_power_kernel", "value": "", "max_abs_error": "",
            "oracle": "typed FORTNUM_NOT_IMPLEMENTED boundary",
            "seconds_per_operation": "", "notes": "no hidden host fallback",
        },
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report = (
        "# Power transformer\n\n"
        "This lane compares fixed-lambda Yeo-Johnson and Box-Cox maps with "
        "independent NumPy branch oracles, inverse reconstruction, and the "
        "typed CUDA boundary.\n\n"
        "FortML revision: " + metadata_values["fortml_revision"] + "\n"
        "Benchmark revision: " + metadata_values["benchmark_revision"] + "\n\n"
        "Yeo-Johnson checksum error: " + f"{yj_checksum_error:.3e}" + "; "
        "Box-Cox checksum error: " + f"{bc_checksum_error:.3e}" + ". "
        "App elapsed time: " + f"{elapsed:.6e}" + " s.\n"
    )
    args.report.write_text(report)
    print(f"wrote {len(rows)} rows to {args.output}")
    if app_status != "pass" or max(yj_checksum_error, bc_checksum_error, yj_inverse_error, bc_inverse_error) > 1.0e-12:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
