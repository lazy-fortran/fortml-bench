#!/usr/bin/env python3
"""Correctness-gated named PINN gradient/HVP diagnostics."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status",
    "parameter_count", "seconds_per_operation", "metric", "value",
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


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update({
        "workload": "pinn_term_products", "phase": "", "backend": "",
        "device": "cpu", "status": "", "parameter_count": 1,
        "seconds_per_operation": "", "metric": "", "value": "",
        "max_abs_error": "", "oracle": "", "notes": "",
    })
    result.update(values)
    return result


def independent_oracle() -> tuple[float, float, float]:
    theta = 0.7
    direction = -0.23
    weight = 2.0
    residual = theta * theta - 0.25
    gradient = weight * residual * 2.0 * theta
    hessian_direction = (weight * (4.0 * theta * theta + 2.0 * residual)
                         * direction)
    step = 1.0e-6
    def objective(point: float) -> float:
        residual_at_point = point * point - 0.25
        return weight * residual_at_point * residual_at_point / 2.0
    def objective_gradient(point: float) -> float:
        residual_at_point = point * point - 0.25
        return weight * residual_at_point * 2.0 * point
    fd_hvp = (objective_gradient(theta + step * direction) -
              objective_gradient(theta - step * direction)) / (2.0 * step)
    finite_error = abs(fd_hvp - hessian_direction)
    if finite_error > 1.0e-8 or abs(objective(theta) - 0.0576) > 1.0e-12:
        raise RuntimeError("independent PINN gradient/HVP oracle failed")
    return float(gradient), float(hessian_direction), float(finite_error)


def run_release(fortml: Path, details: dict[str, str]) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.setdefault("FO_SCAN_FALLBACK", "regex")
    started = time.perf_counter()
    result = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_pinn_term_products"],
        cwd=fortml, env=environment, capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - started
    output = (result.stdout + "\n" + result.stderr).strip()
    lines = [line for line in output.splitlines()
             if line.startswith("quadratic_residual,")]
    if result.returncode != 0 or not lines:
        note = output.splitlines()[-1] if output else "release app failed"
        return row(details, phase="release_app", backend="fortml",
                   status="failed", seconds_per_operation=elapsed,
                   metric="term_product_error", value="nan",
                   max_abs_error="nan",
                   oracle="fortml_bench_pinn_term_products", notes=note)
    fields = lines[-1].split(",")
    values = np.array([float(fields[index]) for index in (3, 4, 5, 6)])
    expected_gradient, expected_hvp, _ = independent_oracle()
    error = float(max(abs(values[0] - expected_gradient),
                      abs(values[1] - expected_hvp),
                      abs(values[2] - expected_gradient),
                      abs(values[3] - expected_hvp)))
    return row(details, phase="release_app", backend="fortml", status="pass",
               seconds_per_operation=elapsed, metric="term_product_error",
               value=error, max_abs_error=error,
               oracle="named weighted quadratic PINN gradient/HVP columns",
               notes="residual column plus inactive-term zero diagnostics")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/pinn_term_products.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": os.environ.get("FFLAGS", "-O3"),
    }
    gradient, hvp, oracle_error = independent_oracle()
    rows = [row(
        details, phase="independent_oracle", backend="numpy_oracle",
        status="pass", metric="term_product_error", value=oracle_error,
        max_abs_error=oracle_error,
        oracle="independent weighted quadratic residual formula",
        notes=f"gradient={gradient:.16e};hvp={hvp:.16e}",
    )]
    if args.skip_fortml:
        rows.append(row(details, phase="release_app", backend="fortml",
                        status="skipped", metric="term_product_error",
                        oracle="fortml_bench_pinn_term_products",
                        notes="--skip-fortml"))
    else:
        rows.append(run_release(fortml, details))
    rows.append(row(
        details, phase="device_contract", backend="fortml", device="cuda",
        status="unavailable", metric="resident_pinn_residual_graph",
        value="nan", max_abs_error="nan",
        oracle="typed FORTNUM_NOT_IMPLEMENTED boundary",
        notes="no host fallback or resident CUDA PINN graph is claimed",
    ))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
