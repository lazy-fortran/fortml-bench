#!/usr/bin/env python3
"""Correctness-gated Chebyshev basis value/derivative benchmark.

The NumPy path independently evaluates the first-kind recurrence and its first
two input derivatives.  The Fortran application reports the same reductions;
the CUDA row is an explicit typed-refusal record because the basis has no
resident device lowering yet.
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
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_inputs", "degree", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
N_SAMPLES = 4096
N_INPUTS = 3
DEGREE = 8


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


def rows_base(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update({
        "workload": "chebyshev_basis", "phase": "", "backend": "",
        "device": "cpu", "status": "", "n_samples": N_SAMPLES,
        "n_inputs": N_INPUTS, "degree": DEGREE, "seconds_per_operation": "",
        "metric": "", "value": "", "max_abs_error": "", "oracle": "",
        "notes": "",
    })
    result.update(values)
    return result


def oracle() -> dict[str, float]:
    index = np.arange(1, N_SAMPLES + 1, dtype=np.float64)[:, None]
    feature = np.arange(1, N_INPUTS + 1, dtype=np.float64)[None, :]
    x = 0.8*np.sin(0.003*index + 0.19*feature)
    x_dot = 0.2*np.cos(0.005*(index + 3.0*feature))
    phi = np.empty((N_SAMPLES, 1 + N_INPUTS*DEGREE), dtype=np.float64)
    phi[:, 0] = 1.0
    jvp = np.zeros_like(phi)
    u = np.empty_like(phi)
    u[:, 0] = 0.11*np.sin(0.009*(index[:, 0] + 1.0))
    phi[:, 0] = 1.0
    column = 1
    for j in range(N_INPUTS):
        t_previous = np.ones(N_SAMPLES)
        t_current = x[:, j].copy()
        d_previous = np.zeros(N_SAMPLES)
        d_current = np.ones(N_SAMPLES)
        phi[:, column] = t_current
        jvp[:, column] = d_current*x_dot[:, j]
        for k in range(1, DEGREE):
            t_next = 2.0*x[:, j]*t_current - t_previous
            d_next = 2.0*t_current + 2.0*x[:, j]*d_current - d_previous
            phi[:, column + k] = t_next
            jvp[:, column + k] = d_next*x_dot[:, j]
            t_previous, t_current = t_current, t_next
            d_previous, d_current = d_current, d_next
        for c in range(DEGREE):
            u[:, column + c] = 0.11*np.sin(0.009*(index[:, 0] + column + c))
        column += DEGREE
    vjp = np.zeros((N_SAMPLES, N_INPUTS), dtype=np.float64)
    hvp = np.zeros_like(vjp)
    column = 1
    for j in range(N_INPUTS):
        t_previous = np.ones(N_SAMPLES)
        t_current = x[:, j].copy()
        d_previous = np.zeros(N_SAMPLES)
        d_current = np.ones(N_SAMPLES)
        dd_previous = np.zeros(N_SAMPLES)
        dd_current = np.zeros(N_SAMPLES)
        vjp[:, j] += u[:, column]*d_current
        for k in range(1, DEGREE):
            t_next = 2.0*x[:, j]*t_current - t_previous
            d_next = 2.0*t_current + 2.0*x[:, j]*d_current - d_previous
            dd_next = 4.0*d_current + 2.0*x[:, j]*dd_current - dd_previous
            vjp[:, j] += u[:, column + k]*d_next
            hvp[:, j] += u[:, column + k]*dd_next*x_dot[:, j]
            t_previous, t_current = t_current, t_next
            d_previous, d_current = d_current, d_next
            dd_previous, dd_current = dd_current, dd_next
        column += DEGREE
    return {
        "value_sum": float(np.sum(phi)), "jvp_sum": float(np.sum(jvp)),
        "vjp_sum": float(np.sum(vjp)), "hvp_sum": float(np.sum(hvp)),
    }


def parse(stdout: str) -> dict[str, float | int]:
    records: dict[str, float | int] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if not fields or not fields[0].startswith("chebyshev_"):
            continue
        if len(fields) < 6:
            raise RuntimeError(f"malformed Chebyshev record: {line!r}")
        phase = fields[0].removeprefix("chebyshev_")
        records[f"{phase}_n_samples"] = int(fields[1])
        records[f"{phase}_n_inputs"] = int(fields[2])
        records[f"{phase}_degree"] = int(fields[3])
        records[f"{phase}_seconds"] = float(fields[4])
        records[f"{phase}_sum"] = float(fields[5])
    expected = {"value", "jvp", "vjp", "hvp"}
    missing = expected.difference({key.removesuffix("_n_samples") for key in records
                                   if key.endswith("_n_samples")})
    if missing:
        raise RuntimeError(f"release app omitted phases: {sorted(missing)}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/chebyshev_basis.csv"))
    parser.add_argument("--target", default="fortml_bench_chebyshev")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1",
                        "FO_SCAN_FALLBACK": "regex"})
    subprocess.run(["fo", "build", "--flag", "-O2"], cwd=fortml,
                   env=environment, check=True, stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE, text=True)
    completed = subprocess.run(
        ["fo", "exec", "--no-build", args.target], cwd=fortml,
        env=environment, check=True, capture_output=True, text=True,
    )
    observed = parse(completed.stdout)
    expected = oracle()
    phases = ("value", "jvp", "vjp", "hvp")
    errors = {
        phase: abs(float(observed[f"{phase}_sum"]) - expected[f"{phase}_sum"])
        for phase in phases
    }
    if any(error > 5.0e-10 for error in errors.values()):
        raise RuntimeError(f"Chebyshev NumPy oracle mismatch: {errors}")
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O2",
    }
    rows = []
    for phase in phases:
        rows.append(rows_base(
            details, phase=phase, backend="fortml", status="pass",
            seconds_per_operation=observed[f"{phase}_seconds"],
            metric="sum_features_or_product", value=observed[f"{phase}_sum"],
            max_abs_error=errors[phase],
            oracle="independent NumPy Chebyshev recurrence and derivative oracle",
            notes="parameter-free input-major T_1..T_8; shared intercept excluded from derivatives",
        ))
    rows.append(rows_base(
        details, phase="device_contract", backend="fortml", device="cuda",
        status="unavailable", metric="resident_chebyshev_basis",
        value="FORTNUM_NOT_IMPLEMENTED", max_abs_error=0.0,
        oracle="typed resident-CUDA refusal", notes="no host fallback is claimed",
    ))
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
