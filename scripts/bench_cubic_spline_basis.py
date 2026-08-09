#!/usr/bin/env python3
"""Correctness-gated cubic B-spline value/JVP/VJP/HVP benchmark.

The NumPy oracle builds the same clamped knot vector independently and uses a
recursive Cox--de Boor evaluator.  Input derivatives are checked by central
finite differences away from breakpoints, which is the documented fixed-span
contract of the Fortran map.  The CUDA row records the typed host-only
boundary rather than silently falling back to a host transform.
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
    "n_inputs", "degree", "nbreak", "seconds_per_operation", "metric",
    "value", "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
N_SAMPLES = 2048
N_INPUTS = 2
DEGREE = 3
ORDER = DEGREE + 1
NBREAK = 6
REPETITIONS = 8


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
        "workload": "cubic_spline_basis", "phase": "", "backend": "",
        "device": "cpu", "status": "", "n_samples": N_SAMPLES,
        "n_inputs": N_INPUTS, "degree": DEGREE, "nbreak": NBREAK,
        "seconds_per_operation": "", "metric": "", "value": "",
        "max_abs_error": "", "oracle": "", "notes": "",
    })
    result.update(values)
    return result


def clamped_knots(breaks: np.ndarray) -> np.ndarray:
    return np.concatenate((np.repeat(breaks[0], ORDER), breaks[1:-1],
                           np.repeat(breaks[-1], ORDER)))


def cox(index: int, order: int, x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    if order == 1:
        return ((knots[index] <= x) & (x < knots[index + 1])) | (
            (x == knots[-1]) & (index == len(knots) - 2))
    left_den = knots[index + order - 1] - knots[index]
    right_den = knots[index + order] - knots[index + 1]
    left = ((x - knots[index])/left_den*cox(index, order - 1, x, knots)
            if left_den > 0.0 else 0.0)
    right = ((knots[index + order] - x)/right_den*cox(index + 1, order - 1, x, knots)
             if right_den > 0.0 else 0.0)
    return left + right


def evaluate(x: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    ncoef = NBREAK + ORDER - 2
    result = np.zeros((x.shape[0], 1 + N_INPUTS*ncoef), dtype=np.float64)
    result[:, 0] = 1.0
    column = 1
    for j in range(N_INPUTS):
        knots = clamped_knots(breaks[:, j])
        for index in range(ncoef):
            result[:, column + index] = cox(index, ORDER, x[:, j], knots)
        column += ncoef
    return result


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    breaks = np.array([
        [0.0, -1.0], [0.31, -0.43], [0.72, 0.02],
        [1.18, 0.51], [1.63, 1.24], [2.0, 2.0],
    ], dtype=np.float64)
    index = np.arange(1, N_SAMPLES + 1, dtype=np.float64)
    x = np.empty((N_SAMPLES, N_INPUTS), dtype=np.float64)
    x_dot = np.empty_like(x)
    for j in range(N_INPUTS):
        x[:, j] = breaks[0, j] + (breaks[-1, j] - breaks[0, j]) * (
            0.5 + 0.45*np.sin(0.013*(index + 3.0*(j + 1))))
        x_dot[:, j] = 0.2*np.cos(0.007*(index + 5.0*(j + 1)))
    u = np.empty((N_SAMPLES, 1 + N_INPUTS*(NBREAK + ORDER - 2)), dtype=np.float64)
    for j in range(u.shape[1]):
        u[:, j] = 0.11*np.sin(0.009*(index + (j + 1)))
    return x, x_dot, u, breaks


def oracle() -> dict[str, float]:
    x, x_dot, u, breaks = fixture()
    step = 2.0e-6
    phi = evaluate(x, breaks)
    phi_plus = evaluate(x + step*x_dot, breaks)
    phi_minus = evaluate(x - step*x_dot, breaks)
    jvp = (phi_plus - phi_minus)/(2.0*step)
    def vjp_from_x(points: np.ndarray) -> np.ndarray:
        output = np.zeros_like(points)
        for j in range(N_INPUTS):
            x_plus = points.copy()
            x_minus = points.copy()
            x_plus[:, j] += step
            x_minus[:, j] -= step
            output[:, j] = np.sum(u*(evaluate(x_plus, breaks) -
                                     evaluate(x_minus, breaks)), axis=1)/(2.0*step)
        return output

    vjp = vjp_from_x(x)
    vjp_plus = vjp_from_x(x + step*x_dot)
    vjp_minus = vjp_from_x(x - step*x_dot)
    hvp = (vjp_plus - vjp_minus)/(2.0*step)
    return {
        "value_sum": float(np.sum(phi)), "jvp_sum": float(np.sum(jvp)),
        "vjp_sum": float(np.sum(vjp)), "hvp_sum": float(np.sum(hvp)),
    }


def parse(stdout: str) -> dict[str, float | int]:
    records: dict[str, float | int] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if not fields or not fields[0].startswith("cubic_spline_"):
            continue
        if len(fields) < 6:
            raise RuntimeError(f"malformed cubic-spline record: {line!r}")
        phase = fields[0].removeprefix("cubic_spline_")
        records[f"{phase}_n_samples"] = int(fields[1])
        records[f"{phase}_n_inputs"] = int(fields[2])
        records[f"{phase}_features"] = int(fields[3])
        records[f"{phase}_seconds"] = float(fields[4])
        records[f"{phase}_sum"] = float(fields[-1])
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
                        default=Path("results/cubic_spline_basis.csv"))
    parser.add_argument("--target", default="fortml_bench_cubic_spline")
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
    tolerances = {"value": 2.0e-10, "jvp": 2.0e-6,
                  "vjp": 2.0e-5, "hvp": 3.0e-3}
    if any(errors[phase] > tolerances[phase] for phase in phases):
        raise RuntimeError(f"cubic-spline NumPy oracle mismatch: {errors}")
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
            oracle="independent NumPy clamped Cox--de Boor plus central FD products",
            notes="cubic order four; fixtures avoid knot breakpoints",
        ))
    rows.append(rows_base(
        details, phase="device_contract", backend="fortml", device="cuda",
        status="unavailable", metric="resident_cubic_spline_basis",
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
