#!/usr/bin/env python3
"""Independent weighted multi-output linear quantile-regression oracle."""

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
    "max_abs_error", "seconds", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
TOLERANCE = 2.0e-12


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names: set[str] = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            continue
    dirty = any(
        line[3:].split(" -> ")[-1].strip() not in ignored_names
        for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True,
        ).splitlines()
    )
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    x = np.asarray([
        [-1.2, 0.7], [-0.8, -0.5], [-0.3, 1.1], [0.1, -1.3],
        [0.6, 0.4], [1.0, 1.7], [1.5, -0.9],
    ])
    target = np.asarray([
        [0.3, -0.4], [0.8, 0.5], [-0.2, 1.4], [1.1, -0.8],
        [0.1, 1.1], [1.8, 0.2], [0.7, 1.7],
    ])
    weights = np.asarray([0.6, 1.0, 1.3, 0.8, 1.2, 0.9, 0.7])
    levels = np.asarray([0.25, 0.75])
    theta = np.asarray([0.1, -0.2, 1.0, 0.3, 0.4, 0.8])
    return x, target, weights, levels, theta


def oracle() -> tuple[float, float]:
    x, target, weights, levels, theta = fixture()
    coefficient = theta.reshape((3, 2), order="F")
    design = np.column_stack((np.ones(x.shape[0]), x))
    prediction = design @ coefficient
    residual = prediction - target
    pinball = np.where(residual >= 0.0, levels * residual, (levels - 1.0) * residual)
    normalizer = weights.sum() * target.shape[1]
    value = float((weights[:, None] * pinball).sum() / normalizer)
    value += 0.5e-3 * float(np.sum(coefficient[1:, :] ** 2))
    derivative = np.where(residual > 0.0, levels, levels - 1.0)
    gradient_matrix = (design.T @ (weights[:, None] * derivative)) / normalizer
    gradient_matrix[1:, :] += 1.0e-3 * coefficient[1:, :]
    return value, float(np.linalg.norm(gradient_matrix.reshape(-1, order="F")))


def run_app(fortml: Path) -> tuple[dict[str, float | str], float]:
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    started = time.perf_counter()
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_quantile_regression"],
        cwd=fortml, env=environment, check=True, capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - started
    values: dict[str, float | str] = {}
    for line in completed.stdout.splitlines():
        if line.strip() == "quantile_cuda,unavailable":
            values["quantile_cuda_status"] = "unavailable"
            continue
        fields = line.split()
        if len(fields) != 2:
            continue
        key, raw = fields
        if key.startswith("quantile_"):
            values[key.rstrip(",")] = float(raw)
    required = {
        "quantile_probe_value", "quantile_probe_gradient_norm",
        "quantile_exact_objective", "quantile_smoothed_gradient_norm",
        "quantile_exact_gradient_norm", "quantile_cuda_status",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise RuntimeError(f"release app omitted {missing}")
    return values, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/quantile_regression.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/QUANTILE_REGRESSION.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    expected_value, expected_gradient_norm = oracle()
    values, seconds = run_app(fortml)
    value_error = abs(float(values["quantile_probe_value"]) - expected_value)
    gradient_error = abs(
        float(values["quantile_probe_gradient_norm"]) - expected_gradient_norm
    )
    if value_error > TOLERANCE or gradient_error > TOLERANCE:
        raise RuntimeError(
            f"NumPy probe mismatch: value={value_error:g}, gradient={gradient_error:g}"
        )
    if float(values["quantile_exact_objective"]) < 0.0:
        raise RuntimeError("exact pinball objective must be nonnegative")
    if values["quantile_cuda_status"] != "unavailable":
        raise RuntimeError("quantile CUDA refusal changed")

    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(
            root, (args.output.resolve(), args.report.resolve())
        ),
        "compiler": "gfortran", "flags": "-O3",
    }
    rows = [
        {**details, "workload": "quantile_regression", "phase": "probe_value",
         "backend": "fortml", "device": "cpu", "status": "pass", "metric": "value",
         "value": float(values["quantile_probe_value"]), "max_abs_error": value_error,
         "seconds": seconds, "oracle": "independent NumPy weighted pinball probe",
         "notes": "levels=[0.25,0.75]; packed Fortran-column-major coefficients"},
        {**details, "workload": "quantile_regression", "phase": "probe_gradient",
         "backend": "fortml", "device": "cpu", "status": "pass", "metric": "norm",
         "value": float(values["quantile_probe_gradient_norm"]),
         "max_abs_error": gradient_error, "seconds": seconds,
         "oracle": "independent NumPy weighted pinball probe",
         "notes": "fixed residual branches and feature L2"},
        {**details, "workload": "quantile_regression", "phase": "fortopt_lbfgsb",
         "backend": "fortml", "device": "cpu", "status": "pass",
         "metric": "exact_objective", "value": float(values["quantile_exact_objective"]),
         "max_abs_error": "", "seconds": seconds,
         "oracle": "exact post-continuation pinball objective",
         "notes": "three-stage C1 continuation; fit_smoothing starts at 0.1"},
        {**details, "workload": "quantile_regression", "phase": "capability_check",
         "backend": "fortml", "device": "cuda", "status": "unavailable",
         "metric": "status", "value": 3.0, "max_abs_error": "", "seconds": seconds,
         "oracle": "declared resident-device contract",
         "notes": "FORTNUM_NOT_IMPLEMENTED; no host fallback"},
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# Weighted linear quantile regression\n\n"
        "The release application fits a weighted multi-output affine model "
        "through FortOpt L-BFGS-B. An independent NumPy pinball oracle checks "
        "the packed probe value and gradient; the fit reports the exact "
        "post-continuation objective and the CUDA row records its typed refusal.\n\n"
        f"- FortML revision: `{details['fortml_revision']}`\n"
        f"- Benchmark revision: `{details['benchmark_revision']}`\n"
        f"- Compiler/flags: `{details['compiler']} {details['flags']}`\n"
        f"- Release probe wall time: `{seconds:.6g}` s\n"
        f"- Probe value error: `{value_error:.6g}`\n"
        f"- Probe gradient-norm error: `{gradient_error:.6g}`\n"
        f"- Exact post-fit objective: `{float(values['quantile_exact_objective']):.6g}`\n"
        f"- Exact post-fit gradient norm: `{float(values['quantile_exact_gradient_norm']):.6g}`\n"
        f"- Raw record: [`{args.output}`]({args.output.name})\n",
    )
    print(f"wrote {args.output} ({len(rows)} rows)")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
