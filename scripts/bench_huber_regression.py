#!/usr/bin/env python3
"""Correctness-gated weighted Huber regression workload.

The fixed probe is evaluated independently in NumPy and compared with the
FortML release application.  The app also runs a bounded FortOpt fit; the
selected CUDA row must remain the declared typed refusal.
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


def probe() -> tuple[float, float]:
    x = np.asarray([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    target = np.asarray([-3.7, -2.5, -1.6, -0.9, 0.2, 1.0, 2.2, 3.0])
    theta = np.asarray([0.5, 1.1])
    delta = 1.0
    l2 = 1.0e-3
    residual = theta[0] + x * theta[1] - target
    loss = np.where(
        np.abs(residual) <= delta,
        0.5 * residual * residual,
        delta * (np.abs(residual) - 0.5 * delta),
    )
    value = float(np.mean(loss) + 0.5 * l2 * theta[1] ** 2)
    derivative = np.where(
        np.abs(residual) <= delta,
        residual,
        delta * np.sign(residual),
    )
    gradient = np.asarray([
        np.mean(derivative), np.mean(derivative * x) + l2 * theta[1],
    ])
    return value, float(np.linalg.norm(gradient))


def run_app(fortml: Path) -> tuple[dict[str, float | str], float]:
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    started = time.perf_counter()
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_huber_regression"],
        cwd=fortml, env=environment, check=True, capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - started
    values: dict[str, float | str] = {}
    for line in completed.stdout.splitlines():
        if line.strip() == "huber_cuda,unavailable":
            values["huber_cuda_status"] = "unavailable"
            continue
        fields = line.split()
        if len(fields) != 2:
            continue
        key, raw = fields
        if key.startswith("huber_"):
            values[key.rstrip(",")] = float(raw)
    required = {
        "huber_probe_value", "huber_probe_gradient_norm", "huber_objective",
        "huber_gradient_norm", "huber_cuda_status",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise RuntimeError(f"release app omitted {missing}")
    return values, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/huber_regression.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/HUBER_REGRESSION.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    expected_value, expected_gradient_norm = probe()
    values, seconds = run_app(fortml)
    value_error = abs(float(values["huber_probe_value"]) - expected_value)
    gradient_error = abs(float(values["huber_probe_gradient_norm"]) - expected_gradient_norm)
    if value_error > TOLERANCE or gradient_error > TOLERANCE:
        raise RuntimeError(
            f"NumPy probe mismatch: value={value_error:g}, gradient={gradient_error:g}"
        )
    if float(values["huber_gradient_norm"]) > 2.0e-5:
        raise RuntimeError("FortOpt gradient norm is too large")
    if values["huber_cuda_status"] != "unavailable":
        raise RuntimeError("Huber CUDA refusal changed")

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
        {**details, "workload": "huber_regression", "phase": "probe_value",
         "backend": "fortml", "device": "cpu", "status": "pass", "metric": "value",
         "value": float(values["huber_probe_value"]), "max_abs_error": value_error,
         "seconds": seconds, "oracle": "independent NumPy weighted Huber probe",
         "notes": "uniform weights; packed [intercept, slope]"},
        {**details, "workload": "huber_regression", "phase": "probe_gradient",
         "backend": "fortml", "device": "cpu", "status": "pass", "metric": "norm",
         "value": float(values["huber_probe_gradient_norm"]), "max_abs_error": gradient_error,
         "seconds": seconds, "oracle": "independent NumPy weighted Huber probe",
         "notes": "fixed residual branches"},
        {**details, "workload": "huber_regression", "phase": "fortopt_lbfgsb",
         "backend": "fortml", "device": "cpu", "status": "pass", "metric": "gradient_norm",
         "value": float(values["huber_gradient_norm"]), "max_abs_error": "",
         "seconds": seconds, "oracle": "FortOpt bounded L-BFGS-B convergence",
         "notes": "weighted fixture; delta=1; l2=1e-3"},
        {**details, "workload": "huber_regression", "phase": "capability_check",
         "backend": "fortml", "device": "cuda", "status": "unavailable", "metric": "status",
         "value": 3.0, "max_abs_error": "", "seconds": seconds,
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
        "# Weighted Huber regression\n\n"
        "The release application fits a weighted linear Huber model through "
        "FortOpt L-BFGS-B and exposes a fixed packed probe. The probe value and "
        "gradient norm are compared with an independent NumPy implementation; "
        "the CUDA row records the typed refusal rather than a host fallback.\n\n"
        f"- FortML revision: `{details['fortml_revision']}`\n"
        f"- Benchmark revision: `{details['benchmark_revision']}`\n"
        f"- Compiler/flags: `{details['compiler']} {details['flags']}`\n"
        f"- Release probe wall time: `{seconds:.6g}` s\n"
        f"- Probe value error: `{value_error:.6g}`\n"
        f"- Probe gradient-norm error: `{gradient_error:.6g}`\n"
        f"- FortOpt gradient norm: `{float(values['huber_gradient_norm']):.6g}`\n"
        f"- Raw record: [`{args.output}`]({args.output.name})\n",
    )
    print(f"wrote {args.output} ({len(rows)} rows)")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
