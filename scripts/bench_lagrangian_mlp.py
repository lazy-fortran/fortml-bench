#!/usr/bin/env python3
"""Correctness-gated scalar Lagrangian MLP workload."""

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
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    ignored_names = set()
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


def oracle() -> dict[str, float]:
    q, v, acceleration = 0.3, -0.4, 0.2
    x = np.asarray([q, v])
    w1 = np.asarray([[0.7, -0.4], [0.2, 0.3]])
    b1 = np.asarray([0.1, -0.2])
    w2 = np.asarray([0.4, -0.5])
    b2 = 0.3
    hidden = np.tanh(x@w1+b1)
    value = float(hidden@w2+b2)
    first = 1.0-hidden**2
    gradient = (first*w2)@w1.T
    second = -2.0*hidden*first
    hessian = np.zeros((2, 2))
    for column in range(2):
        hessian += w2[column]*second[column]*np.outer(w1[:, column], w1[:, column])
    mass = float(hessian[1, 1])
    residual = float(hessian[1, 0]*v + mass*acceleration - gradient[0])
    return {"value": value, "gradient_q": float(gradient[0]),
            "gradient_v": float(gradient[1]), "mass": mass, "residual": residual}


def run_app(fortml: Path) -> tuple[dict[str, float], float]:
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    started = time.perf_counter()
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_lagrangian_mlp"], cwd=fortml,
        env=environment, check=True, capture_output=True, text=True,
    )
    elapsed = time.perf_counter()-started
    values: dict[str, float] = {}
    cuda_available = False
    for line in completed.stdout.splitlines():
        key, separator, raw = line.partition(",")
        if key == "lagrangian_cuda":
            cuda_available = raw.strip() == "available"
        elif separator and key.startswith("lagrangian_"):
            values[key] = float(raw.strip())
    required = {"lagrangian_value", "lagrangian_gradient_q", "lagrangian_gradient_v",
                "lagrangian_mass", "lagrangian_residual"}
    if set(values) != required or cuda_available:
        raise RuntimeError(f"release app rows/refusal mismatch: {sorted(values)}")
    return values, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/lagrangian_mlp.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/LAGRANGIAN_MLP.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    expected = oracle()
    values, seconds = run_app(fortml)
    errors = {key: abs(values[key.replace("value", "lagrangian_value") if key == "value" else "lagrangian_"+key]-expected[key])
              for key in expected}
    if max(errors.values()) > TOLERANCE:
        raise RuntimeError(f"analytic MLP oracle mismatch: {errors}")
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (args.output.resolve(), args.report.resolve())),
        "compiler": "gfortran", "flags": "-O3",
    }
    rows = []
    for phase, metric, key in (
        ("value", "scalar", "value"), ("gradient_q", "dL/dq", "gradient_q"),
        ("gradient_v", "dL/dv", "gradient_v"), ("mass_matrix", "L_vv", "mass"),
        ("euler_lagrange", "residual", "residual"),
    ):
        app_key = {"value": "lagrangian_value", "gradient_q": "lagrangian_gradient_q",
                   "gradient_v": "lagrangian_gradient_v", "mass_matrix": "lagrangian_mass",
                   "euler_lagrange": "lagrangian_residual"}[phase]
        rows.append({**details, "workload": "lagrangian_mlp", "phase": phase,
                     "backend": "fortml", "device": "cpu", "status": "pass",
                     "metric": metric, "value": values[app_key],
                     "max_abs_error": errors[key], "seconds": seconds,
                     "oracle": "independent NumPy tanh-MLP analytic Hessian",
                     "notes": "topology [2,2,1]; fixed packed parameters"})
    rows.append({**details, "workload": "lagrangian_mlp", "phase": "capability_check",
                 "backend": "fortml", "device": "cuda", "status": "unavailable",
                 "metric": "status", "value": 3.0, "max_abs_error": "", "seconds": seconds,
                 "oracle": "declared resident-device contract",
                 "notes": "FORTNUM_NOT_IMPLEMENTED; no host fallback"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# Scalar Lagrangian MLP\n\n"
        "The release application is checked against an independent NumPy analytic tanh-MLP Hessian oracle for L, its state gradient, the velocity mass matrix, and the Euler--Lagrange residual. CUDA is a typed refusal; no host fallback is claimed.\n\n"
        f"- FortML revision: `{details['fortml_revision']}`\n"
        f"- Benchmark revision: `{details['benchmark_revision']}`\n"
        f"- Compiler/flags: `{details['compiler']} {details['flags']}`\n"
        f"- Release wall time: `{seconds:.6g}` s\n"
        f"- Maximum analytic error: `{max(errors.values()):.6g}`\n"
        f"- Raw record: [`{args.output}`]({args.output.name})\n",
    )
    print(f"wrote {args.output} ({len(rows)} rows)")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
