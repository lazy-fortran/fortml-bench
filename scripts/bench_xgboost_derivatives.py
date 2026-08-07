#!/usr/bin/env python3
"""Correctness-gated benchmark for XGBoost query JVP/VJP products.

The release app emits predictions at a query and at two independently chosen
central-difference points.  NumPy recomputes the finite-difference products
and the scalar reverse-mode directional product before accepting timings.  A
fitted tree is piecewise constant: products are zero away from split surfaces,
while the public API must return a typed domain refusal on a surface.  CUDA is
recorded as an explicit unavailable capability, never as a host fallback.
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


EPSILON = 1.0e-6
N_QUERY = 6
FORTNUM_DOMAIN_ERROR = 1
FORTNUM_NOT_IMPLEMENTED = 3

FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_query",
    "seconds_per_operation", "metric", "value", "max_abs_error", "oracle",
    "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            continue
    dirty = False
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        if line[3:].split(" -> ")[-1].strip() not in ignored_names:
            dirty = True
            break
    return head + ("+dirty" if dirty else "")


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output.resolve(),)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def parse(stdout: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    vector_names = {
        "xgb_derivative_query", "xgb_derivative_prediction",
        "xgb_derivative_prediction_plus", "xgb_derivative_prediction_minus",
        "xgb_derivative_jvp", "xgb_derivative_vjp",
    }
    for line in stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        name = fields[0]
        if name in vector_names:
            parsed[name] = np.asarray([float(value) for value in fields[1:]])
        elif name in {
            "xgb_derivative_jvp_boundary_status",
            "xgb_derivative_vjp_boundary_status",
            "xgb_derivative_cuda_status",
        }:
            if len(fields) != 2:
                raise RuntimeError(f"malformed status line: {line!r}")
            parsed[name] = int(fields[1])
        elif name in {"xgb_derivative_jvp_seconds", "xgb_derivative_vjp_seconds"}:
            if len(fields) != 2:
                raise RuntimeError(f"malformed timing line: {line!r}")
            parsed[name] = float(fields[1])
    required = vector_names | {
        "xgb_derivative_jvp_boundary_status",
        "xgb_derivative_vjp_boundary_status", "xgb_derivative_cuda_status",
        "xgb_derivative_jvp_seconds", "xgb_derivative_vjp_seconds",
    }
    missing = required - parsed.keys()
    if missing:
        raise RuntimeError(f"release app omitted {sorted(missing)}")
    if any(parsed[name].size != N_QUERY for name in vector_names if name != "xgb_derivative_query"):
        raise RuntimeError("release app emitted a malformed derivative vector")
    if parsed["xgb_derivative_query"].size != N_QUERY:
        raise RuntimeError("release app emitted a malformed query vector")
    return parsed


def row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(details)
    result.update(values)
    return result


def run(fortml: Path, target: str) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    completed = subprocess.run(
        ["fo", "exec", "--no-build", target], cwd=fortml, env=environment,
        check=True, capture_output=True, text=True,
    )
    return parse(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/xgboost_derivatives.csv"),
    )
    parser.add_argument("--target", default="fortml_bench_xgboost_derivatives")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    details = metadata(root, fortml, args.output.resolve())
    observed = run(fortml, args.target)

    query = observed["xgb_derivative_query"]
    prediction = observed["xgb_derivative_prediction"]
    prediction_plus = observed["xgb_derivative_prediction_plus"]
    prediction_minus = observed["xgb_derivative_prediction_minus"]
    tangent = (prediction_plus - prediction_minus)/(2.0*EPSILON)
    jvp = observed["xgb_derivative_jvp"]
    vjp = observed["xgb_derivative_vjp"]
    cotangent = np.array((1.0, -0.5, 0.25, 0.75, -1.25, 0.5))
    directional_fd = float(np.dot(cotangent, tangent))
    directional_vjp = float(np.dot(vjp, np.ones(N_QUERY)))
    jvp_error = float(np.max(np.abs(jvp - tangent)))
    vjp_error = float(np.max(np.abs(vjp)))
    directional_error = abs(directional_fd - directional_vjp)
    if not np.all(np.isfinite((prediction, prediction_plus, prediction_minus, jvp, vjp))):
        raise RuntimeError("nonfinite derivative product emitted")
    if jvp_error > 1.0e-12 or vjp_error > 1.0e-12 or directional_error > 1.0e-12:
        raise RuntimeError(
            f"tree derivative oracle mismatch: jvp={jvp_error:.3e}, "
            f"vjp={vjp_error:.3e}, directional={directional_error:.3e}"
        )
    if np.any(np.abs(query - 3.5) < EPSILON):
        raise RuntimeError("finite-difference fixture crosses a split surface")
    if observed["xgb_derivative_jvp_boundary_status"] != FORTNUM_DOMAIN_ERROR:
        raise RuntimeError("JVP boundary refusal changed")
    if observed["xgb_derivative_vjp_boundary_status"] != FORTNUM_DOMAIN_ERROR:
        raise RuntimeError("VJP boundary refusal changed")
    if observed["xgb_derivative_cuda_status"] != FORTNUM_NOT_IMPLEMENTED:
        raise RuntimeError("CUDA capability refusal changed")

    records = [
        row(details, workload="xgboost_derivatives", phase="jvp", backend="fortml",
            device="cpu", status="pass", n_query=N_QUERY,
            seconds_per_operation=observed["xgb_derivative_jvp_seconds"],
            metric="central_difference_max_abs_error", value=jvp_error,
            max_abs_error=jvp_error,
            oracle="independent NumPy central difference away from split surfaces",
            notes="piecewise-constant tree JVP is zero"),
        row(details, workload="xgboost_derivatives", phase="vjp", backend="fortml",
            device="cpu", status="pass", n_query=N_QUERY,
            seconds_per_operation=observed["xgb_derivative_vjp_seconds"],
            metric="adjoint_max_abs_value", value=vjp_error,
            max_abs_error=max(vjp_error, directional_error),
            oracle="independent NumPy directional finite difference",
            notes="piecewise-constant tree VJP is zero"),
        row(details, workload="xgboost_derivatives", phase="boundary_jvp",
            backend="fortml", device="cpu", status="pass", n_query=1,
            metric="status_code", value=observed["xgb_derivative_jvp_boundary_status"],
            max_abs_error=0.0, oracle="typed derivative-domain contract",
            notes="learned split surface refuses classical derivative"),
        row(details, workload="xgboost_derivatives", phase="boundary_vjp",
            backend="fortml", device="cpu", status="pass", n_query=1,
            metric="status_code", value=observed["xgb_derivative_vjp_boundary_status"],
            max_abs_error=0.0, oracle="typed derivative-domain contract",
            notes="learned split surface refuses classical derivative"),
        row(details, workload="xgboost_derivatives", phase="predict",
            backend="fortml", device="cuda", status="unavailable", n_query=N_QUERY,
            metric="status_code", value=observed["xgb_derivative_cuda_status"],
            oracle="typed device contract",
            notes="no resident CUDA tree kernel; no host fallback"),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()
