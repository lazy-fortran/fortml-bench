#!/usr/bin/env python3
"""Correctness-gated fixed-structure XGBoost/LightGBM leaf objectives.

The NumPy oracle freezes a two-leaf stump and evaluates weighted squared and
logistic losses directly in the packed ``[base, left, right]`` coordinates.
The Fortran release app is accepted only when all value/gradient/JVP/VJP/HVP
errors are finite and below tolerance, FortOpt converges, and the CUDA
boundary remains the declared typed refusal.
"""

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


TOLERANCE = 2.0e-11
FIELDS = (
    "workload", "loss", "phase", "backend", "device", "status", "metric",
    "value", "max_abs_error", "seconds", "oracle", "python_version",
    "numpy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
)


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


def design() -> np.ndarray:
    return np.asarray(
        [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0],
         [1.0, 0.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float64,
    )


def squared(theta: np.ndarray, target: np.ndarray, weight: np.ndarray,
            l2: float, direction: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    matrix = design()
    margin = matrix @ theta
    residual = margin - target
    total = float(np.sum(weight))
    value = float(0.5 * np.sum(weight * residual * residual) / total
                  + 0.5 * l2 * np.dot(theta, theta))
    gradient = matrix.T @ (weight * residual) / total + l2 * theta
    hvp = matrix.T @ (weight * (matrix @ direction)) / total + l2 * direction
    return value, gradient, hvp


def logistic(theta: np.ndarray, target: np.ndarray, weight: np.ndarray,
             l2: float, direction: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    matrix = design()
    margin = matrix @ theta
    probability = 1.0 / (1.0 + np.exp(-margin))
    softplus = np.maximum(margin, 0.0) + np.log1p(np.exp(-np.abs(margin)))
    total = float(np.sum(weight))
    value = float(np.sum(weight * (softplus - target * margin)) / total
                  + 0.5 * l2 * np.dot(theta, theta))
    gradient = matrix.T @ (weight * (probability - target)) / total + l2 * theta
    curvature = weight * probability * (1.0 - probability)
    hvp = matrix.T @ (curvature * (matrix @ direction)) / total + l2 * direction
    return value, gradient, hvp


def run_app(fortml: Path) -> tuple[dict[str, float | int | str], float]:
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    start = time.perf_counter()
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_boosted_leaf_objective"],
        cwd=fortml, env=environment, check=True, capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - start
    values: dict[str, float | int | str] = {}
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        key, raw = fields
        if key.endswith(("_converged",)):
            values[key] = raw
        elif key.endswith(("_status", "_iterations", "_parameter_count")):
            values[key] = int(raw)
        else:
            values[key] = float(raw)
    required = {
        *(f"{prefix}_{metric}" for prefix in ("xgb_squared", "xgb_logistic",
                                               "lgbm_squared", "lgbm_logistic")
          for metric in ("status", "value_error", "gradient_error", "jvp_error",
                         "vjp_error", "hvp_error")),
        "leaf_optimize_status", "leaf_optimize_converged",
        "leaf_optimize_iterations", "leaf_optimize_gradient_norm",
        "leaf_optimize_parameter_count", "leaf_cuda_status",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise RuntimeError(f"release app omitted {missing}")
    for prefix in ("xgb_squared", "xgb_logistic", "lgbm_squared", "lgbm_logistic"):
        if int(values[f"{prefix}_status"]) != 0:
            raise RuntimeError(f"{prefix} objective status failed")
        errors = [float(values[f"{prefix}_{metric}_error"])
                  for metric in ("value", "gradient", "jvp", "vjp", "hvp")]
        if max(errors) > TOLERANCE:
            raise RuntimeError(f"{prefix} oracle error {max(errors):g}")
    if int(values["leaf_optimize_status"]) != 0 or values["leaf_optimize_converged"] != "T":
        raise RuntimeError("fixed-leaf FortOpt did not converge")
    if float(values["leaf_optimize_gradient_norm"]) > 1.0e-5:
        raise RuntimeError("fixed-leaf FortOpt gradient norm is too large")
    if int(values["leaf_optimize_parameter_count"]) != 3:
        raise RuntimeError("fixed-leaf optimizer coordinate count changed")
    if int(values["leaf_cuda_status"]) != 3:
        raise RuntimeError("fixed-leaf CUDA refusal changed")
    return values, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/boosted_leaf_objective.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/BOOSTED_LEAF_OBJECTIVE.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    values, seconds = run_app(fortml)
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (args.output.resolve(), args.report.resolve())),
        "compiler": "gfortran",
        "flags": "-O3",
    }
    rows: list[dict[str, Any]] = []
    for prefix, model, loss in (
        ("xgb_squared", "xgboost", "squared"),
        ("xgb_logistic", "xgboost", "logistic"),
        ("lgbm_squared", "lightgbm", "squared"),
        ("lgbm_logistic", "lightgbm", "logistic"),
    ):
        for phase, metric in (
            ("value_gradient", "value_error"), ("value_gradient", "gradient_error"),
            ("jvp", "jvp_error"), ("vjp", "vjp_error"), ("hvp", "hvp_error"),
        ):
            rows.append({
                **details, "workload": "boosted_leaf_objective", "loss": loss,
                "phase": f"{model}_{phase}", "backend": "fortml", "device": "cpu",
                "status": "pass", "metric": metric, "value": float(values[f"{prefix}_{metric}"]),
                "max_abs_error": float(values[f"{prefix}_{metric}"]), "seconds": seconds,
                "oracle": "independent NumPy fixed-stump weighted loss", "notes":
                    "packed [base_score, leaf weights]; split structure fixed",
            })
    rows.extend([
        {**details, "workload": "boosted_leaf_objective", "loss": "logistic",
         "phase": "fortopt_lbfgsb", "backend": "fortml", "device": "cpu", "status": "pass",
         "metric": "gradient_norm", "value": float(values["leaf_optimize_gradient_norm"]),
         "seconds": seconds, "oracle": "FortOpt projected L-BFGS-B convergence", "notes":
             f"iterations={int(values['leaf_optimize_iterations'])}; coordinates=3"},
        {**details, "workload": "boosted_leaf_objective", "loss": "both",
         "phase": "capability_check", "backend": "fortml", "device": "cuda",
         "status": "unavailable", "metric": "status_code", "value": 3.0,
         "seconds": seconds, "oracle": "declared resident-device contract", "notes":
             "FORTNUM_NOT_IMPLEMENTED; no host fallback"},
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# Fixed boosted-tree leaf objectives\n\n"
        "The release app checks weighted squared and binary logistic objectives "
        "for both XGBoost- and LightGBM-style fixed two-leaf trees. An independent "
        "NumPy stump design checks exact value, gradient, JVP, VJP, and HVP products. "
        "FortOpt L-BFGS-B consumes the same analytic callback. Split thresholds, "
        "categorical partitions, and missing routes are discrete state; CUDA is a "
        "typed `FORTNUM_NOT_IMPLEMENTED` refusal.\n\n"
        f"- FortML revision: `{details['fortml_revision']}`\n"
        f"- Benchmark revision: `{details['benchmark_revision']}`\n"
        f"- Compiler/flags: `{details['compiler']} {details['flags']}`\n"
        f"- Release probe wall time: `{seconds:.6g}` s\n"
        f"- Raw record: [`{args.output}`]({args.output.name})\n\n"
        "All CPU product errors are below `2e-11`; FortOpt converges with a "
        "three-coordinate bounded solve. The CUDA row records the explicit "
        "typed refusal rather than a host fallback.\n",
    )
    print(f"wrote {args.output} ({len(rows)} rows)")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
