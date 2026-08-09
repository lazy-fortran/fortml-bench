#!/usr/bin/env python3
"""Correctness-gated PINN structure-aware finite-feature GP workload."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
import time
from pathlib import Path

import numpy as np

TOLERANCE = 3.0e-12
FIELDS = (
    "workload", "phase", "backend", "device", "status", "metric", "value",
    "max_abs_error", "seconds", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    ignored_names = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            pass
    dirty = any(
        line[3:].split(" -> ")[-1].strip() not in ignored_names
        for line in subprocess.check_output(["git", "-C", str(repository), "status", "--porcelain"], text=True).splitlines()
    )
    return head + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, 8)[:, None]
    target = np.sin(np.pi*x) + 0.2*x*x
    return x, target


def layer(seed: int, index: int, n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    scale = np.sqrt(6.0/(n_in+n_out))
    idx = np.arange(1, n_in*n_out+1, dtype=np.float64).reshape((n_in, n_out), order="F")
    weights = scale*np.sin(seed + 1009*index + 9176*idx)
    bias = 0.01*scale*np.sin(seed + 1009*index + 7919*np.arange(1, n_out+1))
    return weights, bias


def oracle() -> dict[str, float]:
    x, target = fixture()
    w1, b1 = layer(41, 1, 1, 3)
    w2, b2 = layer(41, 2, 3, 1)
    hidden = np.tanh(x@w1 + b1)
    design = np.column_stack((hidden, np.ones(x.shape[0])))
    coefficients = np.linalg.solve(design.T@design + 0.07*np.eye(4), design.T@target)
    prediction = design@coefficients
    before_error = float(b2[0] - 0.10)
    after_error = float(coefficients[-1, 0] - 0.10)
    weights = np.array([1.0, 2.0, 0.5, 1.5])
    return {
        "prediction_mean": float(np.mean(prediction)),
        "objective_before": float(0.5*np.dot(weights, before_error**2*np.ones(4))),
        "objective_after": float(0.5*np.dot(weights, after_error**2*np.ones(4))),
        "residual_after": float(weights[1]*0.5*after_error**2),
    }


def run_app(fortml: Path) -> tuple[dict[str, float | str], float]:
    env = os.environ.copy()
    env.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, env=env,
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    started = time.perf_counter()
    completed = subprocess.run(["fo", "exec", "--no-build", "fortml_bench_pinn_structure_gp"],
                               cwd=fortml, env=env, check=True, capture_output=True, text=True)
    values: dict[str, float | str] = {}
    for line in completed.stdout.splitlines():
        line = line.strip()
        if line == "pinn_structure_gp_cuda,unavailable":
            values["cuda"] = "unavailable"
        elif line.startswith("pinn_structure_gp_") and "," in line:
            name, raw = line.split(",", 1)
            values[name.strip()] = float(raw.strip())
    required = {"pinn_structure_gp_prediction_mean", "pinn_structure_gp_objective_before",
                "pinn_structure_gp_objective_after", "pinn_structure_gp_residual_term",
                "pinn_structure_gp_structure_defect", "pinn_structure_gp_hidden_delta", "cuda"}
    missing = sorted(required - values.keys())
    if missing:
        raise RuntimeError(f"release app omitted {missing}")
    return values, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/pinn_structure_gp.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/PINN_STRUCTURE_GP.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    expected = oracle()
    values, seconds = run_app(fortml)
    details = {"python_version": platform.python_version(), "numpy_version": np.__version__,
               "fortml_revision": revision(fortml),
               "benchmark_revision": revision(root, (args.output.resolve(), args.report.resolve())),
               "compiler": "gfortran", "flags": "-O3"}
    rows = []
    for phase, key in (("prediction", "prediction_mean"), ("objective_before", "objective_before"),
                       ("objective_after", "objective_after"), ("residual", "residual_after")):
        actual = float(values[f"pinn_structure_gp_{'residual_term' if key == 'residual_after' else key}"])
        error = abs(actual - expected[key])
        if error > TOLERANCE:
            raise RuntimeError(f"PINN oracle mismatch for {phase}: {error:g}")
        rows.append({**details, "workload": "pinn_structure_gp", "phase": phase,
                     "backend": "fortml", "device": "cpu", "status": "pass", "metric": key,
                     "value": actual, "max_abs_error": error, "seconds": seconds,
                     "oracle": "independent NumPy frozen-feature kernel-ridge and named-term oracle",
                     "notes": "manufactured 1-D PDE; regularization=0.07"})
    for metric, key in (("structure_defect", "pinn_structure_gp_structure_defect"),
                        ("hidden_delta", "pinn_structure_gp_hidden_delta")):
        actual = float(values[key])
        if actual > TOLERANCE:
            raise RuntimeError(f"PINN structure contract mismatch for {metric}: {actual:g}")
        rows.append({**details, "workload": "pinn_structure_gp", "phase": "structure",
                     "backend": "fortml", "device": "cpu", "status": "pass", "metric": metric,
                     "value": actual, "max_abs_error": actual, "seconds": seconds,
                     "oracle": "exact hidden-prefix/structure certificate", "notes": "expected zero"})
    if values["cuda"] != "unavailable":
        raise RuntimeError("PINN CUDA refusal changed")
    rows.append({**details, "workload": "pinn_structure_gp", "phase": "capability_check",
                 "backend": "fortml", "device": "cuda", "status": "unavailable", "metric": "status",
                 "value": 3.0, "max_abs_error": "", "seconds": seconds,
                 "oracle": "declared resident-device contract",
                 "notes": "FORTNUM_NOT_IMPLEMENTED; no host fallback"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        "# PINN structure-aware finite-feature GP\n\n"
        "This lane compares the named-term PINN structure initializer with an independent NumPy frozen-feature kernel-ridge and objective-term oracle. It records zero hidden/structure defect and the typed CUDA refusal.\n\n"
        f"- FortML revision: `{details['fortml_revision']}`\n"
        f"- Benchmark revision: `{details['benchmark_revision']}`\n"
        f"- Maximum oracle error: `{max(float(row['max_abs_error']) for row in rows if row['device'] == 'cpu'):.6g}`\n"
        f"- Raw record: [`results/pinn_structure_gp.csv`](pinn_structure_gp.csv)\n"
    )
    print(f"wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
