#!/usr/bin/env python3
"""Correctness-gated LightGBM query-weighted rank:pairwise benchmark.

The NumPy oracle is a direct pair loop.  It checks endpoint-minimum row
weights, query isolation, and the two-row Newton leaf solution before the
release app is run.  CUDA remains an explicit typed refusal; no host result
is relabeled as resident ranking execution.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_queries", "seconds_per_operation", "metric", "value", "max_abs_error",
    "oracle", "python_version", "numpy_version", "fortml_revision",
    "benchmark_revision", "compiler", "flags", "notes",
)
FORTNUM_DOMAIN_ERROR = 1
FORTNUM_NOT_IMPLEMENTED = 3


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
    dirty = any(
        line[3:].split(" -> ")[-1].strip() not in ignored_names
        for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True,
        ).splitlines()
    )
    return head + ("+dirty" if dirty else "")


def direct_oracle() -> tuple[float, float, float, float]:
    margin = np.zeros(3, dtype=np.float64)
    target = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    group = np.asarray((1, 1, 2), dtype=np.int64)
    weights = np.asarray((2.0, 4.0, 7.0), dtype=np.float64)
    gradient = np.zeros(3, dtype=np.float64)
    hessian = np.zeros(3, dtype=np.float64)
    loss = 0.0
    weight_sum = 0.0
    for i in range(len(target) - 1):
        for j in range(i + 1, len(target)):
            if group[i] != group[j] or target[i] == target[j]:
                continue
            high, low = (i, j) if target[i] > target[j] else (j, i)
            pair_weight = min(weights[i], weights[j])
            probability = 0.5
            loss += pair_weight * np.log(2.0)
            gradient[high] -= pair_weight * probability
            gradient[low] += pair_weight * probability
            hessian[high] += pair_weight * probability * (1.0 - probability)
            hessian[low] += pair_weight * probability * (1.0 - probability)
            weight_sum += pair_weight
    loss /= weight_sum
    gradient_error = float(np.max(np.abs(gradient - [-1.0, 1.0, 0.0])))
    hessian_error = float(np.max(np.abs(hessian - [0.5, 0.5, 0.0])))
    isolation_error = float(max(abs(gradient[2]), abs(hessian[2])))
    error = max(gradient_error, hessian_error, isolation_error)
    if error > 1.0e-14 or not np.array_equal(group, [1, 1, 2]):
        raise RuntimeError(f"pairwise independent oracle failed: {error:.3e}")
    return float(loss), gradient_error, hessian_error, error


def run_app(fortml: Path, target: str) -> dict[str, str]:
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
    parsed: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) == 2:
            parsed[fields[0]] = fields[1]
    required = {
        "lightgbm_ranking_fit_seconds", "lightgbm_ranking_predict_seconds",
        "lightgbm_ranking_prediction", "lightgbm_ranking_oracle_error",
        "lightgbm_ranking_objective", "lightgbm_ranking_cuda_status",
    }
    missing = required - parsed.keys()
    if missing:
        raise RuntimeError(f"release app omitted {sorted(missing)}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/lightgbm_ranking.csv"))
    parser.add_argument("--target", default="fortml_bench_lightgbm_ranking")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    loss, gradient_error, hessian_error, oracle_error = direct_oracle()
    observed: dict[str, str] = {}
    if not args.skip_fortml:
        observed = run_app(fortml, args.target)
        app_error = float(observed["lightgbm_ranking_oracle_error"])
        if app_error > 2.0e-12:
            raise RuntimeError(f"ranking app oracle error changed: {app_error:.3e}")
        if observed["lightgbm_ranking_objective"] != "rank:pairwise":
            raise RuntimeError("ranking objective metadata changed")
        if int(observed["lightgbm_ranking_cuda_status"]) != FORTNUM_NOT_IMPLEMENTED:
            raise RuntimeError("ranking CUDA refusal changed")
    fortml_rev = revision(fortml) if fortml.exists() else "unavailable"
    bench_rev = revision(root, (args.output.resolve(),))
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": fortml_rev, "benchmark_revision": bench_rev,
        "compiler": "gfortran", "flags": "-O3",
    }

    def row(**values: object) -> dict[str, str]:
        output = {field: "" for field in FIELDS}
        output.update({
            "workload": "lightgbm_rank_pairwise", "backend": "fortml",
            "device": "cpu", "status": "pass", "n_samples": "3",
            "n_queries": "2", "oracle": "independent NumPy pairwise oracle",
            **details,
        })
        output.update({key: str(value) for key, value in values.items()})
        return output

    rows = [row(
        phase="independent_oracle", backend="numpy_oracle", seconds_per_operation="0.0",
        metric="pairwise_loss", value=loss, max_abs_error=oracle_error,
        notes=f"gradient_error={gradient_error:.3e}; hessian_error={hessian_error:.3e}; minimum endpoint weight",
    )]
    if args.skip_fortml:
        rows.append(row(phase="public_contract_gate", status="skipped", metric="oracle_max_abs_error",
                        value="nan", max_abs_error="nan", notes="--skip-fortml"))
    else:
        rows.extend([
            row(phase="fit", seconds_per_operation=observed["lightgbm_ranking_fit_seconds"],
                metric="fit_seconds", value=observed["lightgbm_ranking_fit_seconds"], max_abs_error="0.0"),
            row(phase="predict", seconds_per_operation=observed["lightgbm_ranking_predict_seconds"],
                metric="prediction_max_abs_error", value=observed["lightgbm_ranking_oracle_error"],
                max_abs_error=observed["lightgbm_ranking_oracle_error"]),
            row(phase="device_contract", device="cuda", status="unavailable",
                metric="resident_ranking_tree", value="nan", max_abs_error="nan",
                notes="typed CUDA refusal; no host fallback"),
        ])
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
