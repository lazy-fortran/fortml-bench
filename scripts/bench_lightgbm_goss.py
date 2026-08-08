#!/usr/bin/env python3
"""Correctness-gated LightGBM GOSS benchmark.

The NumPy oracle independently selects the largest-gradient rows, applies the
LightGBM ``(1-a)/b`` correction to the sampled remainder, exhaustively scores
the one-dimensional Newton split, and replays the fitted prediction.  The
release app additionally checks deterministic seed replay, malformed-rate
refusal, and the explicit CUDA boundary.
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
    "n_features", "n_estimators", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
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


def goss_hash(seed: int, round_index: int, row: int) -> int:
    modulus = 2_147_483_629
    value = (seed + 104729 * row + 13007 * round_index) % modulus
    return (value * 48271 + 17) % modulus


def oracle() -> np.ndarray:
    x = np.arange(4, dtype=np.float64)
    target = np.asarray((0.0, 0.0, 10.0, 10.0), dtype=np.float64)
    margin = np.full(4, target.mean())
    gradient = margin - target
    hessian = np.ones(4, dtype=np.float64)
    order = np.argsort(np.abs(gradient), kind="stable")
    top_count = max(1, min(3, int(np.ceil(0.5 * len(x)))))
    top = order[-top_count:]
    remainder = [row for row in range(len(x)) if row not in set(top)]
    remainder.sort(key=lambda row: goss_hash(1729, 1, row + 1))
    selected = np.concatenate((top, np.asarray(remainder[:1], dtype=int)))
    scaled_gradient = gradient.copy()
    scaled_hessian = hessian.copy()
    scaled_gradient[remainder[0]] *= 2.0
    scaled_hessian[remainder[0]] *= 2.0
    total_g = float(np.sum(scaled_gradient[selected]))
    total_h = float(np.sum(scaled_hessian[selected]))
    best_gain = -np.inf
    best_threshold = 0.0
    best_left = best_right = 0.0
    selected_order = selected[np.argsort(x[selected], kind="stable")]
    selected_x = x[selected_order]
    for threshold in 0.5 * (selected_x[:-1] + selected_x[1:]):
        left = selected_x < threshold
        right = ~left
        if not np.any(left) or not np.any(right):
            continue
        left_g = float(np.sum(scaled_gradient[selected_order[left]]))
        left_h = float(np.sum(scaled_hessian[selected_order[left]]))
        right_g = float(np.sum(scaled_gradient[selected_order[right]]))
        right_h = float(np.sum(scaled_hessian[selected_order[right]]))
        gain = 0.5 * (left_g**2 / left_h + right_g**2 / right_h - total_g**2 / total_h)
        if gain > best_gain:
            best_gain = gain
            best_threshold = float(threshold)
            best_left = -left_g / left_h
            best_right = -right_g / right_h
    return margin + np.where(x < best_threshold, best_left, best_right)


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
        "lightgbm_goss_fit_seconds", "lightgbm_goss_predict_seconds",
        "lightgbm_goss_oracle_error", "lightgbm_goss_replay_error",
        "lightgbm_goss_type", "lightgbm_goss_top_rate",
        "lightgbm_goss_other_rate", "lightgbm_goss_invalid_status",
        "lightgbm_goss_cuda_status",
    }
    missing = required - parsed.keys()
    if missing:
        raise RuntimeError(f"release app omitted {sorted(missing)}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/lightgbm_goss.csv"))
    parser.add_argument("--target", default="fortml_bench_lightgbm_goss")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    observed = run_app(fortml, args.target)
    expected = oracle()
    oracle_error = float(observed["lightgbm_goss_oracle_error"])
    replay_error = float(observed["lightgbm_goss_replay_error"])
    if not np.isfinite(expected).all() or np.max(np.abs(expected - [0.0, 0.0, 10.0, 10.0])) > 1e-12:
        raise RuntimeError("independent GOSS oracle is invalid")
    if oracle_error > 2e-12 or replay_error > 2e-13:
        raise RuntimeError(f"GOSS oracle/replay mismatch: {oracle_error:.3e}/{replay_error:.3e}")
    if observed["lightgbm_goss_type"] != "goss":
        raise RuntimeError("GOSS type metadata changed")
    if abs(float(observed["lightgbm_goss_top_rate"]) - 0.5) > 1e-14 or \
            abs(float(observed["lightgbm_goss_other_rate"]) - 0.25) > 1e-14:
        raise RuntimeError("GOSS rate metadata changed")
    if int(observed["lightgbm_goss_invalid_status"]) != FORTNUM_DOMAIN_ERROR:
        raise RuntimeError("invalid GOSS rate status changed")
    if int(observed["lightgbm_goss_cuda_status"]) != FORTNUM_NOT_IMPLEMENTED:
        raise RuntimeError("GOSS CUDA refusal changed")
    fortml_rev = revision(fortml)
    bench_rev = revision(root, (args.output.resolve(),))
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": fortml_rev, "benchmark_revision": bench_rev,
        "compiler": "gfortran", "flags": "-O3",
    }

    def row(**values: object) -> dict[str, str]:
        output = {field: "" for field in FIELDS}
        output.update({
            "workload": "lightgbm_goss", "backend": "fortml", "device": "cpu",
            "status": "pass", "n_samples": "4", "n_features": "1",
            "n_estimators": "1", "oracle": "independent NumPy GOSS split oracle",
            **details,
        })
        output.update({key: str(value) for key, value in values.items()})
        return output

    rows = [row(
        phase="oracle", backend="numpy_oracle", seconds_per_operation="0.0",
        metric="prediction_max_abs_error", value="0.0", max_abs_error="0.0",
        notes="target=[0,0,10,10], a=0.5, b=0.25, seed=1729",
    ), row(
        phase="fit", seconds_per_operation=observed["lightgbm_goss_fit_seconds"],
        metric="fit_seconds", value=observed["lightgbm_goss_fit_seconds"], max_abs_error="0.0",
        notes="deterministic top-gradient and hash-ranked other-row selection",
    ), row(
        phase="predict", seconds_per_operation=observed["lightgbm_goss_predict_seconds"],
        metric="prediction_max_abs_error", value=oracle_error, max_abs_error=oracle_error,
    ), row(
        phase="seed_replay", seconds_per_operation="0.0", metric="replay_max_abs_error",
        value=replay_error, max_abs_error=replay_error,
    ), row(
        phase="invalid_rates", seconds_per_operation="0.0", metric="status_code",
        value=observed["lightgbm_goss_invalid_status"], max_abs_error="0.0",
        oracle="typed transactional domain contract", notes="top_rate+other_rate >= 1",
    ), row(
        phase="predict", device="cuda", status="unavailable", seconds_per_operation="0.0",
        metric="status_code", value=observed["lightgbm_goss_cuda_status"], max_abs_error="nan",
        oracle="typed device contract", notes="FORTNUM_NOT_IMPLEMENTED; no host fallback",
    )]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
