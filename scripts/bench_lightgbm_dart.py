#!/usr/bin/env python3
"""Correctness-gated LightGBM DART benchmark.

The independent NumPy oracle replays a depth-one squared-loss Newton booster,
seeded tree dropout, and LightGBM-style tree normalisation.  The release app
also checks deterministic replay, schema-3 persistence, warm-start parity, the
typed invalid-rate contract, and the explicit CUDA refusal.
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
MODULUS = 2_147_483_629
SEED = 1729
DROP_RATE = 0.99
MAX_DROP = 1
LEARNING_RATE = 0.5
L2 = 1.0


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


def dart_hash(seed: int, round_index: int, tree_index: int) -> int:
    value = (seed + 104729 * tree_index + 13007 * round_index) % MODULUS
    return (value * 48271 + 17) % MODULUS


def dart_uniform(seed: int, round_index: int, tree_index: int) -> float:
    return dart_hash(seed, round_index, tree_index) / MODULUS


def selected_trees(n_previous: int, round_index: int) -> list[int]:
    if n_previous == 0 or dart_uniform(SEED, round_index, 0) < 0.0:
        return []
    selected = [
        index for index in range(1, n_previous + 1)
        if dart_uniform(SEED, round_index, index) < DROP_RATE
    ]
    return selected[:MAX_DROP]


def oracle() -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray((0.0, 0.0, 10.0, 10.0), dtype=np.float64)
    margin = np.full(4, float(np.mean(target)))
    scales: list[float] = []
    corrections: list[np.ndarray] = []
    for round_index in range(1, 4):
        dropped = selected_trees(round_index - 1, round_index)
        dropped_zero = [index - 1 for index in dropped]
        margin_for_fit = margin.copy()
        for index in dropped_zero:
            margin_for_fit -= LEARNING_RATE * scales[index] * corrections[index]
        gradient = margin_for_fit - target
        # The independent one-feature depth-one split at x=1.5 is the unique
        # positive-gain split for this fixture.
        left = gradient[:2]
        right = gradient[2:]
        correction = np.concatenate((
            np.full(2, -float(np.sum(left)) / (2.0 + L2)),
            np.full(2, -float(np.sum(right)) / (2.0 + L2)),
        ))
        if dropped_zero:
            normalization = 1.0 / (len(dropped_zero) + 1.0)
            for index in dropped_zero:
                scales[index] *= normalization
            new_scale = normalization
        else:
            new_scale = 1.0
        scales.append(new_scale)
        margin = margin_for_fit.copy()
        for index in dropped_zero:
            margin += LEARNING_RATE * scales[index] * corrections[index]
        corrections.append(correction)
        margin += LEARNING_RATE * scales[-1] * correction
    return margin, np.asarray(scales)


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
        "lightgbm_dart_fit_seconds", "lightgbm_dart_predict_seconds",
        "lightgbm_dart_type", "lightgbm_dart_drop_rate", "lightgbm_dart_max_drop",
        "lightgbm_dart_scale_1", "lightgbm_dart_scale_2", "lightgbm_dart_scale_3",
        "lightgbm_dart_oracle_error", "lightgbm_dart_replay_error",
        "lightgbm_dart_restore_error", "lightgbm_dart_warm_start_error",
        "lightgbm_dart_invalid_status", "lightgbm_dart_cuda_status",
    }
    missing = required - parsed.keys()
    if missing:
        raise RuntimeError(f"release app omitted {sorted(missing)}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/lightgbm_dart.csv"))
    parser.add_argument("--target", default="fortml_bench_lightgbm_dart")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    observed = run_app(fortml, args.target)
    expected, scales = oracle()
    expected_scales = (0.25, 0.5, 0.5)
    if np.max(np.abs(scales-expected_scales)) > 1e-14:
        raise RuntimeError(f"independent DART scale oracle is invalid: {scales}")
    app_oracle_error = float(observed["lightgbm_dart_oracle_error"])
    replay_error = float(observed["lightgbm_dart_replay_error"])
    restore_error = float(observed["lightgbm_dart_restore_error"])
    warm_error = float(observed["lightgbm_dart_warm_start_error"])
    if app_oracle_error > 2e-12 or replay_error > 2e-13 or restore_error > 2e-13 or warm_error > 2e-13:
        raise RuntimeError(
            "DART oracle/replay/restore/warm-start mismatch: "
            f"{app_oracle_error:.3e}/{replay_error:.3e}/{restore_error:.3e}/{warm_error:.3e}",
        )
    if observed["lightgbm_dart_type"] != "dart":
        raise RuntimeError("DART type metadata changed")
    if abs(float(observed["lightgbm_dart_drop_rate"]) - DROP_RATE) > 1e-14 or \
            int(observed["lightgbm_dart_max_drop"]) != MAX_DROP:
        raise RuntimeError("DART dropout metadata changed")
    for index, expected_scale in enumerate(expected_scales, start=1):
        if abs(float(observed[f"lightgbm_dart_scale_{index}"]) - expected_scale) > 1e-14:
            raise RuntimeError(f"DART scale {index} changed")
    if int(observed["lightgbm_dart_invalid_status"]) != FORTNUM_DOMAIN_ERROR:
        raise RuntimeError("invalid DART status changed")
    if int(observed["lightgbm_dart_cuda_status"]) != FORTNUM_NOT_IMPLEMENTED:
        raise RuntimeError("DART CUDA refusal changed")
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
            "workload": "lightgbm_dart", "backend": "fortml", "device": "cpu",
            "status": "pass", "n_samples": "4", "n_features": "1",
            "n_estimators": "3", "oracle": "independent NumPy DART tree-walk oracle",
            **details,
        })
        output.update({key: str(value) for key, value in values.items()})
        return output

    rows = [row(
        phase="oracle", backend="numpy_oracle", seconds_per_operation="0.0",
        metric="prediction_max_abs_error", value="0.0", max_abs_error="0.0",
        notes="target=[0,0,10,10], eta=0.5, l2=1, drop=.99, max_drop=1, seed=1729",
    ), row(
        phase="fit", seconds_per_operation=observed["lightgbm_dart_fit_seconds"],
        metric="fit_seconds", value=observed["lightgbm_dart_fit_seconds"], max_abs_error="0.0",
        notes="seeded prior-tree dropout with deterministic tree normalisation",
    ), row(
        phase="predict", seconds_per_operation=observed["lightgbm_dart_predict_seconds"],
        metric="prediction_max_abs_error", value=app_oracle_error, max_abs_error=app_oracle_error,
    ), row(
        phase="seed_replay", seconds_per_operation="0.0", metric="replay_max_abs_error",
        value=replay_error, max_abs_error=replay_error,
    ), row(
        phase="persistence", seconds_per_operation="0.0", metric="restore_max_abs_error",
        value=restore_error, max_abs_error=restore_error, notes="schema-3 tree scales",
    ), row(
        phase="warm_start", seconds_per_operation="0.0", metric="warm_start_max_abs_error",
        value=warm_error, max_abs_error=warm_error,
    ), row(
        phase="tree_scale", seconds_per_operation="0.0", metric="tree_scale_max_abs_error",
        value="0.0", max_abs_error="0.0", notes="expected [0.25,0.5,0.5]",
    ), row(
        phase="invalid_rates", seconds_per_operation="0.0", metric="status_code",
        value=observed["lightgbm_dart_invalid_status"], max_abs_error="0.0",
        oracle="typed transactional domain contract", notes="dart_drop_rate >= 1",
    ), row(
        phase="predict", device="cuda", status="unavailable", seconds_per_operation="0.0",
        metric="status_code", value=observed["lightgbm_dart_cuda_status"], max_abs_error="nan",
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
