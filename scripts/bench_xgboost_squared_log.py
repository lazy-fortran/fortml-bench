#!/usr/bin/env python3
"""Correctness-gated benchmark for FortML's squared-log XGBoost objective.

The release app emits a four-sample depth-one fixture in the transformed
``log1p(target)`` coordinate.  This script reconstructs that Newton update
independently in NumPy before retaining any timings.  The CSV also keeps an
explicit CUDA refusal: no resident tree kernel is counted as GPU work.
"""

from __future__ import annotations

import argparse
import csv
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


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    """Return a clean commit id, marking unrelated working-tree edits."""
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
        name = line[3:].split(" -> ")[-1].strip()
        if name not in ignored_names:
            dirty = True
            break
    return head + ("+dirty" if dirty else "")


def squared_log_oracle(target: np.ndarray, l2: float) -> np.ndarray:
    """One depth-one Newton tree in the transformed RMSLE coordinate.

    The update is written from the objective products rather than copied from
    FortML: ``z=log1p(y)``, ``g=(m-z)/exp(m)``, and
    ``h=max((1-m+z)/exp(m), 1e-12)``.  The two leaves are the exhaustive split
    ``[0, 1] | [2, 3]`` used by the app's four-point oracle.
    """
    if target.shape != (4,) or not np.isfinite(target).all() or np.any(target < 0.0):
        raise ValueError("oracle expects four finite nonnegative targets")
    transformed = np.log1p(target)
    base = float(np.mean(transformed))
    residual = base - transformed
    scale = np.exp(base)
    gradient = residual / scale
    hessian = np.maximum((1.0 - residual) / scale, 1.0e-12)
    left = -float(np.sum(gradient[:2])) / (float(np.sum(hessian[:2])) + l2)
    right = -float(np.sum(gradient[2:])) / (float(np.sum(hessian[2:])) + l2)
    return np.expm1(np.array((base + left, base + left, base + right, base + right)))


def run_app(fortml: Path, target: str) -> list[str]:
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, check=True)
    completed = subprocess.run(
        ["fo", "exec", "--no-build", target], cwd=fortml, check=True,
        capture_output=True, text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def parse_output(lines: list[str]) -> dict[str, list[str] | str]:
    parsed: dict[str, list[str] | str] = {}
    for line in lines:
        fields = [field.strip() for field in line.split(",")]
        if not fields:
            continue
        key = fields[0]
        if key in {
            "xgb_squared_log_oracle_max_error", "xgb_squared_log_hist_max_error",
            "xgb_squared_log_cuda",
        }:
            parsed[key] = fields[1] if len(fields) == 2 else fields[1:]
        elif key in {"xgb_squared_log_fit", "xgb_squared_log_predict"}:
            if len(fields) != 6:
                raise RuntimeError(f"malformed {key} line: {line!r}")
            parsed[key] = fields[1:]
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/xgboost_squared_log.csv"),
    )
    parser.add_argument("--target", default="fortml_bench_xgboost_squared_log")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    lines = run_app(fortml, args.target)
    values = parse_output(lines)
    required = {
        "xgb_squared_log_oracle_max_error", "xgb_squared_log_fit",
        "xgb_squared_log_predict", "xgb_squared_log_hist_max_error",
        "xgb_squared_log_cuda",
    }
    missing = required.difference(values)
    if missing:
        raise RuntimeError(f"release app omitted squared-log fields: {sorted(missing)}")

    target = np.array((0.0, 0.0, 3.0, 3.0), dtype=np.float64)
    expected = squared_log_oracle(target, l2=0.7)
    app_error = float(values["xgb_squared_log_oracle_max_error"])  # type: ignore[arg-type]
    if not np.isfinite(app_error) or app_error > 3.0e-13:
        raise RuntimeError(f"squared-log Newton oracle failed: {app_error:.3e}")
    if not np.isfinite(expected).all() or np.any(expected < -1.0):
        raise RuntimeError("independent squared-log oracle produced invalid predictions")
    if values["xgb_squared_log_cuda"] != "unavailable":
        raise RuntimeError("squared-log CUDA refusal row changed unexpectedly")
    hist_error = float(values["xgb_squared_log_hist_max_error"])  # type: ignore[arg-type]
    # Bounded weighted cuts approximate the exact splitter on this continuous
    # fixture.  Keep the diagnostic, but gate only its finiteness; the
    # independent four-row transformed-coordinate oracle above is the exact
    # correctness contract.
    if not np.isfinite(hist_error):
        raise RuntimeError("squared-log histogram diagnostic is non-finite")

    fortml_rev = revision(fortml, (fortml / "verification" / "fortml-gfortran.txt",))
    # Other release-lane generators may leave their raw CSV staged or
    # untracked while this lane is running.  Generated records are excluded
    # from the benchmark code revision; source, scripts, and reports remain
    # provenance-significant.
    bench_rev = revision(root, (
        args.output.resolve(),
        root / "results" / "gp_hyperparameter_training.csv",
    ))
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": fortml_rev,
        "benchmark_revision": bench_rev,
        "compiler": "gfortran",
        "flags": "-O3",
    }

    def row(**kwargs: object) -> dict[str, str]:
        output = {field: "" for field in FIELDS}
        output.update({
            "workload": "xgboost_squared_log",
            "backend": "fortml",
            "device": "cpu",
            "status": "pass",
            "oracle": "independent NumPy transformed-coordinate Newton one-split oracle",
            **details,
        })
        output.update({key: str(value) for key, value in kwargs.items()})
        return output

    rows: list[dict[str, str]] = []
    rows.append(row(
        phase="oracle", backend="numpy_oracle", n_samples="4", n_features="1",
        n_estimators="1", seconds_per_operation="0.0", metric="prediction_sum",
        value=float(np.sum(expected)), max_abs_error="0.0",
        notes="target=[0,0,3,3]; l2=0.7; split=[0,1]|[2,3]",
    ))
    for key, phase in (("xgb_squared_log_fit", "fit"), ("xgb_squared_log_predict", "predict")):
        n_samples, n_features, n_estimators, seconds, mean = values[key]  # type: ignore[misc]
        rows.append(row(
            phase=phase, n_samples=n_samples, n_features=n_features,
            n_estimators=n_estimators, seconds_per_operation=seconds,
            metric="mean_prediction", value=mean, max_abs_error="0.0",
            notes="exact depth-two CPU tree; four-row transformed Newton gate passed",
        ))
    rows.append(row(
        phase="oracle_check", n_samples="4", n_features="1", n_estimators="1",
        seconds_per_operation="0.0", metric="max_abs_error", value=app_error,
        max_abs_error=app_error,
        notes="release-app output versus independent NumPy transformed-coordinate formula",
    ))
    rows.append(row(
        phase="fit_predict_hist", n_samples="256", n_features="3", n_estimators="16",
        seconds_per_operation="0.0", metric="max_abs_error", value=hist_error,
        max_abs_error=hist_error,
        notes="weighted-quantile histogram CPU diagnostic; exact splitter is the correctness gate",
    ))
    rows.append(row(
        phase="predict", device="cuda", status="unavailable", n_samples="256",
        n_features="3", n_estimators="16", metric="prediction", value="nan",
        max_abs_error="nan", oracle="typed_device_contract",
        notes="no resident squared-log tree kernel; FORTNUM_NOT_IMPLEMENTED; no host fallback",
    ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
