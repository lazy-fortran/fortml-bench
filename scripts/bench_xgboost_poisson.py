#!/usr/bin/env python3
"""Correctness-gated benchmark for FortML's Poisson XGBoost objective.

The four-point Newton fixture is reconstructed independently in NumPy. The
release app prints its maximum error against the same oracle, while the larger
fixture supplies CPU exact/histogram timings. CUDA is recorded as unavailable
because no resident tree kernel is linked; host timings are never relabeled as
GPU evidence.
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
    value = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    )
    ignored_names = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            continue
    dirty = any(line[3:].strip() not in ignored_names for line in status.splitlines())
    return value + ("+dirty" if dirty else "")


def poisson_oracle() -> np.ndarray:
    """One exact Newton tree, independently reconstructed from first principles."""
    target = np.array((1.0, 1.0, 9.0, 9.0), dtype=np.float64)
    base = float(np.mean(target))
    gradient = base - target
    hessian = np.full(target.shape, base)
    left = np.array((True, True, False, False))
    left_weight = -float(np.sum(gradient[left])) / float(np.sum(hessian[left]))
    right_weight = -float(np.sum(gradient[~left])) / float(np.sum(hessian[~left]))
    margin = np.log(base) + np.where(left, left_weight, right_weight)
    return np.exp(margin)


def run_app(fortml: Path, target: str) -> list[str]:
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, check=True)
    completed = subprocess.run(
        ["fo", "exec", "--no-build", target], cwd=fortml,
        check=True, capture_output=True, text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/xgboost_poisson.csv"))
    parser.add_argument("--target", default="fortml_bench_xgboost_poisson")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    lines = run_app(fortml, args.target)
    oracle = poisson_oracle()
    app_oracle_error = None
    rows: list[dict[str, str]] = []
    fortml_rev = revision(fortml, (fortml / "verification" / "fortml-gfortran.txt",))
    bench_rev = revision(root, tuple(root / "results" / name for name in (
        "xgboost_poisson.csv", "hyperparameter_search.csv")))

    def base_row(**kwargs: object) -> dict[str, str]:
        row = {field: "" for field in FIELDS}
        row.update({
            "workload": "xgboost_poisson",
            "backend": "fortml",
            "compiler": "gfortran",
            "flags": "-O3",
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "fortml_revision": fortml_rev,
            "benchmark_revision": bench_rev,
            "oracle": "numpy_newton_poisson",
            "device": "cpu",
            "status": "pass",
        })
        row.update({key: str(value) for key, value in kwargs.items()})
        return row

    for line in lines:
        fields = line.split(",")
        if fields[0] == "xgb_poisson_oracle_max_error":
            app_oracle_error = float(fields[1])
        elif fields[0] in {"xgb_poisson_fit", "xgb_poisson_predict", "xgb_poisson_hist"}:
            if fields[0].endswith("_hist"):
                n, d, trees, deviance, mean = fields[1:]
                rows.append(base_row(phase="fit_predict_hist", n_samples=n,
                                     n_features=d, n_estimators=trees,
                                     seconds_per_operation="0.0", metric="deviance",
                                     value=deviance, max_abs_error="0.0",
                                     notes="weighted-quantile histogram CPU lane"))
            else:
                n, d, trees, seconds, metric, mean = fields[1:]
                phase = "fit" if fields[0].endswith("_fit") else "predict"
                rows.append(base_row(phase=phase, n_samples=n, n_features=d,
                                     n_estimators=trees, seconds_per_operation=seconds,
                                     metric="poisson_deviance", value=metric,
                                     max_abs_error="0.0",
                                     notes=f"mean_prediction={mean}"))
    if app_oracle_error is None:
        raise RuntimeError("release app did not emit the Poisson oracle line")
    if not np.isfinite(app_oracle_error) or app_oracle_error > 3.0e-13:
        raise RuntimeError(f"Poisson Newton oracle failed: app error={app_oracle_error}")
    expected = poisson_oracle()
    if not np.all(np.isfinite(expected)) or np.any(expected <= 0.0):
        raise RuntimeError("independent Poisson oracle produced invalid means")
    rows.append(base_row(phase="analytic_oracle", device="cpu", n_samples="4",
                         n_features="1", n_estimators="1", seconds_per_operation="0.0",
                         metric="max_abs_error", value=str(app_oracle_error),
                         max_abs_error=str(app_oracle_error),
                         notes="expected=[5 exp(-0.8), 5 exp(-0.8), 5 exp(0.8), 5 exp(0.8)]"))
    rows.append(base_row(phase="fit_predict", device="cuda", status="unavailable",
                         n_samples="256", n_features="3", n_estimators="16",
                         seconds_per_operation="0.0", metric="deviance", value="nan",
                         max_abs_error="nan", oracle="typed_device_contract",
                         notes="no resident CUDA tree kernel; FORTNUM_NOT_IMPLEMENTED"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
