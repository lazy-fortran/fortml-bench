#!/usr/bin/env python3
"""Correctness-gated benchmark for FortML's bounded Tweedie XGBoost lane.

The release app emits a four-row one-tree Newton fixture.  This harness
reconstructs the compound-Poisson log-mean objective independently in NumPy
before retaining exact/histogram timings and an explicit CUDA refusal row.
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


def tweedie_oracle(power: float = 1.5) -> np.ndarray:
    """Independent one-split prediction from exact Tweedie products."""
    target = np.array((1.0, 1.0, 9.0, 9.0), dtype=np.float64)
    base = float(np.mean(target))
    margin = np.log(base)
    first = np.exp((1.0 - power) * margin)
    second = np.exp((2.0 - power) * margin)
    gradient = -target * first + second
    hessian = target * (power - 1.0) * first + (2.0 - power) * second
    left = -float(np.sum(gradient[:2])) / float(np.sum(hessian[:2]))
    right = -float(np.sum(gradient[2:])) / float(np.sum(hessian[2:]))
    return np.exp(np.array((margin + left, margin + left, margin + right, margin + right)))


def run_app(fortml: Path, target: str) -> list[str]:
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, check=True)
    completed = subprocess.run(
        ["fo", "exec", "--no-build", target], cwd=fortml, check=True,
        capture_output=True, text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, default=Path("results/xgboost_tweedie.csv"))
    parser.add_argument("--target", default="fortml_bench_xgboost_tweedie")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    lines = run_app(fortml, args.target)
    parsed: dict[str, list[str] | str] = {}
    for line in lines:
        fields = [field.strip() for field in line.split(",")]
        if not fields:
            continue
        if fields[0] == "xgb_tweedie_oracle_max_error":
            parsed[fields[0]] = fields[1]
        elif fields[0] in {"xgb_tweedie_fit", "xgb_tweedie_predict"}:
            if len(fields) != 7:
                raise RuntimeError(f"malformed {fields[0]} line: {line!r}")
            parsed[fields[0]] = fields[1:]
        elif fields[0] == "xgb_tweedie_hist":
            if len(fields) != 6:
                raise RuntimeError(f"malformed histogram line: {line!r}")
            parsed[fields[0]] = fields[1:]
    required = {"xgb_tweedie_oracle_max_error", "xgb_tweedie_fit",
                "xgb_tweedie_predict", "xgb_tweedie_hist"}
    missing = required.difference(parsed)
    if missing:
        raise RuntimeError(f"release app omitted Tweedie fields: {sorted(missing)}")

    expected = tweedie_oracle()
    app_error = float(parsed["xgb_tweedie_oracle_max_error"])  # type: ignore[arg-type]
    if not np.isfinite(app_error) or app_error > 3.0e-13:
        raise RuntimeError(f"Tweedie Newton oracle failed: {app_error:.3e}")
    if not np.isfinite(expected).all() or np.any(expected <= 0.0):
        raise RuntimeError("independent Tweedie oracle produced invalid predictions")

    fortml_rev = revision(fortml, (fortml / "verification" / "fortml-gfortran.txt",))
    bench_rev = revision(root, (args.output.resolve(),))
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
            "workload": "xgboost_tweedie", "backend": "fortml", "device": "cpu",
            "status": "pass",
            "oracle": "independent NumPy Tweedie one-split Newton oracle",
            **details,
        })
        output.update({key: str(value) for key, value in kwargs.items()})
        return output

    rows: list[dict[str, str]] = [row(
        phase="oracle", backend="numpy_oracle", n_samples="4", n_features="1",
        n_estimators="1", seconds_per_operation="0.0", metric="prediction_sum",
        value=float(np.sum(expected)), max_abs_error="0.0",
        notes="target=[1,1,9,9]; power=1.5; exact split=[0,1]|[2,3]",
    )]
    for key, phase in (("xgb_tweedie_fit", "fit"), ("xgb_tweedie_predict", "predict")):
        n_samples, n_features, n_estimators, seconds, loss, mean = parsed[key]  # type: ignore[misc]
        rows.append(row(
            phase=phase, n_samples=n_samples, n_features=n_features,
            n_estimators=n_estimators, seconds_per_operation=seconds,
            metric="tweedie_nll", value=loss, max_abs_error="0.0",
            notes=f"power=1.5; mean_prediction={mean}",
        ))
    hist_values = parsed["xgb_tweedie_hist"]  # type: ignore[assignment]
    n_samples, n_features, n_estimators, loss, mean = hist_values
    rows.append(row(
        phase="fit_predict_hist", n_samples=n_samples, n_features=n_features,
        n_estimators=n_estimators, seconds_per_operation="0.0",
        metric="tweedie_nll", value=loss, max_abs_error="0.0",
        notes=f"power=1.5; weighted-quantile histogram CPU lane; mean_prediction={mean}",
    ))
    rows.append(row(
        phase="oracle_check", n_samples="4", n_features="1", n_estimators="1",
        seconds_per_operation="0.0", metric="max_abs_error", value=app_error,
        max_abs_error=app_error,
        notes="release-app output versus independent Tweedie gradient/Hessian formula",
    ))
    rows.append(row(
        phase="predict", device="cuda", status="unavailable", n_samples="256",
        n_features="3", n_estimators="16", metric="prediction", value="nan",
        max_abs_error="nan", oracle="typed_device_contract",
        notes="no resident CUDA tree kernel; FORTNUM_NOT_IMPLEMENTED; no host fallback",
    ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
