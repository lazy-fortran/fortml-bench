#!/usr/bin/env python3
"""Benchmark FortML's exact depth-one second-order boosting lane.

The Fortran workload is checked against an independent NumPy implementation of
the XGBoost leaf-weight and split-gain formulas before its timings are written.
The optional xgboost package is recorded as a contextual availability row; its
histogram, tie, and regularisation policies are not silently treated as a
bitwise oracle for this exact-split fixture.
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


N_SAMPLES = 192
N_FEATURES = 3
N_ESTIMATORS = 12
MIN_SAMPLES_LEAF = 2
LEARNING_RATE = 0.25
L1_REG = 0.15
L2_REG = 1.5
GAMMA = 0.01
MIN_CHILD_WEIGHT = 0.1

FIELDS = (
    "workload",
    "phase",
    "backend",
    "device",
    "status",
    "n_samples",
    "n_features",
    "n_estimators",
    "seconds_per_operation",
    "metric",
    "value",
    "max_abs_error",
    "oracle",
    "python_version",
    "numpy_version",
    "xgboost_version",
    "fortml_revision",
    "benchmark_revision",
    "compiler",
    "flags",
    "notes",
)


def revision(repository: Path) -> str:
    value = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).strip()
    return value + ("+dirty" if dirty else "")


def package_version(name: str) -> str:
    try:
        module = __import__(name)
    except ImportError:
        return "unavailable"
    return str(getattr(module, "__version__", "installed"))


def metadata(root: Path, fortml: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "xgboost_version": package_version("xgboost"),
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root),
        "compiler": "gfortran",
        "flags": "-O3",
        "device": "cpu",
    }


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.empty((N_SAMPLES, N_FEATURES), dtype=np.float64)
    regression = np.empty(N_SAMPLES, dtype=np.float64)
    for i in range(1, N_SAMPLES + 1):
        x[i - 1, 0] = -1.0 + 2.0 * (i - 1) / (N_SAMPLES - 1)
        x[i - 1, 1] = np.sin(0.09 * i)
        x[i - 1, 2] = np.cos(0.04 * i)
        regression[i - 1] = np.where(
            x[i - 1, 0] >= 0.08,
            1.5 + 0.25 * x[i - 1, 1],
            -0.7 + 0.12 * x[i - 1, 2],
        )
    labels = (x[:, 0] + 0.2 * x[:, 1] >= 0.0).astype(np.float64)
    return x, regression, labels


def sigmoid(value: np.ndarray) -> np.ndarray:
    output = np.empty_like(value)
    positive = value >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def logit(probability: float) -> float:
    clipped = np.clip(probability, 1.0e-12, 1.0 - 1.0e-12)
    return float(np.log(clipped) - np.log(1.0 - clipped))


def leaf_weight(gradient: float, hessian: float, l1: float, l2: float) -> float:
    thresholded = max(abs(gradient) - l1, 0.0)
    if thresholded == 0.0:
        return 0.0
    return -np.copysign(thresholded, gradient) / (hessian + l2)


def leaf_score(gradient: float, hessian: float, l1: float, l2: float) -> float:
    thresholded = max(abs(gradient) - l1, 0.0)
    return thresholded**2 / (hessian + l2)


def build_stump(
    x: np.ndarray,
    gradient: np.ndarray,
    hessian: np.ndarray,
    *,
    l1: float,
    l2: float,
    gamma: float,
    min_child_weight: float,
) -> tuple[int, float, float, float, float]:
    total_gradient = float(np.sum(gradient))
    total_hessian = float(np.sum(hessian))
    root_weight = leaf_weight(total_gradient, total_hessian, l1, l2)
    best_gain = 0.0
    best: tuple[int, float, float, float, float] | None = None
    for feature in range(x.shape[1]):
        order = np.argsort(x[:, feature], kind="stable")
        left_gradient = 0.0
        left_hessian = 0.0
        for k in range(1, x.shape[0]):
            index = order[k - 1]
            left_gradient += float(gradient[index])
            left_hessian += float(hessian[index])
            if k < MIN_SAMPLES_LEAF or x.shape[0] - k < MIN_SAMPLES_LEAF:
                continue
            if x[order[k - 1], feature] >= x[order[k], feature]:
                continue
            right_gradient = total_gradient - left_gradient
            right_hessian = total_hessian - left_hessian
            if left_hessian < min_child_weight or right_hessian < min_child_weight:
                continue
            gain = (
                0.5
                * (
                    leaf_score(left_gradient, left_hessian, l1, l2)
                    + leaf_score(right_gradient, right_hessian, l1, l2)
                    - leaf_score(total_gradient, total_hessian, l1, l2)
                )
                - gamma
            )
            if gain > best_gain:
                threshold = 0.5 * (x[order[k - 1], feature] + x[order[k], feature])
                best_gain = gain
                best = (
                    feature,
                    threshold,
                    leaf_weight(left_gradient, left_hessian, l1, l2),
                    leaf_weight(right_gradient, right_hessian, l1, l2),
                    gain,
                )
    if best is None:
        return 0, 0.0, root_weight, root_weight, 0.0
    return best


def stump_predict(
    x: np.ndarray, feature: int, threshold: float, left: float, right: float
) -> np.ndarray:
    if threshold == 0.0 and left == right:
        return np.full(x.shape[0], left)
    return np.where(x[:, feature] < threshold, left, right)


def fit_boosting(
    x: np.ndarray, target: np.ndarray, *, logistic: bool
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    if logistic:
        prediction = np.full(x.shape[0], logit(float(np.mean(target))))
    else:
        prediction = np.full(x.shape[0], float(np.mean(target)))
    trees: list[tuple[int, float, float, float, float]] = []
    for _ in range(N_ESTIMATORS):
        if logistic:
            probability = sigmoid(prediction)
            gradient = probability - target
            hessian = np.maximum(probability * (1.0 - probability), 1.0e-12)
        else:
            gradient = prediction - target
            hessian = np.ones_like(target)
        tree = build_stump(
            x,
            gradient,
            hessian,
            l1=0.0 if logistic else L1_REG,
            l2=1.0 if logistic else L2_REG,
            gamma=GAMMA,
            min_child_weight=MIN_CHILD_WEIGHT,
        )
        trees.append(tree)
        prediction += LEARNING_RATE * stump_predict(x, *tree[:4])
    return prediction, trees


def row(metadata_values: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(metadata_values)
    result.update(values)
    return result


def parse_fortran(stdout: str) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and fields[0].startswith("xgb_"):
            rows[fields[0]] = fields[1:]
    expected = {
        "xgb_regression_fit",
        "xgb_regression_predict",
        "xgb_logistic_fit",
        "xgb_logistic_predict",
    }
    if expected - rows.keys():
        raise RuntimeError(f"FortML app omitted {sorted(expected - rows.keys())}")
    return rows


def run_fortran(
    root: Path, fortml: Path, metadata_values: dict[str, str]
) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment, check=True
    )
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_xgboost"],
        cwd=fortml,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    parsed = parse_fortran(completed.stdout)
    x, regression, labels = fixture()
    regression_prediction, regression_trees = fit_boosting(
        x, regression, logistic=False
    )
    logistic_prediction, logistic_trees = fit_boosting(x, labels, logistic=True)
    regression_mse = float(np.mean((regression_prediction - regression) ** 2))
    logistic_probability = sigmoid(logistic_prediction)
    logistic_logloss = float(
        -np.mean(
            labels * np.log(np.maximum(logistic_probability, 1.0e-15))
            + (1.0 - labels) * np.log(np.maximum(1.0 - logistic_probability, 1.0e-15))
        )
    )
    logistic_accuracy = float(np.mean((logistic_probability >= 0.5) == (labels >= 0.5)))
    regression_error = max(
        abs(float(parsed["xgb_regression_fit"][4]) - regression_mse),
        abs(float(parsed["xgb_regression_fit"][5]) - regression_trees[0][4]),
        abs(
            float(parsed["xgb_regression_fit"][6])
            - regression_trees[0][2]
            - regression_trees[0][3]
        ),
        abs(float(parsed["xgb_regression_predict"][4]) - regression_mse),
        abs(float(parsed["xgb_regression_predict"][5]) - np.sum(regression_prediction)),
    )
    logistic_error = max(
        abs(float(parsed["xgb_logistic_fit"][4]) - logistic_logloss),
        abs(float(parsed["xgb_logistic_fit"][5]) - logistic_trees[0][4]),
        abs(float(parsed["xgb_logistic_fit"][6]) - np.sum(logistic_probability)),
        abs(float(parsed["xgb_logistic_predict"][4]) - logistic_logloss),
        abs(float(parsed["xgb_logistic_predict"][5]) - logistic_accuracy),
    )
    if regression_error > 2.0e-11:
        raise RuntimeError(
            f"FortML XGBoost regression oracle mismatch: {regression_error:.3e}"
        )
    if logistic_error > 2.0e-11:
        raise RuntimeError(
            f"FortML XGBoost logistic oracle mismatch: {logistic_error:.3e}"
        )
    return [
        row(
            metadata_values,
            workload="xgboost_squared",
            phase="fit",
            backend="fortml",
            status="pass",
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_estimators=N_ESTIMATORS,
            seconds_per_operation=float(parsed["xgb_regression_fit"][3]),
            metric="mse",
            value=regression_mse,
            max_abs_error=regression_error,
            oracle="independent NumPy exact second-order stump search",
            notes="depth=1; L1/L2/gamma/min-child-Hessian/shrinkage matched",
        ),
        row(
            metadata_values,
            workload="xgboost_squared",
            phase="predict",
            backend="fortml",
            status="pass",
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_estimators=N_ESTIMATORS,
            seconds_per_operation=float(parsed["xgb_regression_predict"][3]),
            metric="mse",
            value=regression_mse,
            max_abs_error=regression_error,
            oracle="independent NumPy exact second-order stump search",
            notes="piecewise-constant input JVP is available away from splits",
        ),
        row(
            metadata_values,
            workload="xgboost_logistic",
            phase="fit",
            backend="fortml",
            status="pass",
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_estimators=N_ESTIMATORS,
            seconds_per_operation=float(parsed["xgb_logistic_fit"][3]),
            metric="logloss",
            value=logistic_logloss,
            max_abs_error=logistic_error,
            oracle="independent NumPy logistic Newton stump search",
            notes="stable sigmoid, clipped base logit, exact Hessian aggregation",
        ),
        row(
            metadata_values,
            workload="xgboost_logistic",
            phase="predict",
            backend="fortml",
            status="pass",
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_estimators=N_ESTIMATORS,
            seconds_per_operation=float(parsed["xgb_logistic_predict"][3]),
            metric="accuracy",
            value=logistic_accuracy,
            max_abs_error=logistic_error,
            oracle="independent NumPy logistic Newton stump search",
            notes="probability output and two-column probability product checked",
        ),
    ]


def optional_xgboost_row(metadata_values: dict[str, str]) -> dict[str, Any]:
    version = metadata_values["xgboost_version"]
    return row(
        metadata_values,
        workload="xgboost_reference",
        phase="dependency_check",
        backend="xgboost",
        status="available_not_timed" if version != "unavailable" else "unavailable",
        oracle="dependency availability",
        notes=(
            "Optional package is recorded as context; exact Fortran split policy "
            "is the measured oracle lane."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/xgboost_workloads.csv")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    metadata_values = metadata(root, fortml)
    records = run_fortran(root, fortml, metadata_values)
    records.append(optional_xgboost_row(metadata_values))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()
