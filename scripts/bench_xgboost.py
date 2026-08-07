#!/usr/bin/env python3
"""Benchmark FortML's exact depth-limited second-order boosting lane.

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
from dataclasses import dataclass
from typing import Any

import numpy as np


N_SAMPLES = 192
N_FEATURES = 3
N_ESTIMATORS = 12
MAX_DEPTH = 2
MIN_SAMPLES_LEAF = 2
LEARNING_RATE = 0.25
L1_REG = 0.15
L2_REG = 1.5
GAMMA = 0.01
MIN_CHILD_WEIGHT = 0.1
MULTICLASS_CLASSES = np.array((-1, 4, 9), dtype=np.int64)
MISSING_SAMPLES = 6

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


def revision(repository: Path, *, ignored: tuple[Path, ...] = ()) -> str:
    value = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    )
    ignored_names = set()
    for path in ignored:
        try:
            ignored_names.add(
                path.resolve().relative_to(repository.resolve()).as_posix()
            )
        except ValueError:
            continue
    dirty = any(line[3:].strip() not in ignored_names for line in status.splitlines())
    return value + ("+dirty" if dirty else "")


def package_version(name: str) -> str:
    try:
        module = __import__(name)
    except ImportError:
        return "unavailable"
    return str(getattr(module, "__version__", "installed"))


def metadata(root: Path, fortml: Path, output: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "xgboost_version": package_version("xgboost"),
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored=(output,)),
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


def multiclass_fixture(x: np.ndarray) -> np.ndarray:
    """Return the sorted arbitrary labels used by the OVR workload."""
    return np.where(
        x[:, 0] < -0.35,
        MULTICLASS_CLASSES[0],
        np.where(x[:, 0] < 0.35, MULTICLASS_CLASSES[1], MULTICLASS_CLASSES[2]),
    )


def missing_fixture() -> tuple[np.ndarray, np.ndarray]:
    """Small fixture whose default direction has an analytic exact oracle."""
    x = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [np.nan]], dtype=np.float64)
    target = np.array([0.0, 0.0, 10.0, 10.0, 10.0, 0.0], dtype=np.float64)
    return x, target


def missing_oracle() -> tuple[np.ndarray, float]:
    """Independently score finite thresholds and both NaN default branches."""
    x, target = missing_fixture()
    base = float(np.mean(target))
    gradient = base - target
    hessian = np.ones_like(target)
    finite = np.isfinite(x[:, 0])
    values = np.unique(x[finite, 0])
    total_gradient = float(np.sum(gradient))
    total_hessian = float(np.sum(hessian))
    total_score = leaf_score(total_gradient, total_hessian, 0.0, 0.0)
    best_gain = 0.0
    best_prediction = np.full(target.shape, base)
    for threshold in 0.5 * (values[:-1] + values[1:]):
        for missing_left in (True, False):
            left = finite & (x[:, 0] < threshold)
            if missing_left:
                left |= np.isnan(x[:, 0])
            right = ~left
            left_gradient = float(np.sum(gradient[left]))
            right_gradient = float(np.sum(gradient[right]))
            gain = 0.5 * (
                leaf_score(left_gradient, float(np.sum(hessian[left])), 0.0, 0.0)
                + leaf_score(right_gradient, float(np.sum(hessian[right])), 0.0, 0.0)
                - total_score
            )
            if gain > best_gain:
                best_gain = gain
                best_prediction = np.full(target.shape, base)
                best_prediction[left] += leaf_weight(
                    left_gradient, float(np.sum(hessian[left])), 0.0, 0.0
                )
                best_prediction[right] += leaf_weight(
                    right_gradient, float(np.sum(hessian[right])), 0.0, 0.0
                )
    return best_prediction, best_gain


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


@dataclass
class TreeNode:
    """One independently reconstructed exact-split tree node."""

    weight: float
    gain: float = 0.0
    feature: int = -1
    threshold: float = 0.0
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


def build_tree(
    x: np.ndarray,
    gradient: np.ndarray,
    hessian: np.ndarray,
    *,
    l1: float,
    l2: float,
    gamma: float,
    min_child_weight: float,
    max_depth: int,
    min_samples_leaf: int,
    indices: np.ndarray | None = None,
    depth: int = 0,
) -> TreeNode:
    """Reconstruct FortML's recursive exact-split tree independently.

    Samples retain their incoming order when a node is partitioned, matching
    the Fortran recursion while making the feature/threshold tie policy
    explicit in this behavioral oracle.
    """

    if indices is None:
        indices = np.arange(x.shape[0], dtype=np.int64)
    total_gradient = float(np.sum(gradient[indices]))
    total_hessian = float(np.sum(hessian[indices]))
    root_weight = leaf_weight(total_gradient, total_hessian, l1, l2)
    node = TreeNode(weight=root_weight)
    n_local = indices.size
    if depth >= max_depth or n_local < 2 * min_samples_leaf:
        return node

    best_gain = 0.0
    best: tuple[int, float] | None = None
    for feature in range(x.shape[1]):
        order = np.argsort(x[indices, feature], kind="stable")
        sorted_indices = indices[order]
        left_gradient = 0.0
        left_hessian = 0.0
        for k in range(1, n_local):
            index = sorted_indices[k - 1]
            left_gradient += float(gradient[index])
            left_hessian += float(hessian[index])
            if k < min_samples_leaf or n_local - k < min_samples_leaf:
                continue
            if x[sorted_indices[k - 1], feature] >= x[sorted_indices[k], feature]:
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
                threshold = 0.5 * (
                    x[sorted_indices[k - 1], feature] + x[sorted_indices[k], feature]
                )
                best_gain = gain
                best = (feature, threshold)
    if best is None:
        return node
    feature, threshold = best
    left_indices = indices[x[indices, feature] < threshold]
    right_indices = indices[x[indices, feature] >= threshold]
    if left_indices.size < min_samples_leaf or right_indices.size < min_samples_leaf:
        raise RuntimeError("exact-split oracle violated minimum leaf size")
    node.feature = feature
    node.threshold = threshold
    node.gain = best_gain
    node.left = build_tree(
        x,
        gradient,
        hessian,
        l1=l1,
        l2=l2,
        gamma=gamma,
        min_child_weight=min_child_weight,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        indices=left_indices,
        depth=depth + 1,
    )
    node.right = build_tree(
        x,
        gradient,
        hessian,
        l1=l1,
        l2=l2,
        gamma=gamma,
        min_child_weight=min_child_weight,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        indices=right_indices,
        depth=depth + 1,
    )
    return node


def tree_predict(x: np.ndarray, tree: TreeNode) -> np.ndarray:
    prediction = np.empty(x.shape[0], dtype=np.float64)
    for i in range(x.shape[0]):
        node = tree
        while node.left is not None:
            node = node.left if x[i, node.feature] < node.threshold else node.right
        prediction[i] = node.weight
    return prediction


def fit_boosting(
    x: np.ndarray, target: np.ndarray, *, logistic: bool
) -> tuple[np.ndarray, list[TreeNode]]:
    if logistic:
        prediction = np.full(x.shape[0], logit(float(np.mean(target))))
    else:
        prediction = np.full(x.shape[0], float(np.mean(target)))
    trees: list[TreeNode] = []
    for _ in range(N_ESTIMATORS):
        if logistic:
            probability = sigmoid(prediction)
            gradient = probability - target
            hessian = np.maximum(probability * (1.0 - probability), 1.0e-12)
        else:
            gradient = prediction - target
            hessian = np.ones_like(target)
        tree = build_tree(
            x,
            gradient,
            hessian,
            l1=0.0 if logistic else L1_REG,
            l2=1.0 if logistic else L2_REG,
            gamma=GAMMA,
            min_child_weight=MIN_CHILD_WEIGHT,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
        )
        trees.append(tree)
        prediction += LEARNING_RATE * tree_predict(x, tree)
    return prediction, trees


def root_diagnostics(tree: TreeNode) -> tuple[float, float, float, float, float]:
    """Return the public first-tree diagnostics reported by the Fortran app."""
    if tree.left is None or tree.right is None:
        return 0.0, 0.0, tree.weight, tree.weight, tree.gain
    return (
        float(tree.feature),
        tree.threshold,
        tree.left.weight,
        tree.right.weight,
        tree.gain,
    )


def multiclass_oracle(x: np.ndarray) -> tuple[float, float]:
    """Reconstruct one exact binary booster and OVR normalization per class."""
    labels = multiclass_fixture(x)
    positive = []
    for class_label in MULTICLASS_CLASSES:
        margin, _trees = fit_boosting(
            x, (labels == class_label).astype(np.float64), logistic=True
        )
        positive.append(sigmoid(margin))
    matrix = np.column_stack(positive)
    probabilities = matrix / np.sum(matrix, axis=1, keepdims=True)
    predicted = MULTICLASS_CLASSES[np.argmax(probabilities, axis=1)]
    return float(np.mean(predicted == labels)), float(np.sum(probabilities))


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
        "xgb_multiclass_fit",
        "xgb_multiclass_predict",
        "xgb_missing_fit",
        "xgb_missing_predict",
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
    multiclass_accuracy, multiclass_probability_sum = multiclass_oracle(x)
    missing_prediction, missing_gain = missing_oracle()
    regression_root = root_diagnostics(regression_trees[0])
    logistic_root = root_diagnostics(logistic_trees[0])
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
        abs(float(parsed["xgb_regression_fit"][5]) - regression_root[4]),
        abs(
            float(parsed["xgb_regression_fit"][6])
            - regression_root[2]
            - regression_root[3]
        ),
        abs(float(parsed["xgb_regression_predict"][4]) - regression_mse),
        abs(float(parsed["xgb_regression_predict"][5]) - np.sum(regression_prediction)),
    )
    logistic_error = max(
        abs(float(parsed["xgb_logistic_fit"][4]) - logistic_logloss),
        abs(float(parsed["xgb_logistic_fit"][5]) - logistic_root[4]),
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
    multiclass_error = max(
        abs(float(parsed["xgb_multiclass_fit"][4]) - multiclass_accuracy),
        abs(float(parsed["xgb_multiclass_fit"][5]) - multiclass_probability_sum),
        abs(float(parsed["xgb_multiclass_predict"][4]) - multiclass_accuracy),
        abs(float(parsed["xgb_multiclass_predict"][5]) - multiclass_probability_sum),
    )
    if multiclass_error > 2.0e-11:
        raise RuntimeError(
            f"FortML XGBoost multiclass oracle mismatch: {multiclass_error:.3e}"
        )
    missing_fit_values = np.asarray(
        [float(value) for value in parsed["xgb_missing_fit"][6:12]], dtype=np.float64
    )
    missing_predict_values = np.asarray(
        [float(value) for value in parsed["xgb_missing_predict"][6:12]], dtype=np.float64
    )
    missing_error = max(
        float(np.max(np.abs(missing_fit_values - missing_prediction))),
        float(np.max(np.abs(missing_predict_values - missing_prediction))),
        abs(float(parsed["xgb_missing_fit"][4]) - np.sum(missing_prediction)),
        abs(float(parsed["xgb_missing_predict"][4]) - np.sum(missing_prediction)),
        abs(float(parsed["xgb_missing_fit"][5]) - missing_gain),
        abs(float(parsed["xgb_missing_predict"][5]) - missing_gain),
    )
    if missing_error > 2.0e-11:
        raise RuntimeError(f"FortML XGBoost missing-value oracle mismatch: {missing_error:.3e}")
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
            oracle="independent NumPy exact recursive second-order tree search",
            notes=(
                "depth=2; L1/L2/gamma/min-child-Hessian/shrinkage matched; "
                "recursive exact split"
            ),
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
            oracle="independent NumPy exact recursive second-order tree search",
            notes="depth=2 recursive tree; piecewise-constant input JVP away from splits",
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
            oracle="independent NumPy recursive logistic Newton tree search",
            notes=(
                "depth=2; stable sigmoid, clipped base logit, exact Hessian aggregation"
            ),
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
            oracle="independent NumPy recursive logistic Newton tree search",
            notes="probability output and two-column probability product checked",
        ),
        row(
            metadata_values,
            workload="xgboost_multiclass",
            phase="fit",
            backend="fortml",
            status="pass",
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_estimators=N_ESTIMATORS,
            seconds_per_operation=float(parsed["xgb_multiclass_fit"][3]),
            metric="accuracy",
            value=multiclass_accuracy,
            max_abs_error=multiclass_error,
            oracle="independent NumPy one-vs-rest recursive second-order tree search",
            notes="sorted labels=-1,4,9; normalized positive OVR probabilities",
        ),
        row(
            metadata_values,
            workload="xgboost_multiclass",
            phase="predict",
            backend="fortml",
            status="pass",
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_estimators=N_ESTIMATORS,
            seconds_per_operation=float(parsed["xgb_multiclass_predict"][3]),
            metric="accuracy",
            value=multiclass_accuracy,
            max_abs_error=multiclass_error,
            oracle="independent NumPy one-vs-rest recursive second-order tree search",
            notes="simplex sum checked at the independent row-normalization oracle",
        ),
        row(
            metadata_values,
            workload="xgboost_missing",
            phase="fit",
            backend="fortml",
            status="pass",
            n_samples=MISSING_SAMPLES,
            n_features=1,
            n_estimators=1,
            seconds_per_operation=float(parsed["xgb_missing_fit"][3]),
            metric="prediction_sum",
            value=float(np.sum(missing_prediction)),
            max_abs_error=missing_error,
            oracle="independent NumPy exact threshold/default-direction oracle",
            notes="learned NaN default direction; finite values plus one IEEE NaN",
        ),
        row(
            metadata_values,
            workload="xgboost_missing",
            phase="predict",
            backend="fortml",
            status="pass",
            n_samples=MISSING_SAMPLES,
            n_features=1,
            n_estimators=1,
            seconds_per_operation=float(parsed["xgb_missing_predict"][3]),
            metric="prediction_sum",
            value=float(np.sum(missing_prediction)),
            max_abs_error=missing_error,
            oracle="independent NumPy exact threshold/default-direction oracle",
            notes="stored default branch reused by prediction; infinities remain refused",
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


def unsupported_policy_rows(metadata_values: dict[str, str]) -> list[dict[str, Any]]:
    """Keep unimplemented histogram-family contracts visible in release CSVs."""
    return [
        row(
            metadata_values,
            workload="xgboost_histogram",
            phase="capability_check",
            backend="fortml",
            status="unavailable",
            oracle="declared capability boundary",
            notes="weighted quantile/histogram growth is not implemented in the exact backend",
        ),
        row(
            metadata_values,
            workload="lightgbm_histogram",
            phase="capability_check",
            backend="fortml",
            status="unavailable",
            oracle="declared capability boundary",
            notes="LightGBM leaf-wise histogram, GOSS, and EFB policies remain planned",
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/xgboost_workloads.csv")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    metadata_values = metadata(root, fortml, args.output)
    records = run_fortran(root, fortml, metadata_values)
    records.append(optional_xgboost_row(metadata_values))
    records.extend(unsupported_policy_rows(metadata_values))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()
