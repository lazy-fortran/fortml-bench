#!/usr/bin/env python3
"""Benchmark FortML's exact and weighted-histogram boosting lanes.

The Fortran workload is checked against independent NumPy implementations of
the exact and weighted-quantile XGBoost leaf-weight and split-gain formulas
before timings are written. The optional xgboost package is recorded as a
contextual availability row; its histogram, tie, and regularisation policies
are not silently treated as a bitwise oracle for these fixtures.
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
HIST_SAMPLES = 6
HIST_FEATURES = 1
HIST_MAX_BIN = 2
HIST_WEIGHTS = np.array((1.0, 1.0, 1.0, 1.0, 5.0, 5.0), dtype=np.float64)

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
    ignored = tuple(root / "results" / name for name in (
        "knn.csv", "rmsprop.csv", "xgboost_workloads.csv",
        "gp_classification_training.csv", "adamw_beta_hypergradient.csv",
        "cuda_adamw.csv", "ridge.csv"))
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "xgboost_version": package_version("xgboost"),
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored=ignored),
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


def weighted_histogram_cuts(
    values: np.ndarray,
    indices: np.ndarray,
    sample_weight: np.ndarray,
    max_bin: int,
) -> list[int]:
    """Return the Fortran weighted-quantile cut positions independently."""
    order = np.argsort(values[indices], kind="stable")
    sorted_indices = indices[order]
    n = sorted_indices.size
    n_bins = min(max_bin, n)
    total_weight = float(np.sum(sample_weight[sorted_indices]))
    cuts: list[int] = []
    for bin_index in range(1, n_bins):
        target = total_weight * float(bin_index) / float(n_bins)
        cumulative = 0.0
        position = n
        for k in range(n - 1):
            cumulative += float(sample_weight[sorted_indices[k]])
            if cumulative >= target:
                position = k
                break
        if position >= n - 1:
            continue
        if values[sorted_indices[position]] >= values[sorted_indices[position + 1]]:
            continue
        if not cuts or cuts[-1] != position:
            cuts.append(position)
    return cuts


def build_hist_tree(
    x: np.ndarray,
    gradient: np.ndarray,
    hessian: np.ndarray,
    sample_weight: np.ndarray,
    *,
    l1: float,
    l2: float,
    gamma: float,
    min_child_weight: float,
    max_depth: int,
    min_samples_leaf: int,
    max_bin: int,
    indices: np.ndarray | None = None,
    depth: int = 0,
) -> TreeNode:
    """Independent weighted-quantile histogram tree oracle."""
    if indices is None:
        indices = np.arange(x.shape[0], dtype=np.int64)
    total_gradient = float(np.sum(gradient[indices]))
    total_hessian = float(np.sum(hessian[indices]))
    node = TreeNode(weight=leaf_weight(total_gradient, total_hessian, l1, l2))
    n_local = indices.size
    if depth >= max_depth or n_local < 2 * min_samples_leaf:
        return node

    best_gain = 0.0
    best: tuple[int, float] | None = None
    for feature in range(x.shape[1]):
        order = np.argsort(x[indices, feature], kind="stable")
        sorted_indices = indices[order]
        cuts = set(
            weighted_histogram_cuts(
                x[:, feature], indices, sample_weight, max_bin
            )
        )
        left_gradient = 0.0
        left_hessian = 0.0
        for k in range(1, n_local):
            index = sorted_indices[k - 1]
            left_gradient += float(gradient[index])
            left_hessian += float(hessian[index])
            if k - 1 not in cuts:
                continue
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
                best_gain = gain
                best = (
                    feature,
                    0.5
                    * (
                        x[sorted_indices[k - 1], feature]
                        + x[sorted_indices[k], feature]
                    ),
                )
    if best is None:
        return node
    feature, threshold = best
    left_indices = indices[x[indices, feature] < threshold]
    right_indices = indices[x[indices, feature] >= threshold]
    if left_indices.size < min_samples_leaf or right_indices.size < min_samples_leaf:
        raise RuntimeError("histogram oracle violated minimum leaf size")
    node.feature = feature
    node.threshold = threshold
    node.gain = best_gain
    node.left = build_hist_tree(
        x,
        gradient,
        hessian,
        sample_weight,
        l1=l1,
        l2=l2,
        gamma=gamma,
        min_child_weight=min_child_weight,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_bin=max_bin,
        indices=left_indices,
        depth=depth + 1,
    )
    node.right = build_hist_tree(
        x,
        gradient,
        hessian,
        sample_weight,
        l1=l1,
        l2=l2,
        gamma=gamma,
        min_child_weight=min_child_weight,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_bin=max_bin,
        indices=right_indices,
        depth=depth + 1,
    )
    return node


def hist_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(HIST_SAMPLES, dtype=np.float64).reshape(-1, 1)
    regression = np.array((0.0, 0.0, 0.0, 10.0, 10.0, 10.0), dtype=np.float64)
    labels = np.array((0.0, 0.0, 0.0, 0.0, 1.0, 1.0), dtype=np.float64)
    return x, regression, labels


def fit_hist_boosting(
    x: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
    *,
    logistic: bool,
) -> tuple[np.ndarray, TreeNode]:
    weight_sum = float(np.sum(sample_weight))
    if logistic:
        margin = np.full(x.shape[0], logit(float(np.sum(sample_weight * target) / weight_sum)))
        probability = sigmoid(margin)
        gradient = sample_weight * (probability - target)
        hessian = sample_weight * np.maximum(probability * (1.0 - probability), 1.0e-12)
        l1, l2 = 0.0, 0.0
    else:
        margin = np.full(x.shape[0], float(np.sum(sample_weight * target) / weight_sum))
        gradient = sample_weight * (margin - target)
        hessian = sample_weight.copy()
        l1, l2 = 0.0, 0.0
    tree = build_hist_tree(
        x,
        gradient,
        hessian,
        sample_weight,
        l1=l1,
        l2=l2,
        gamma=0.0,
        min_child_weight=0.0,
        max_depth=1,
        min_samples_leaf=1,
        max_bin=HIST_MAX_BIN,
    )
    margin = margin + tree_predict(x, tree)
    return margin, tree


def hist_oracle() -> dict[str, float]:
    x, regression, labels = hist_fixture()
    regression_margin, regression_tree = fit_hist_boosting(
        x, regression, HIST_WEIGHTS, logistic=False
    )
    logistic_margin, logistic_tree = fit_hist_boosting(
        x, labels, HIST_WEIGHTS, logistic=True
    )
    regression_mse = float(
        np.sum(HIST_WEIGHTS * (regression_margin - regression) ** 2)
        / np.sum(HIST_WEIGHTS)
    )
    logistic_probability = sigmoid(logistic_margin)
    logistic_logloss = float(
        -np.sum(
            HIST_WEIGHTS
            * (
                labels * np.log(np.maximum(logistic_probability, 1.0e-15))
                + (1.0 - labels) * np.log(np.maximum(1.0 - logistic_probability, 1.0e-15))
            )
        )
        / np.sum(HIST_WEIGHTS)
    )
    multiclass_labels = np.array((-1, -1, 4, 4, 9, 9), dtype=np.int64)
    positive_probabilities = []
    for class_label in (-1, 4, 9):
        margin, _tree = fit_hist_boosting(
            x,
            (multiclass_labels == class_label).astype(np.float64),
            HIST_WEIGHTS,
            logistic=True,
        )
        positive_probabilities.append(sigmoid(margin))
    positive = np.column_stack(positive_probabilities)
    probabilities = positive / np.sum(positive, axis=1, keepdims=True)
    predicted = np.array((-1, 4, 9), dtype=np.int64)[np.argmax(probabilities, axis=1)]
    return {
        "regression_mse": regression_mse,
        "regression_sum": float(np.sum(regression_margin)),
        "regression_gain": regression_tree.gain,
        "logistic_logloss": logistic_logloss,
        "logistic_accuracy": float(np.mean((logistic_probability >= 0.5) == (labels >= 0.5))),
        "logistic_probability_sum": float(np.sum(logistic_probability)),
        "logistic_gain": logistic_tree.gain,
        "multiclass_accuracy": float(np.mean(predicted == multiclass_labels)),
        "multiclass_probability_sum": float(np.sum(probabilities)),
        "multiclass_class0_sum": float(np.sum(probabilities[:, 0])),
    }


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


def staged_boosting(
    x: np.ndarray, target: np.ndarray, *, logistic: bool
) -> tuple[np.ndarray, list[TreeNode]]:
    """Return independent cumulative raw/probability stages."""
    if logistic:
        margin = np.full(x.shape[0], logit(float(np.mean(target))))
    else:
        margin = np.full(x.shape[0], float(np.mean(target)))
    stages = np.empty((x.shape[0], N_ESTIMATORS), dtype=np.float64)
    trees: list[TreeNode] = []
    for stage in range(N_ESTIMATORS):
        if logistic:
            probability = sigmoid(margin)
            gradient = probability - target
            hessian = np.maximum(probability * (1.0 - probability), 1.0e-12)
        else:
            gradient = margin - target
            hessian = np.ones(x.shape[0], dtype=np.float64)
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
        margin = margin + LEARNING_RATE * tree_predict(x, tree)
        stages[:, stage] = sigmoid(margin) if logistic else margin
    return stages, trees


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


def multiclass_staged_oracle(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized OVR probability stages and raw margin stages."""
    labels = multiclass_fixture(x)
    positive_stages: list[np.ndarray] = []
    margin_stages: list[np.ndarray] = []
    for class_label in MULTICLASS_CLASSES:
        stages, _trees = staged_boosting(
            x, (labels == class_label).astype(np.float64), logistic=True
        )
        positive_stages.append(stages)
        margin_stages.append(np.log(np.clip(stages, 1.0e-15, 1.0 - 1.0e-15) /
                                   np.clip(1.0 - stages, 1.0e-15, 1.0)))
    positive = np.stack(positive_stages, axis=1)
    margins = np.stack(margin_stages, axis=1)
    probabilities = positive / np.sum(positive, axis=1, keepdims=True)
    return probabilities, margins


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
        "xgb_regression_staged",
        "xgb_logistic_fit",
        "xgb_logistic_predict",
        "xgb_logistic_staged",
        "xgb_multiclass_fit",
        "xgb_multiclass_predict",
        "xgb_multiclass_staged",
        "xgb_multiclass_staged_margin",
        "xgb_missing_fit",
        "xgb_missing_predict",
        "xgb_hist_regression_fit",
        "xgb_hist_regression_predict",
        "xgb_hist_logistic_fit",
        "xgb_hist_logistic_predict",
        "xgb_hist_multiclass_fit",
        "xgb_hist_multiclass_predict",
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
    regression_stages, _ = staged_boosting(x, regression, logistic=False)
    logistic_stages, _ = staged_boosting(x, labels, logistic=True)
    multiclass_accuracy, multiclass_probability_sum = multiclass_oracle(x)
    multiclass_stages, multiclass_margin_stages = multiclass_staged_oracle(x)
    missing_prediction, missing_gain = missing_oracle()
    hist_values = hist_oracle()
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
    regression_staged_error = max(
        abs(float(parsed["xgb_regression_staged"][4]) - np.sum(regression_stages[:, 0])),
        abs(float(parsed["xgb_regression_staged"][5]) - np.sum(regression_stages[:, -1])),
        abs(float(parsed["xgb_regression_staged"][6]) - 1.0),
    )
    logistic_staged_error = max(
        abs(float(parsed["xgb_logistic_staged"][4]) - np.sum(logistic_stages[:, 0])),
        abs(float(parsed["xgb_logistic_staged"][5]) - np.sum(logistic_stages[:, -1])),
        abs(float(parsed["xgb_logistic_staged"][6]) - 1.0),
    )
    multiclass_staged_error = max(
        abs(float(parsed["xgb_multiclass_staged"][4]) - np.sum(multiclass_stages[:, :, 0])),
        abs(float(parsed["xgb_multiclass_staged"][5]) - np.sum(multiclass_stages[:, :, -1])),
        abs(float(parsed["xgb_multiclass_staged"][6]) - 1.0),
    )
    multiclass_margin_error = max(
        abs(float(parsed["xgb_multiclass_staged_margin"][4]) -
            np.max(np.abs(multiclass_margin_stages[:, :, -1]))),
        abs(float(parsed["xgb_multiclass_staged_margin"][5]) -
            np.max(np.abs(multiclass_margin_stages[:, :, -1]))),
    )
    if max(regression_staged_error, logistic_staged_error, multiclass_staged_error,
           multiclass_margin_error) > 2.0e-10:
        raise RuntimeError(
            "FortML XGBoost staged/importance oracle mismatch: "
            f"regression={regression_staged_error:.3e}; "
            f"logistic={logistic_staged_error:.3e}; "
            f"multiclass={multiclass_staged_error:.3e}; "
            f"margins={multiclass_margin_error:.3e}"
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
    hist_regression_error = max(
        abs(float(parsed["xgb_hist_regression_fit"][4]) - hist_values["regression_mse"]),
        abs(float(parsed["xgb_hist_regression_fit"][5]) - hist_values["regression_sum"]),
        abs(float(parsed["xgb_hist_regression_fit"][6]) - hist_values["regression_gain"]),
        abs(float(parsed["xgb_hist_regression_predict"][4]) - hist_values["regression_mse"]),
        abs(float(parsed["xgb_hist_regression_predict"][5]) - hist_values["regression_sum"]),
        abs(float(parsed["xgb_hist_regression_predict"][6]) - hist_values["regression_gain"]),
    )
    hist_logistic_error = max(
        abs(float(parsed["xgb_hist_logistic_fit"][4]) - hist_values["logistic_logloss"]),
        abs(float(parsed["xgb_hist_logistic_fit"][5]) - hist_values["logistic_accuracy"]),
        abs(float(parsed["xgb_hist_logistic_fit"][6]) - hist_values["logistic_gain"]),
        abs(float(parsed["xgb_hist_logistic_predict"][4]) - hist_values["logistic_logloss"]),
        abs(float(parsed["xgb_hist_logistic_predict"][5]) - hist_values["logistic_probability_sum"]),
        abs(float(parsed["xgb_hist_logistic_predict"][6]) - hist_values["logistic_gain"]),
    )
    hist_multiclass_error = max(
        abs(float(parsed["xgb_hist_multiclass_fit"][4]) - hist_values["multiclass_accuracy"]),
        abs(float(parsed["xgb_hist_multiclass_fit"][5]) - hist_values["multiclass_probability_sum"]),
        abs(float(parsed["xgb_hist_multiclass_fit"][6]) - hist_values["multiclass_class0_sum"]),
        abs(float(parsed["xgb_hist_multiclass_predict"][4]) - hist_values["multiclass_accuracy"]),
        abs(float(parsed["xgb_hist_multiclass_predict"][5]) - hist_values["multiclass_probability_sum"]),
        abs(float(parsed["xgb_hist_multiclass_predict"][6]) - hist_values["multiclass_class0_sum"]),
    )
    if max(hist_regression_error, hist_logistic_error, hist_multiclass_error) > 2.0e-11:
        raise RuntimeError(
            "FortML weighted histogram oracle mismatch: "
            f"regression={hist_regression_error:.3e}; "
            f"logistic={hist_logistic_error:.3e}; "
            f"multiclass={hist_multiclass_error:.3e}"
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
            workload="xgboost_regression_staged",
            phase="diagnostics",
            backend="fortml",
            status="pass",
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_estimators=N_ESTIMATORS,
            seconds_per_operation=float(parsed["xgb_regression_staged"][3]),
            metric="final_stage_sum",
            value=float(np.sum(regression_stages[:, -1])),
            max_abs_error=regression_staged_error,
            oracle="independent NumPy cumulative exact-tree stage oracle",
            notes="regression margins; normalized gain importance sum checked",
        ),
        row(
            metadata_values,
            workload="xgboost_logistic_staged",
            phase="diagnostics",
            backend="fortml",
            status="pass",
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_estimators=N_ESTIMATORS,
            seconds_per_operation=float(parsed["xgb_logistic_staged"][3]),
            metric="final_stage_positive_probability_sum",
            value=float(np.sum(logistic_stages[:, -1])),
            max_abs_error=logistic_staged_error,
            oracle="independent NumPy cumulative logistic stage oracle",
            notes="positive-class staged probabilities; normalized gain importance sum checked",
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
            workload="xgboost_multiclass_staged",
            phase="diagnostics",
            backend="fortml",
            status="pass",
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_estimators=N_ESTIMATORS,
            seconds_per_operation=float(parsed["xgb_multiclass_staged"][3]),
            metric="final_stage_probability_sum",
            value=float(np.sum(multiclass_stages[:, :, -1])),
            max_abs_error=multiclass_staged_error,
            oracle="independent NumPy normalized one-vs-rest stage oracle",
            notes="probability simplex and normalized gain importance sum checked",
        ),
        row(
            metadata_values,
            workload="xgboost_multiclass_staged_margin",
            phase="diagnostics",
            backend="fortml",
            status="pass",
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_estimators=N_ESTIMATORS,
            seconds_per_operation=float(parsed["xgb_multiclass_staged_margin"][3]),
            metric="max_abs_final_margin",
            value=float(np.max(np.abs(multiclass_margin_stages[:, :, -1]))),
            max_abs_error=multiclass_margin_error,
            oracle="independent NumPy cumulative one-vs-rest margin oracle",
            notes="staged raw margins agree with final decision_function",
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
        row(
            metadata_values,
            workload="xgboost_hist_weighted_regression",
            phase="fit",
            backend="fortml",
            status="pass",
            n_samples=HIST_SAMPLES,
            n_features=HIST_FEATURES,
            n_estimators=1,
            seconds_per_operation=float(parsed["xgb_hist_regression_fit"][3]),
            metric="weighted_mse",
            value=hist_values["regression_mse"],
            max_abs_error=hist_regression_error,
            oracle="independent NumPy weighted-quantile histogram Newton tree",
            notes="max_bin=2; weights=[1,1,1,1,5,5]; one weighted-median cut",
        ),
        row(
            metadata_values,
            workload="xgboost_hist_weighted_regression",
            phase="predict",
            backend="fortml",
            status="pass",
            n_samples=HIST_SAMPLES,
            n_features=HIST_FEATURES,
            n_estimators=1,
            seconds_per_operation=float(parsed["xgb_hist_regression_predict"][3]),
            metric="prediction_sum",
            value=hist_values["regression_sum"],
            max_abs_error=hist_regression_error,
            oracle="independent NumPy weighted-quantile histogram Newton tree",
            notes="stored histogram threshold reused by prediction",
        ),
        row(
            metadata_values,
            workload="xgboost_hist_weighted_logistic",
            phase="fit",
            backend="fortml",
            status="pass",
            n_samples=HIST_SAMPLES,
            n_features=HIST_FEATURES,
            n_estimators=1,
            seconds_per_operation=float(parsed["xgb_hist_logistic_fit"][3]),
            metric="weighted_logloss",
            value=hist_values["logistic_logloss"],
            max_abs_error=hist_logistic_error,
            oracle="independent NumPy weighted-quantile logistic Newton tree",
            notes="max_bin=2; weighted base logit and Hessian reductions checked",
        ),
        row(
            metadata_values,
            workload="xgboost_hist_weighted_logistic",
            phase="predict",
            backend="fortml",
            status="pass",
            n_samples=HIST_SAMPLES,
            n_features=HIST_FEATURES,
            n_estimators=1,
            seconds_per_operation=float(parsed["xgb_hist_logistic_predict"][3]),
            metric="probability_sum",
            value=hist_values["logistic_probability_sum"],
            max_abs_error=hist_logistic_error,
            oracle="independent NumPy weighted-quantile logistic Newton tree",
            notes="binary probability path checked after histogram fit",
        ),
        row(
            metadata_values,
            workload="xgboost_hist_weighted_multiclass",
            phase="fit",
            backend="fortml",
            status="pass",
            n_samples=HIST_SAMPLES,
            n_features=HIST_FEATURES,
            n_estimators=1,
            seconds_per_operation=float(parsed["xgb_hist_multiclass_fit"][3]),
            metric="accuracy",
            value=hist_values["multiclass_accuracy"],
            max_abs_error=hist_multiclass_error,
            oracle="independent NumPy weighted one-vs-rest histogram Newton trees",
            notes="labels=[-1,-1,4,4,9,9]; max_bin=2; weighted OVR normalization",
        ),
        row(
            metadata_values,
            workload="xgboost_hist_weighted_multiclass",
            phase="predict",
            backend="fortml",
            status="pass",
            n_samples=HIST_SAMPLES,
            n_features=HIST_FEATURES,
            n_estimators=1,
            seconds_per_operation=float(parsed["xgb_hist_multiclass_predict"][3]),
            metric="class0_probability_sum",
            value=hist_values["multiclass_class0_sum"],
            max_abs_error=hist_multiclass_error,
            oracle="independent NumPy weighted one-vs-rest histogram Newton trees",
            notes="probability simplex sum and class-0 checksum checked",
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
            workload="xgboost_histogram_gpu",
            phase="capability_check",
            backend="fortml",
            status="unavailable",
            oracle="declared capability boundary",
            notes="CPU weighted histogram growth is measured; native CUDA histogram growth remains open",
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
    metadata_values = metadata(root, fortml, args.output.resolve())
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
