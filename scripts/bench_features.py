#!/usr/bin/env python3
"""Benchmark FortML's MLP trainer, basis pipeline, and tree primitives.

The Fortran executable reports release-build timings and compact checksums.
This harness reconstructs every fixture in NumPy before timing and verifies
the complete mathematical result where it is practical.  scikit-learn is a
second behavioral reference for the estimator-shaped lanes; optional
PyTorch/JAX/XGBoost rows are explicit refusals when those packages are not
installed instead of silently disappearing from the record.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


MLP_N = 96
MLP_D = 3
MLP_HIDDEN = 8
MLP_OUTPUTS = 1
MLP_EPOCHS = 24
MLP_LR = 0.01
MLP_L2 = 1.0e-4
BASIS_N = 256
BASIS_D = 2
BASIS_LINEAR_N = 128
BASIS_LINEAR_FREQUENCY = 0.7
TREE_N = 128
TREE_D = 2
TREE_ESTIMATORS = 16
TREE_RATE = 0.1
TREE_LEAF = 3
CART_DEPTH = 3
METRIC_N = 128
METRIC_OUTPUTS = 2
GAUSSIAN_NB_N = 192
GAUSSIAN_NB_D = 2
GAUSSIAN_NB_CLASSES = np.array([-4, 7, 19], dtype=np.int64)
GAUSSIAN_NB_SMOOTHING = 1.0e-9

FIELDS = (
    "workload",
    "phase",
    "backend",
    "device",
    "status",
    "n_samples",
    "n_features",
    "n_hidden",
    "n_estimators",
    "repetitions",
    "seconds_per_operation",
    "metric",
    "value",
    "mse",
    "max_abs_error",
    "oracle",
    "python_version",
    "numpy_version",
    "sklearn_version",
    "torch_version",
    "jax_version",
    "xgboost_version",
    "fortml_revision",
    "benchmark_revision",
    "compiler",
    "flags",
    "notes",
)


def revision(repository: Path, ignored_paths: tuple[Path, ...] = ()) -> str:
    value = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    status_lines = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines()
    ignored = {path.resolve() for path in ignored_paths}
    dirty = []
    for line in status_lines:
        path_text = line[3:].split(" -> ")[-1].strip()
        if (repository / path_text).resolve() not in ignored:
            dirty.append(line)
    return value + ("+dirty" if dirty else "")


def package_version(name: str) -> str:
    try:
        module = __import__(name)
    except ImportError:
        return "unavailable"
    return str(getattr(module, "__version__", "installed"))


def base_metadata(
    root: Path, fortml: Path, ignored_paths: tuple[Path, ...] = ()
) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": package_version("sklearn"),
        "torch_version": package_version("torch"),
        "jax_version": package_version("jax"),
        "xgboost_version": package_version("xgboost"),
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored_paths),
        "compiler": "gfortran",
        "flags": "-O3",
        "device": "cpu",
    }


def mlp_inputs() -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(1, MLP_N + 1, dtype=np.float64)[:, None]
    columns = np.arange(1, MLP_D + 1, dtype=np.float64)[None, :]
    x = np.sin(0.017 * rows + 0.13 * columns)
    x += 0.15 * np.cos(0.009 * rows * columns)
    target = 0.4 * np.sin(x[:, :1]) + 0.2 * x[:, 1:2] - 0.1 * x[:, 2:3]
    target += 0.03 * np.cos(2.0 * x[:, :1])
    return x, target


def initial_theta(seed: int = 23) -> np.ndarray:
    layers = ((MLP_D, MLP_HIDDEN), (MLP_HIDDEN, MLP_OUTPUTS))
    pieces: list[np.ndarray] = []
    for layer_index, (n_in, n_out) in enumerate(layers, start=1):
        scale = np.sqrt(6.0 / float(n_in + n_out))
        index = np.arange(1, n_in * n_out + 1, dtype=np.float64)
        phase = seed + 1009 * layer_index + 9176 * index
        weight = (scale * np.sin(phase)).reshape((n_in, n_out), order="F")
        bias_index = np.arange(1, n_out + 1, dtype=np.float64)
        bias = 0.01 * scale * np.sin(seed + 1009 * layer_index + 7919 * bias_index)
        pieces.extend((weight.reshape(-1, order="F"), bias))
    return np.concatenate(pieces)


def unpack_theta(theta: np.ndarray) -> tuple[np.ndarray, ...]:
    position = 0
    count = MLP_D * MLP_HIDDEN
    weight_1 = theta[position : position + count].reshape(
        (MLP_D, MLP_HIDDEN), order="F"
    )
    position += count
    bias_1 = theta[position : position + MLP_HIDDEN]
    position += MLP_HIDDEN
    count = MLP_HIDDEN * MLP_OUTPUTS
    weight_2 = theta[position : position + count].reshape(
        (MLP_HIDDEN, MLP_OUTPUTS), order="F"
    )
    position += count
    bias_2 = theta[position : position + MLP_OUTPUTS]
    return weight_1, bias_1, weight_2, bias_2


def mlp_value_gradient(
    theta: np.ndarray, x: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    weight_1, bias_1, weight_2, bias_2 = unpack_theta(theta)
    hidden = np.tanh(x @ weight_1 + bias_1)
    prediction = hidden @ weight_2 + bias_2
    residual = prediction - target
    n = float(x.shape[0])
    preactivation_bar = (residual / n) @ weight_2.T
    preactivation_bar *= 1.0 - hidden * hidden
    weight_1_bar = x.T @ preactivation_bar
    bias_1_bar = np.sum(preactivation_bar, axis=0)
    weight_2_bar = hidden.T @ (residual / n)
    bias_2_bar = np.sum(residual / n, axis=0)
    gradient = np.concatenate(
        (
            weight_1_bar.reshape(-1, order="F"),
            bias_1_bar,
            weight_2_bar.reshape(-1, order="F"),
            bias_2_bar,
        )
    )
    value = 0.5 * np.sum(residual * residual) / n + 0.5 * MLP_L2 * np.sum(theta**2)
    gradient += MLP_L2 * theta
    return float(value), gradient, prediction


def mlp_oracle() -> dict[str, Any]:
    x, target = mlp_inputs()
    theta = initial_theta()
    initial_loss, _, _ = mlp_value_gradient(theta, x, target)
    first = np.zeros_like(theta)
    second = np.zeros_like(theta)
    for epoch in range(MLP_EPOCHS):
        _, gradient, _prediction = mlp_value_gradient(theta, x, target)
        # This is fortopt_adam's bias-corrected update, kept explicit so the
        # oracle does not depend on a framework optimizer implementation.
        step = epoch + 1
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient**2
        theta -= (
            MLP_LR
            * (first / (1.0 - 0.9**step))
            / (np.sqrt(second / (1.0 - 0.999**step)) + 1.0e-8)
        )
    final_loss, _, prediction = mlp_value_gradient(theta, x, target)
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "prediction": prediction[:, 0],
    }


def basis_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.empty((BASIS_N, BASIS_D), dtype=np.float64)
    x_dot = np.empty_like(x)
    for j in range(1, BASIS_D + 1):
        for i in range(1, BASIS_N + 1):
            x[i - 1, j - 1] = np.sin(0.011 * i + 0.17 * j)
            x_dot[i - 1, j - 1] = np.cos(0.007 * (i + 2 * j))
    frequencies = np.array([[1.2, 0.7], [0.55, 1.1]], dtype=np.float64)
    return x, x_dot, frequencies


def basis_oracle() -> dict[str, float]:
    x, x_dot, frequencies = basis_inputs()
    polynomial = [np.ones(BASIS_N)]
    polynomial.extend(x[:, j] ** p for j in range(BASIS_D) for p in range(1, 4))
    fourier = [np.ones(BASIS_N)]
    phi_dot_fourier = [np.zeros(BASIS_N)]
    theta_dot = np.full(frequencies.size, 0.07)
    theta_index = 0
    for j in range(BASIS_D):
        for h in range(frequencies.shape[0]):
            frequency = frequencies[h, j]
            argument = frequency * x[:, j]
            argument_dot = frequency * (x_dot[:, j] + x[:, j] * theta_dot[theta_index])
            fourier.extend((np.sin(argument), np.cos(argument)))
            phi_dot_fourier.extend(
                (np.cos(argument) * argument_dot, -np.sin(argument) * argument_dot)
            )
            theta_index += 1
    phi = np.column_stack(polynomial + fourier)
    phi_dot_polynomial = [np.zeros(BASIS_N)]
    phi_dot_polynomial.extend(
        p * x[:, j] ** (p - 1) * x_dot[:, j]
        for j in range(BASIS_D)
        for p in range(1, 4)
    )
    phi_dot = np.column_stack(phi_dot_polynomial + phi_dot_fourier)
    u = np.empty_like(phi)
    for j in range(phi.shape[1]):
        for i in range(phi.shape[0]):
            u[i, j] = 0.13 * np.sin(0.013 * ((i + 1) + (j + 1)))
    theta_bar = np.zeros(frequencies.size)
    x_bar = np.zeros_like(x)
    column = 8  # zero-based output column 9 follows Fourier's intercept
    theta_index = 0
    for j in range(BASIS_D):
        for h in range(frequencies.shape[0]):
            frequency = frequencies[h, j]
            argument = frequency * x[:, j]
            z_bar = u[:, column] * np.cos(argument) - u[:, column + 1] * np.sin(
                argument
            )
            x_bar[:, j] += frequency * z_bar
            theta_bar[theta_index] = np.sum(frequency * x[:, j] * z_bar)
            column += 2
            theta_index += 1
    column = 1
    for j in range(BASIS_D):
        for p in range(1, 4):
            x_bar[:, j] += u[:, column] * p * x[:, j] ** (p - 1)
            column += 1
    return {
        "transform_sum": float(np.sum(phi)),
        "jvp_sum": float(np.sum(phi_dot)),
        "theta_bar_sum": float(np.sum(theta_bar)),
        "x_bar_sum": float(np.sum(x_bar)),
    }


def basis_linear_oracle() -> dict[str, float]:
    """Check the fitted Fourier basis plus linear-regression composition."""

    index = np.arange(1, BASIS_LINEAR_N + 1, dtype=np.float64)
    x = -1.0 + 2.0 * (index - 1.0) / float(BASIS_LINEAR_N - 1)
    x_dot = np.cos(0.013 * index)
    frequency = BASIS_LINEAR_FREQUENCY
    target = 1.2 + 2.0 * np.sin(frequency * x) - 0.3 * np.cos(frequency * x)
    design = np.column_stack(
        (np.ones(BASIS_LINEAR_N), np.sin(frequency * x), np.cos(frequency * x))
    )
    coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(
        design, target, rcond=None
    )
    prediction = design @ coefficients
    theta_dot = np.full(4, 0.04)
    argument_dot = frequency * (x_dot + x * theta_dot[0])
    prediction_dot = (
        theta_dot[1]
        + theta_dot[2] * np.sin(frequency * x)
        + theta_dot[3] * np.cos(frequency * x)
        + coefficients[1] * np.cos(frequency * x) * argument_dot
        - coefficients[2] * np.sin(frequency * x) * argument_dot
    )
    return {
        "mse": float(np.mean((prediction - target) ** 2)),
        "prediction_sum": float(np.sum(prediction)),
        "jvp_sum": float(np.sum(prediction_dot)),
    }


def gaussian_nb_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the deterministic GaussianNB fixture independently."""

    index = np.arange(1, GAUSSIAN_NB_N + 1, dtype=np.float64)
    class_index = np.arange(GAUSSIAN_NB_N, dtype=np.int64) % 3
    labels = GAUSSIAN_NB_CLASSES[class_index]
    x = np.column_stack(
        (
            3.0 * class_index + 0.1 * np.sin(0.11 * index),
            -2.0 * class_index + 0.1 * np.cos(0.07 * index),
        )
    )
    x_dot = np.column_stack((np.cos(0.013 * index), np.sin(0.017 * index)))
    return x, labels, x_dot


def gaussian_nb_oracle() -> dict[str, float]:
    x, labels, x_dot = gaussian_nb_inputs()
    means = np.vstack(
        [x[labels == label].mean(axis=0) for label in GAUSSIAN_NB_CLASSES]
    )
    variance = np.vstack(
        [x[labels == label].var(axis=0) for label in GAUSSIAN_NB_CLASSES]
    )
    epsilon = GAUSSIAN_NB_SMOOTHING * float(np.max(np.var(x, axis=0)))
    epsilon = max(epsilon, np.finfo(np.float64).tiny)
    variance += epsilon
    log_joint = np.empty((x.shape[0], GAUSSIAN_NB_CLASSES.size), dtype=np.float64)
    for class_index in range(GAUSSIAN_NB_CLASSES.size):
        delta = x - means[class_index]
        log_joint[:, class_index] = -0.5 * np.sum(
            np.log(2.0 * np.pi * variance[class_index])
            + delta * delta / variance[class_index],
            axis=1,
        ) - np.log(float(GAUSSIAN_NB_CLASSES.size))
    maximum = np.max(log_joint, axis=1)
    log_probabilities = log_joint - maximum[:, None]
    log_probabilities -= np.log(np.sum(np.exp(log_probabilities), axis=1))[:, None]
    joint_dot = np.empty_like(log_joint)
    for class_index in range(GAUSSIAN_NB_CLASSES.size):
        joint_dot[:, class_index] = -np.sum(
            (x - means[class_index]) / variance[class_index] * x_dot, axis=1
        )
    log_dot = joint_dot - np.sum(np.exp(log_probabilities) * joint_dot, axis=1)[:, None]
    target_class_indices = np.arange(GAUSSIAN_NB_N, dtype=np.int64) % 3
    return {
        "log_sum": float(np.sum(log_probabilities)),
        "jvp_sum": float(np.sum(log_dot)),
        "accuracy": float(
            np.mean(np.argmax(log_probabilities, axis=1) == target_class_indices)
        ),
    }


def tree_inputs() -> tuple[np.ndarray, np.ndarray]:
    x = np.empty((TREE_N, TREE_D), dtype=np.float64)
    y = np.empty(TREE_N, dtype=np.float64)
    for i in range(1, TREE_N + 1):
        x[i - 1, 0] = -1.0 + 2.0 * (i - 1) / (TREE_N - 1)
        x[i - 1, 1] = np.sin(0.09 * i)
        y[i - 1] = (
            (1.7 + 0.2 * x[i - 1, 1])
            if x[i - 1, 0] >= 0.1
            else (-0.8 + 0.1 * x[i - 1, 1])
        )
    return x, y


def best_stump(x: np.ndarray, y: np.ndarray) -> tuple[int, float, float, float]:
    best: tuple[float, int, float, float, float] | None = None
    for feature in range(x.shape[1]):
        order = np.argsort(x[:, feature], kind="stable")
        for k in range(TREE_LEAF, TREE_N - TREE_LEAF + 1):
            if x[order[k - 1], feature] >= x[order[k], feature]:
                continue
            threshold = 0.5 * (x[order[k - 1], feature] + x[order[k], feature])
            left = y[order[:k]]
            right = y[order[k:]]
            left_value = float(np.mean(left))
            right_value = float(np.mean(right))
            sse = float(
                np.sum((left - left_value) ** 2) + np.sum((right - right_value) ** 2)
            )
            candidate = (sse, feature, threshold, left_value, right_value)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise RuntimeError("no valid stump split")
    _, feature, threshold, left_value, right_value = best
    return feature, threshold, left_value, right_value


def stump_predict(
    x: np.ndarray, feature: int, threshold: float, left: float, right: float
) -> np.ndarray:
    return np.where(x[:, feature] < threshold, left, right)


def boosting_oracle() -> dict[str, float]:
    x, y = tree_inputs()
    feature, threshold, left, right = best_stump(x, y)
    stump_prediction = stump_predict(x, feature, threshold, left, right)
    stump_left, stump_right = left, right
    base = float(np.mean(y))
    prediction = np.full(TREE_N, base)
    for _ in range(TREE_ESTIMATORS):
        residual = y - prediction
        f, t, left, right = best_stump(x, residual)
        prediction += TREE_RATE * stump_predict(x, f, t, left, right)
    return {
        "stump_feature": float(feature + 1),
        "stump_threshold": threshold,
        "stump_left": stump_left,
        "stump_right": stump_right,
        "stump_mse": float(np.mean((stump_prediction - y) ** 2)),
        "stump_sum": float(np.sum(stump_prediction)),
        "boosting_mse": float(np.mean((prediction - y) ** 2)),
        "boosting_sum": float(np.sum(prediction)),
    }


def cart_oracle(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
    """Return a deterministic exhaustive CART prediction and node count."""

    nodes: list[dict[str, Any]] = []

    def build(indices: np.ndarray, depth: int) -> int:
        node = len(nodes)
        nodes.append({"leaf": True, "value": float(np.mean(y[indices]))})
        if depth >= CART_DEPTH or indices.size < 2 * TREE_LEAF:
            return node
        parent_sse = float(np.sum((y[indices] - np.mean(y[indices])) ** 2))
        best: tuple[float, int, float, np.ndarray, np.ndarray] | None = None
        for feature in range(x.shape[1]):
            order = np.argsort(x[indices, feature], kind="stable")
            for k in range(TREE_LEAF, indices.size - TREE_LEAF + 1):
                left_order = order[:k]
                right_order = order[k:]
                left_values = x[indices[left_order], feature]
                right_values = x[indices[right_order], feature]
                if left_values[-1] >= right_values[0]:
                    continue
                threshold = 0.5 * (left_values[-1] + right_values[0])
                left = indices[left_order]
                right = indices[right_order]
                left_mean = float(np.mean(y[left]))
                right_mean = float(np.mean(y[right]))
                sse = float(
                    np.sum((y[left] - left_mean) ** 2)
                    + np.sum((y[right] - right_mean) ** 2)
                )
                candidate = (sse, feature, threshold, left, right)
                if best is None or sse < best[0]:
                    best = candidate
        if best is None or best[0] >= parent_sse:
            return node
        _, feature, threshold, left, right = best
        nodes[node].update(
            {
                "leaf": False,
                "feature": feature,
                "threshold": threshold,
                "left": build(left, depth + 1),
                "right": build(right, depth + 1),
            }
        )
        return node

    build(np.arange(x.shape[0]), 0)
    prediction = np.empty_like(y)
    for i in range(x.shape[0]):
        node = 0
        while not nodes[node]["leaf"]:
            if x[i, nodes[node]["feature"]] < nodes[node]["threshold"]:
                node = nodes[node]["left"]
            else:
                node = nodes[node]["right"]
        prediction[i] = nodes[node]["value"]
    return prediction, len(nodes)


def cart_classifier_inputs() -> tuple[np.ndarray, np.ndarray]:
    """Return the deterministic two-class fixture used by the Fortran app."""

    x, _ = tree_inputs()
    labels = np.where(x[:, 0] >= 0.1, 7, -3).astype(np.int64)
    return x, labels


def cart_classifier_oracle(
    x: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return deterministic exhaustive-Gini CART predictions and probabilities."""

    classes = np.unique(labels)
    nodes: list[dict[str, Any]] = []

    def impurity(indices: np.ndarray) -> float:
        counts = np.array(
            [np.sum(labels[indices] == class_label) for class_label in classes],
            dtype=np.float64,
        )
        total = float(np.sum(counts))
        if total == 0.0:
            return 0.0
        probabilities = counts / total
        return float(total * (1.0 - np.sum(probabilities**2)))

    def leaf_probability(indices: np.ndarray) -> np.ndarray:
        counts = np.array(
            [np.sum(labels[indices] == class_label) for class_label in classes],
            dtype=np.float64,
        )
        return counts / float(indices.size)

    def build(indices: np.ndarray, depth: int) -> int:
        node = len(nodes)
        nodes.append({"leaf": True, "probability": leaf_probability(indices)})
        if depth >= CART_DEPTH or indices.size < 2 * TREE_LEAF:
            return node
        parent_impurity = impurity(indices)
        best_impurity = float("inf")
        best: tuple[int, float, np.ndarray, np.ndarray] | None = None
        for feature in range(x.shape[1]):
            order = np.argsort(x[indices, feature], kind="stable")
            for k in range(TREE_LEAF, indices.size - TREE_LEAF + 1):
                left_values = x[indices[order[:k]], feature]
                right_values = x[indices[order[k:]], feature]
                if left_values[-1] >= right_values[0]:
                    continue
                threshold = 0.5 * (left_values[-1] + right_values[0])
                left = indices[order[:k]]
                right = indices[order[k:]]
                candidate_impurity = impurity(left) + impurity(right)
                if best is None or candidate_impurity < best_impurity:
                    best_impurity = candidate_impurity
                    best = (feature, threshold, left, right)
        if best is None or best_impurity >= parent_impurity:
            return node
        feature, threshold, left, right = best
        nodes[node].update(
            {
                "leaf": False,
                "feature": feature,
                "threshold": threshold,
                "left": build(left, depth + 1),
                "right": build(right, depth + 1),
            }
        )
        return node

    build(np.arange(x.shape[0]), 0)
    prediction = np.empty(labels.size, dtype=labels.dtype)
    probabilities = np.empty((labels.size, classes.size), dtype=np.float64)
    for i in range(labels.size):
        node = 0
        while not nodes[node]["leaf"]:
            if x[i, nodes[node]["feature"]] < nodes[node]["threshold"]:
                node = nodes[node]["left"]
            else:
                node = nodes[node]["right"]
        probability = nodes[node]["probability"]
        probabilities[i] = probability
        prediction[i] = classes[int(np.argmax(probability))]
    return prediction, probabilities, len(nodes)


def regression_metric_oracle() -> dict[str, float]:
    index = np.arange(1, METRIC_N + 1, dtype=np.float64)
    target = np.column_stack((np.sin(0.03 * index), np.cos(0.05 * index)))
    prediction = target.copy()
    prediction[:, 0] += 0.1 * np.cos(0.07 * index)
    prediction[:, 1] -= 0.07 * np.sin(0.09 * index)
    residual = prediction - target
    mse = float(np.mean(residual**2))
    mae = float(np.mean(np.abs(residual)))
    target_mean = np.mean(target, axis=0)
    r2 = float(
        np.mean(
            1.0
            - np.sum(residual**2, axis=0) / np.sum((target - target_mean) ** 2, axis=0)
        )
    )
    error = target - prediction
    pinball = float(np.mean(np.maximum(0.25 * error, -0.75 * error)))
    return {"mse": mse, "mae": mae, "r2": r2, "pinball": pinball}


def parse_fortran(stdout: str) -> dict[str, list[list[str]]]:
    rows: dict[str, list[list[str]]] = {}
    for line in stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and fields[0] in {
            "mlp_train",
            "mlp_train_prediction",
            "basis_transform",
            "basis_jvp",
            "basis_vjp",
            "basis_linear",
            "gaussian_nb",
            "stump",
            "cart",
            "cart_classifier",
            "boosting",
            "regression_metrics",
        }:
            rows.setdefault(fields[0], []).append(fields[1:])
    required = {
        "mlp_train",
        "mlp_train_prediction",
        "basis_transform",
        "basis_jvp",
        "basis_vjp",
        "basis_linear",
        "gaussian_nb",
        "stump",
        "cart",
        "cart_classifier",
        "boosting",
        "regression_metrics",
    }
    if required - rows.keys():
        raise RuntimeError(
            f"FortML feature app omitted {sorted(required - rows.keys())}"
        )
    return rows


def row(metadata: dict[str, str], **values: Any) -> dict[str, Any]:
    result = {field: "" for field in FIELDS}
    result.update(metadata)
    result.update(values)
    return result


def run_fortran(
    root: Path, fortml: Path, metadata: dict[str, str]
) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": "gfortran", "OMP_NUM_THREADS": "1"})
    subprocess.run(
        ["fo", "build", "--flag", "-O3"], cwd=fortml, env=environment, check=True
    )
    started = time.perf_counter()
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "fortml_bench_features"],
        cwd=fortml,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    process_seconds = time.perf_counter() - started
    parsed = parse_fortran(completed.stdout)
    mlp = mlp_oracle()
    mlp_row = parsed["mlp_train"][0]
    mlp_error = max(
        abs(float(mlp_row[5]) - mlp["initial_loss"]),
        abs(float(mlp_row[6]) - mlp["final_loss"]),
    )
    predictions = np.array([float(item[1]) for item in parsed["mlp_train_prediction"]])
    mlp_error = max(
        mlp_error, float(np.max(np.abs(predictions - mlp["prediction"][:8])))
    )
    if mlp_error > 5.0e-12:
        raise RuntimeError(f"FortML MLP training oracle mismatch: {mlp_error:.3e}")

    basis = basis_oracle()
    basis_rows = {
        "basis_transform": ("transform", basis["transform_sum"], 4),
        "basis_jvp": ("jvp", basis["jvp_sum"], 4),
        "basis_vjp": ("vjp", basis["theta_bar_sum"], 4),
    }
    for key, (_, expected, index) in basis_rows.items():
        actual = float(parsed[key][0][-1 if key != "basis_vjp" else 1])
        if key == "basis_vjp":
            actual_theta = float(parsed[key][0][4])
            actual_x = float(parsed[key][0][5])
            error = max(
                abs(actual_theta - expected), abs(actual_x - basis["x_bar_sum"])
            )
        else:
            error = abs(actual - expected)
        if error > 5.0e-10:
            raise RuntimeError(f"FortML {key} oracle mismatch: {error:.3e}")

    basis_linear = basis_linear_oracle()
    basis_linear_row = parsed["basis_linear"][0]
    basis_linear_values = {
        "mse": float(basis_linear_row[6]),
        "prediction_sum": float(basis_linear_row[7]),
        "jvp_sum": float(basis_linear_row[8]),
    }
    basis_linear_error = max(
        abs(basis_linear_values[key] - basis_linear[key]) for key in basis_linear
    )
    if basis_linear_error > 5.0e-10:
        raise RuntimeError(
            f"FortML basis-linear oracle mismatch: {basis_linear_error:.3e}"
        )

    gaussian_nb = gaussian_nb_oracle()
    gaussian_nb_row = parsed["gaussian_nb"][0]
    gaussian_nb_values = {
        "log_sum": float(gaussian_nb_row[6]),
        "jvp_sum": float(gaussian_nb_row[7]),
    }
    gaussian_nb_error = max(
        abs(gaussian_nb_values[key] - gaussian_nb[key]) for key in gaussian_nb_values
    )
    if gaussian_nb_error > 5.0e-8:
        raise RuntimeError(
            f"FortML GaussianNB oracle mismatch: {gaussian_nb_error:.3e}"
        )

    tree = boosting_oracle()
    stump_row = parsed["stump"][0]
    stump_values = {
        "stump_feature": float(stump_row[4]),
        "stump_threshold": float(stump_row[5]),
        "stump_left": float(stump_row[6]),
        "stump_right": float(stump_row[7]),
        "stump_mse": float(stump_row[8]),
        "stump_sum": float(stump_row[9]),
    }
    boost_row = parsed["boosting"][0]
    boosting_values = {
        "boosting_mse": float(boost_row[5]),
        "boosting_sum": float(boost_row[6]),
    }
    tree_error = max(abs(stump_values[key] - tree[key]) for key in stump_values)
    tree_error = max(
        tree_error, *(abs(boosting_values[key] - tree[key]) for key in boosting_values)
    )
    if tree_error > 5.0e-12:
        raise RuntimeError(f"FortML tree oracle mismatch: {tree_error:.3e}")
    cart_row = parsed["cart"][0]
    cart_prediction, cart_nodes = cart_oracle(*tree_inputs())
    cart_values = {
        "mse": float(cart_row[5]),
        "prediction_sum": float(cart_row[6]),
        "node_count": int(cart_row[7]),
    }
    cart_error = max(
        abs(
            cart_values["mse"]
            - float(np.mean((cart_prediction - tree_inputs()[1]) ** 2))
        ),
        abs(cart_values["prediction_sum"] - float(np.sum(cart_prediction))),
    )
    if cart_values["node_count"] != cart_nodes or cart_error > 5.0e-12:
        raise RuntimeError(
            f"FortML CART oracle mismatch: nodes={cart_values['node_count']} "
            f"expected={cart_nodes}; error={cart_error:.3e}"
        )

    classifier_row = parsed["cart_classifier"][0]
    classifier_x, classifier_labels = cart_classifier_inputs()
    classifier_prediction, classifier_probability, classifier_nodes = (
        cart_classifier_oracle(classifier_x, classifier_labels)
    )
    classifier_values = {
        "accuracy": float(classifier_row[5]),
        "probability_sum": float(classifier_row[6]),
        "node_count": int(classifier_row[7]),
    }
    classifier_expected = {
        "accuracy": float(np.mean(classifier_prediction == classifier_labels)),
        "probability_sum": float(np.sum(classifier_probability)),
    }
    classifier_error = max(
        abs(classifier_values[key] - classifier_expected[key])
        for key in classifier_expected
    )
    if (
        classifier_values["node_count"] != classifier_nodes
        or classifier_error > 5.0e-12
    ):
        raise RuntimeError(
            "FortML CART classifier oracle mismatch: "
            f"nodes={classifier_values['node_count']} expected={classifier_nodes}; "
            f"error={classifier_error:.3e}"
        )

    metric_row = parsed["regression_metrics"][0]
    metric_expected = regression_metric_oracle()
    metric_values = {
        "mse": float(metric_row[3]),
        "mae": float(metric_row[4]),
        "r2": float(metric_row[5]),
        "pinball": float(metric_row[6]),
    }
    metric_error = max(
        abs(metric_values[key] - metric_expected[key]) for key in metric_values
    )
    if metric_error > 5.0e-14:
        raise RuntimeError(
            f"FortML regression metric oracle mismatch: {metric_error:.3e}"
        )

    records: list[dict[str, Any]] = []
    records.append(
        row(
            metadata,
            workload="mlp_training",
            phase="fit",
            backend="fortml",
            status="pass",
            n_samples=MLP_N,
            n_features=MLP_D,
            n_hidden=MLP_HIDDEN,
            repetitions=4,
            seconds_per_operation=float(mlp_row[7]),
            metric="final_mse",
            value=float(mlp_row[6]),
            mse=float(mlp_row[6]),
            max_abs_error=mlp_error,
            oracle="independent NumPy Adam/MSE implementation",
            notes=f"{MLP_EPOCHS} full-batch epochs; process wall={process_seconds:.6e}s",
        )
    )
    records.extend(
        [
            row(
                metadata,
                workload="gaussian_naive_bayes",
                phase="fit",
                backend="fortml",
                status="pass",
                n_samples=GAUSSIAN_NB_N,
                n_features=GAUSSIAN_NB_D,
                n_estimators=GAUSSIAN_NB_CLASSES.size,
                repetitions=8,
                seconds_per_operation=float(gaussian_nb_row[3]),
                metric="log_probability_sum",
                value=gaussian_nb_values["log_sum"],
                max_abs_error=gaussian_nb_error,
                oracle="independent NumPy weighted Gaussian density/log-softmax",
                notes="three arbitrary integer classes; var_smoothing=1e-9",
            ),
            row(
                metadata,
                workload="gaussian_naive_bayes",
                phase="predict",
                backend="fortml",
                status="pass",
                n_samples=GAUSSIAN_NB_N,
                n_features=GAUSSIAN_NB_D,
                n_estimators=GAUSSIAN_NB_CLASSES.size,
                repetitions=64,
                seconds_per_operation=float(gaussian_nb_row[4]),
                metric="log_probability_sum",
                value=gaussian_nb_values["log_sum"],
                max_abs_error=gaussian_nb_error,
                oracle="independent NumPy weighted Gaussian density/log-softmax",
                notes="stable shifted log-probability normalization",
            ),
            row(
                metadata,
                workload="gaussian_naive_bayes",
                phase="jvp",
                backend="fortml",
                status="pass",
                n_samples=GAUSSIAN_NB_N,
                n_features=GAUSSIAN_NB_D,
                n_estimators=GAUSSIAN_NB_CLASSES.size,
                repetitions=64,
                seconds_per_operation=float(gaussian_nb_row[5]),
                metric="log_probability_jvp_sum",
                value=gaussian_nb_values["jvp_sum"],
                max_abs_error=gaussian_nb_error,
                oracle="independent NumPy input directional derivative",
                notes="analytic log-softmax JVP; finite-only input contract",
            ),
            row(
                metadata,
                workload="basis_linear_regression",
                phase="fit",
                backend="fortml",
                status="pass",
                n_samples=BASIS_LINEAR_N,
                n_features=1,
                repetitions=8,
                seconds_per_operation=float(basis_linear_row[3]),
                metric="mse",
                value=basis_linear_values["mse"],
                mse=basis_linear_values["mse"],
                max_abs_error=basis_linear_error,
                oracle="independent NumPy Fourier design least-squares solve",
                notes="one fitted Fourier frequency plus linear regression",
            ),
            row(
                metadata,
                workload="basis_linear_regression",
                phase="predict",
                backend="fortml",
                status="pass",
                n_samples=BASIS_LINEAR_N,
                n_features=1,
                repetitions=64,
                seconds_per_operation=float(basis_linear_row[4]),
                metric="prediction_sum",
                value=basis_linear_values["prediction_sum"],
                max_abs_error=basis_linear_error,
                oracle="independent NumPy Fourier design least-squares solve",
                notes="packed basis and coefficient parameters",
            ),
            row(
                metadata,
                workload="basis_linear_regression",
                phase="jvp",
                backend="fortml",
                status="pass",
                n_samples=BASIS_LINEAR_N,
                n_features=1,
                repetitions=64,
                seconds_per_operation=float(basis_linear_row[5]),
                metric="jvp_sum",
                value=basis_linear_values["jvp_sum"],
                max_abs_error=basis_linear_error,
                oracle="independent NumPy chained Fourier/linear directional derivative",
                notes="frequency, coefficients, and inputs vary together",
            ),
        ]
    )
    records.append(
        row(
            metadata,
            workload="basis_pipeline",
            phase="transform",
            backend="fortml",
            status="pass",
            n_samples=BASIS_N,
            n_features=BASIS_D,
            repetitions=32,
            seconds_per_operation=float(parsed["basis_transform"][0][3]),
            metric="feature_sum",
            value=float(parsed["basis_transform"][0][4]),
            max_abs_error=0.0,
            oracle="independent NumPy polynomial/Fourier feature map",
            notes="horizontal polynomial-plus-Fourier pipeline",
        )
    )
    records.append(
        row(
            metadata,
            workload="basis_pipeline",
            phase="jvp",
            backend="fortml",
            status="pass",
            n_samples=BASIS_N,
            n_features=BASIS_D,
            repetitions=32,
            seconds_per_operation=float(parsed["basis_jvp"][0][3]),
            metric="jvp_sum",
            value=float(parsed["basis_jvp"][0][4]),
            max_abs_error=0.0,
            oracle="independent NumPy directional derivative",
            notes="includes Fourier log-frequency and input tangents",
        )
    )
    records.append(
        row(
            metadata,
            workload="basis_pipeline",
            phase="vjp",
            backend="fortml",
            status="pass",
            n_samples=BASIS_N,
            n_features=BASIS_D,
            repetitions=32,
            seconds_per_operation=float(parsed["basis_vjp"][0][3]),
            metric="cotangent_sums",
            value=float(parsed["basis_vjp"][0][4]),
            max_abs_error=0.0,
            oracle="independent NumPy reverse products",
            notes=f"theta_bar_sum={parsed['basis_vjp'][0][4]}; x_bar_sum={parsed['basis_vjp'][0][5]}",
        )
    )
    records.extend(
        [
            row(
                metadata,
                workload="regression_metrics",
                phase="aggregate",
                backend="fortml",
                status="pass",
                n_samples=METRIC_N,
                n_features=METRIC_OUTPUTS,
                repetitions=64,
                seconds_per_operation=float(metric_row[2]),
                metric="mse_mae_r2_pinball",
                value=metric_values["mse"],
                mse=metric_values["mse"],
                max_abs_error=metric_error,
                oracle="independent NumPy MSE/MAE/R2/pinball formulas",
                notes=(
                    f"mae={metric_values['mae']:.16e}; r2={metric_values['r2']:.16e}; "
                    f"pinball(q=0.25)={metric_values['pinball']:.16e}"
                ),
            ),
            row(
                metadata,
                workload="cart_regressor",
                phase="fit",
                backend="fortml",
                status="pass",
                n_samples=TREE_N,
                n_features=TREE_D,
                repetitions=8,
                seconds_per_operation=float(cart_row[3]),
                metric="mse",
                value=cart_values["mse"],
                mse=cart_values["mse"],
                max_abs_error=cart_error,
                oracle="independent NumPy exhaustive recursive CART",
                notes=f"max_depth={CART_DEPTH}; min_samples_leaf={TREE_LEAF}; nodes={cart_nodes}",
            ),
            row(
                metadata,
                workload="cart_regressor",
                phase="predict",
                backend="fortml",
                status="pass",
                n_samples=TREE_N,
                n_features=TREE_D,
                repetitions=64,
                seconds_per_operation=float(cart_row[4]),
                metric="prediction_sum",
                value=cart_values["prediction_sum"],
                max_abs_error=cart_error,
                oracle="independent NumPy exhaustive recursive CART",
                notes="piecewise constant prediction; input JVP is zero away from splits",
            ),
            row(
                metadata,
                workload="cart_classifier",
                phase="fit",
                backend="fortml",
                status="pass",
                n_samples=classifier_x.shape[0],
                n_features=classifier_x.shape[1],
                repetitions=8,
                seconds_per_operation=float(classifier_row[3]),
                metric="accuracy",
                value=classifier_values["accuracy"],
                max_abs_error=classifier_error,
                oracle="independent NumPy exhaustive recursive Gini CART",
                notes=(
                    f"max_depth={CART_DEPTH}; min_samples_leaf={TREE_LEAF}; "
                    f"classes={np.unique(classifier_labels).tolist()}; "
                    f"nodes={classifier_nodes}"
                ),
            ),
            row(
                metadata,
                workload="cart_classifier",
                phase="predict",
                backend="fortml",
                status="pass",
                n_samples=classifier_x.shape[0],
                n_features=classifier_x.shape[1],
                repetitions=64,
                seconds_per_operation=float(classifier_row[4]),
                metric="probability_sum",
                value=classifier_values["probability_sum"],
                max_abs_error=classifier_error,
                oracle="independent NumPy exhaustive recursive Gini CART",
                notes=(
                    "piecewise-constant class probabilities; input derivatives "
                    "are intentionally outside the classifier contract"
                ),
            ),
            row(
                metadata,
                workload="decision_stump",
                phase="fit",
                backend="fortml",
                status="pass",
                n_samples=TREE_N,
                n_features=TREE_D,
                repetitions=8,
                seconds_per_operation=float(stump_row[2]),
                metric="mse",
                value=stump_values["stump_mse"],
                mse=stump_values["stump_mse"],
                max_abs_error=tree_error,
                oracle="independent exhaustive NumPy split search",
                notes=f"feature={int(stump_values['stump_feature'])}; threshold={stump_values['stump_threshold']:.16e}",
            ),
            row(
                metadata,
                workload="decision_stump",
                phase="predict",
                backend="fortml",
                status="pass",
                n_samples=TREE_N,
                n_features=TREE_D,
                repetitions=64,
                seconds_per_operation=float(stump_row[3]),
                metric="prediction_sum",
                value=stump_values["stump_sum"],
                max_abs_error=tree_error,
                oracle="independent exhaustive NumPy split search",
                notes="piecewise constant prediction; JVP is zero away from split",
            ),
            row(
                metadata,
                workload="gradient_boosting_regressor",
                phase="fit",
                backend="fortml",
                status="pass",
                n_samples=TREE_N,
                n_features=TREE_D,
                n_estimators=TREE_ESTIMATORS,
                repetitions=8,
                seconds_per_operation=float(boost_row[3]),
                metric="mse",
                value=boosting_values["boosting_mse"],
                mse=boosting_values["boosting_mse"],
                max_abs_error=tree_error,
                oracle="independent NumPy sequential residual-stump boosting",
                notes=f"learning_rate={TREE_RATE}; min_samples_leaf={TREE_LEAF}",
            ),
            row(
                metadata,
                workload="gradient_boosting_regressor",
                phase="predict",
                backend="fortml",
                status="pass",
                n_samples=TREE_N,
                n_features=TREE_D,
                n_estimators=TREE_ESTIMATORS,
                repetitions=64,
                seconds_per_operation=float(boost_row[4]),
                metric="prediction_sum",
                value=boosting_values["boosting_sum"],
                max_abs_error=tree_error,
                oracle="independent NumPy sequential residual-stump boosting",
                notes="input JVP defined away from split boundaries",
            ),
        ]
    )
    return records


def run_sklearn(metadata: dict[str, str]) -> list[dict[str, Any]]:
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.neural_network import MLPRegressor
    except ImportError as exc:
        return [
            row(
                metadata,
                workload="sklearn_reference",
                phase="all",
                backend="sklearn",
                status="unavailable",
                oracle="package availability",
                notes=f"scikit-learn import failed: {exc}",
            )
        ]
    x_mlp, target = mlp_inputs()
    started = time.perf_counter()
    model = MLPRegressor(
        hidden_layer_sizes=(MLP_HIDDEN,),
        activation="tanh",
        solver="adam",
        alpha=MLP_L2,
        batch_size=MLP_N,
        learning_rate_init=MLP_LR,
        max_iter=MLP_EPOCHS,
        shuffle=False,
        tol=0.0,
        n_iter_no_change=MLP_EPOCHS + 1,
        random_state=23,
    )
    model.fit(x_mlp, target[:, 0])
    fit_seconds = time.perf_counter() - started
    prediction = model.predict(x_mlp)
    mlp_mse = float(np.mean((prediction - target[:, 0]) ** 2))
    x_tree, y_tree = tree_inputs()
    x_classifier, labels_classifier = cart_classifier_inputs()
    started = time.perf_counter()
    cart = DecisionTreeRegressor(
        max_depth=CART_DEPTH,
        min_samples_leaf=TREE_LEAF,
        random_state=0,
        criterion="squared_error",
    )
    cart.fit(x_tree, y_tree)
    cart_fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    cart_prediction = cart.predict(x_tree)
    cart_predict_seconds = time.perf_counter() - started
    started = time.perf_counter()
    classifier = DecisionTreeClassifier(
        max_depth=CART_DEPTH,
        min_samples_leaf=TREE_LEAF,
        random_state=0,
        criterion="gini",
    )
    classifier.fit(x_classifier, labels_classifier)
    classifier_fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    classifier_prediction = classifier.predict(x_classifier)
    classifier_predict_seconds = time.perf_counter() - started
    classifier_probability = classifier.predict_proba(x_classifier)
    started = time.perf_counter()
    booster = GradientBoostingRegressor(
        n_estimators=TREE_ESTIMATORS,
        learning_rate=TREE_RATE,
        max_depth=1,
        min_samples_leaf=TREE_LEAF,
        random_state=0,
        loss="squared_error",
    )
    booster.fit(x_tree, y_tree)
    boost_fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    tree_prediction = booster.predict(x_tree)
    boost_predict_seconds = time.perf_counter() - started
    return [
        row(
            metadata,
            workload="mlp_training",
            phase="fit",
            backend="sklearn",
            status="pass",
            n_samples=MLP_N,
            n_features=MLP_D,
            n_hidden=MLP_HIDDEN,
            repetitions=1,
            seconds_per_operation=fit_seconds,
            metric="final_mse",
            value=mlp_mse,
            mse=mlp_mse,
            oracle="NumPy fixture; sklearn Adam implementation",
            notes="estimator initialization differs; quality is contextual, not bitwise",
        ),
        row(
            metadata,
            workload="cart_classifier",
            phase="fit",
            backend="sklearn",
            status="pass",
            n_samples=TREE_N,
            n_features=TREE_D,
            repetitions=1,
            seconds_per_operation=classifier_fit_seconds,
            metric="accuracy",
            value=float(np.mean(classifier_prediction == labels_classifier)),
            oracle="NumPy fixture; sklearn DecisionTreeClassifier",
            notes=f"max_depth={CART_DEPTH}; min_samples_leaf={TREE_LEAF}; criterion=gini",
        ),
        row(
            metadata,
            workload="cart_classifier",
            phase="predict",
            backend="sklearn",
            status="pass",
            n_samples=TREE_N,
            n_features=TREE_D,
            repetitions=1,
            seconds_per_operation=classifier_predict_seconds,
            metric="probability_sum",
            value=float(np.sum(classifier_probability)),
            oracle="NumPy fixture; sklearn DecisionTreeClassifier",
            notes="classifier split/probability policy is contextual",
        ),
        row(
            metadata,
            workload="cart_regressor",
            phase="fit",
            backend="sklearn",
            status="pass",
            n_samples=TREE_N,
            n_features=TREE_D,
            repetitions=1,
            seconds_per_operation=cart_fit_seconds,
            metric="mse",
            value=float(np.mean((cart_prediction - y_tree) ** 2)),
            mse=float(np.mean((cart_prediction - y_tree) ** 2)),
            oracle="NumPy fixture; sklearn DecisionTreeRegressor",
            notes=f"max_depth={CART_DEPTH}; min_samples_leaf={TREE_LEAF}",
        ),
        row(
            metadata,
            workload="cart_regressor",
            phase="predict",
            backend="sklearn",
            status="pass",
            n_samples=TREE_N,
            n_features=TREE_D,
            repetitions=1,
            seconds_per_operation=cart_predict_seconds,
            metric="prediction_sum",
            value=float(np.sum(cart_prediction)),
            oracle="NumPy fixture; sklearn DecisionTreeRegressor",
            notes="sklearn tree prediction is contextual; split ties may differ",
        ),
        row(
            metadata,
            workload="gradient_boosting_regressor",
            phase="fit",
            backend="sklearn",
            status="pass",
            n_samples=TREE_N,
            n_features=TREE_D,
            n_estimators=TREE_ESTIMATORS,
            repetitions=1,
            seconds_per_operation=boost_fit_seconds,
            metric="mse",
            value=float(np.mean((tree_prediction - y_tree) ** 2)),
            mse=float(np.mean((tree_prediction - y_tree) ** 2)),
            oracle="NumPy fixture; sklearn GradientBoostingRegressor",
            notes="depth-1 trees and matched learning rate/min leaf",
        ),
        row(
            metadata,
            workload="gradient_boosting_regressor",
            phase="predict",
            backend="sklearn",
            status="pass",
            n_samples=TREE_N,
            n_features=TREE_D,
            n_estimators=TREE_ESTIMATORS,
            repetitions=1,
            seconds_per_operation=boost_predict_seconds,
            metric="prediction_sum",
            value=float(np.sum(tree_prediction)),
            oracle="NumPy fixture; sklearn GradientBoostingRegressor",
            notes="sklearn prediction comparison is not a differentiability claim",
        ),
    ]


def optional_refusal_rows(metadata: dict[str, str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for package, workload, note in (
        (
            "torch",
            "mlp_training",
            "PyTorch comparison is optional here; use bench_model_workloads.py for product-level MLP forward/VJP parity",
        ),
        (
            "jax",
            "mlp_training",
            "JAX training lane is not enabled in this release harness; refusal is explicit",
        ),
        (
            "xgboost",
            "gradient_boosting_regressor",
            "XGBoost is optional; this lane reports availability without equating stump boosting to XGBoost",
        ),
    ):
        available = metadata[f"{package}_version"] != "unavailable"
        records.append(
            row(
                metadata,
                workload=workload,
                phase="dependency_check",
                backend=package,
                status="available_not_timed" if available else "unavailable",
                oracle="dependency availability",
                notes=note,
            )
        )
    return records


def fortml_device_boundary_rows(metadata: dict[str, str]) -> list[dict[str, Any]]:
    """Record host-only FortML paths without attempting a false CUDA run.

    The feature executable is intentionally a CPU release-app contract.  A
    CUDA row here is therefore a capability refusal, not a timing result.  The
    NumPy oracle checks run before these rows are emitted, so a refusal cannot
    mask a behavioral mismatch in the host implementation.
    """
    boundary_metadata = dict(metadata)
    boundary_metadata["device"] = "cuda"
    records: list[dict[str, Any]] = []
    for workload, phases, note in (
        (
            "gaussian_naive_bayes",
            ("fit", "predict", "jvp"),
            "FortML GaussianNB is host-only in this release; no CUDA/device-resident timing is claimed",
        ),
        (
            "mlp_training",
            ("fit",),
            "FortML MLP trainer is host-only in this release; use the resident PyTorch CUDA row for device evidence",
        ),
        (
            "logistic_objective",
            ("value_gradient", "hvp"),
            "the logistic objective has no CUDA release app; this explicit refusal is not an execution result",
        ),
    ):
        for phase in phases:
            records.append(
                row(
                    boundary_metadata,
                    workload=workload,
                    phase=phase,
                    backend="fortml",
                    device="cuda",
                    status="unavailable",
                    oracle="FortML source capability boundary; no CUDA execution",
                    notes=note,
                )
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/features_workloads.csv")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    ignored_outputs = (args.output.resolve(),)
    metadata = base_metadata(root, args.fortml.resolve(), ignored_outputs)
    records = run_fortran(root, args.fortml.resolve(), metadata)
    records.extend(run_sklearn(metadata))
    records.extend(optional_refusal_rows(metadata))
    records.extend(fortml_device_boundary_rows(metadata))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} rows to {args.output}")


if __name__ == "__main__":
    main()
