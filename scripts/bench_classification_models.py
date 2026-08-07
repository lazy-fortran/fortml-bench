#!/usr/bin/env python3
"""Benchmark multinomial and neural classification against NumPy oracles.

The fixture deliberately uses arbitrary integer labels and a small, fixed
three-class problem.  NumPy implements both the regularized multinomial
objective and the full-batch Adam MLP update independently.  Optional
scikit-learn rows are contextual references; a FortML row is accepted only
when its release app emits a complete oracle file and matching checksums.

The FortML app protocol is intentionally narrow and machine-readable.  The
target is ``fortml_bench_classifiers`` and the app receives
``FORTML_BENCH_CLASSIFIER_ORACLE``.  Its CSV quantities are ``label``,
``softmax_probability``, ``softmax_prediction``, ``mlp_probability``, and
``mlp_prediction``.  The stdout records are documented in the report.  A
missing target/dependency produces explicit ``unavailable`` rows.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np


N_SAMPLES = 192
N_FEATURES = 6
N_CLASSES = 3
N_HIDDEN = 12
CLASS_LABELS = np.array([-7, 3, 11], dtype=np.int64)
SOFTMAX_L2 = 5.0e-2
SOFTMAX_MAX_ITERATIONS = 1000
SOFTMAX_TOLERANCE = 1.0e-9
MLP_EPOCHS = 80
MLP_LEARNING_RATE = 3.0e-2
MLP_L2 = 1.0e-3
MLP_INITIALIZATION_SEED = 23


def git_revision(repository: Path) -> str:
    revision = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).strip()
    return revision + ("+dirty" if dirty else "")


def fixture() -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(1, N_SAMPLES + 1, dtype=np.float64)[:, None]
    columns = np.arange(1, N_FEATURES + 1, dtype=np.float64)[None, :]
    x = np.sin(0.017 * rows + 0.071 * columns)
    x += 0.2 * np.cos(0.009 * rows * columns)
    score = np.column_stack(
        (
            0.4 * x[:, 0] - 0.2 * x[:, 1] + 0.1 * x[:, 2],
            -0.1 * x[:, 0] + 0.5 * x[:, 1] - 0.2 * x[:, 3],
            0.2 * x[:, 2] + 0.3 * x[:, 4] - 0.4 * x[:, 5],
        )
    )
    # Interleaving by row avoids a class-contiguous fixture and exercises the
    # sorted arbitrary-label convention used by both FortML classifiers.
    phase = rows[:, 0]
    class_bias = np.column_stack(
        (
            0.3 * np.sin(0.11 * phase),
            0.3 * np.cos(0.11 * phase),
            0.3 * np.sin(0.13 * phase + 1.0),
        )
    )
    labels = CLASS_LABELS[np.argmax(score + class_bias, axis=1)]
    return x, labels


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=1, keepdims=True)


def softmax_objective(
    theta: np.ndarray, x: np.ndarray, encoded: np.ndarray, l2: float
) -> tuple[float, np.ndarray, np.ndarray]:
    n_features = x.shape[1]
    coefficients = theta[: n_features * N_CLASSES].reshape(
        (n_features, N_CLASSES), order="F"
    )
    intercept = theta[n_features * N_CLASSES :]
    logits = x @ coefficients + intercept
    probabilities = stable_softmax(logits)
    residual = probabilities.copy()
    residual[np.arange(x.shape[0]), encoded] -= 1.0
    value = float(
        -np.mean(
            np.log(np.maximum(probabilities[np.arange(x.shape[0]), encoded], 1.0e-300))
        )
        + 0.5 * l2 * np.sum(coefficients * coefficients)
    )
    gradient_coefficients = x.T @ residual / x.shape[0] + l2 * coefficients
    gradient_intercept = np.mean(residual, axis=0)
    gradient = np.concatenate(
        (gradient_coefficients.reshape(-1, order="F"), gradient_intercept)
    )
    return value, gradient, probabilities


def softmax_oracle(x: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Solve the regularized objective with independent damped Newton steps."""
    encoded = np.searchsorted(CLASS_LABELS, labels)
    n_features = x.shape[1]
    theta = np.zeros(n_features * N_CLASSES + N_CLASSES, dtype=np.float64)
    # The Hessian is intentionally assembled here rather than calling a
    # package optimizer: this is the behavioral oracle for the Fortran fit.
    for _ in range(SOFTMAX_MAX_ITERATIONS):
        value, gradient, probabilities = softmax_objective(
            theta, x, encoded, SOFTMAX_L2
        )
        hessian = np.zeros((theta.size, theta.size), dtype=np.float64)
        for row in range(x.shape[0]):
            covariance = np.diag(probabilities[row]) - np.outer(
                probabilities[row], probabilities[row]
            )
            for first_class in range(N_CLASSES):
                first = first_class * n_features
                for second_class in range(N_CLASSES):
                    second = second_class * n_features
                    block = covariance[first_class, second_class] / x.shape[0]
                    hessian[
                        first : first + n_features, second : second + n_features
                    ] += block * np.outer(x[row], x[row])
                    intercept_offset = n_features * N_CLASSES
                    hessian[
                        first : first + n_features, intercept_offset + second_class
                    ] += block * x[row]
                    hessian[
                        intercept_offset + first_class, second : second + n_features
                    ] += block * x[row]
                    hessian[
                        intercept_offset + first_class, intercept_offset + second_class
                    ] += block
        hessian[: n_features * N_CLASSES, : n_features * N_CLASSES] += (
            SOFTMAX_L2 * np.eye(n_features * N_CLASSES)
        )
        try:
            direction = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            direction = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        if np.linalg.norm(direction, ord=np.inf) <= SOFTMAX_TOLERANCE:
            break
        step = 1.0
        while step > 1.0e-10:
            candidate = theta - step * direction
            candidate_value, _, _ = softmax_objective(candidate, x, encoded, SOFTMAX_L2)
            if candidate_value <= value - 1.0e-4 * step * np.dot(gradient, direction):
                theta = candidate
                break
            step *= 0.5
        else:
            theta -= 1.0e-3 * direction
        if np.linalg.norm(gradient, ord=np.inf) <= SOFTMAX_TOLERANCE:
            break
    value, gradient, probabilities = softmax_objective(theta, x, encoded, SOFTMAX_L2)
    predicted = CLASS_LABELS[np.argmax(probabilities, axis=1)]
    return {
        "theta": theta,
        "probabilities": probabilities,
        "predicted": predicted,
        "loss": value,
        "gradient_norm": float(np.linalg.norm(gradient)),
    }


def mlp_initialize() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def layer(
        in_features: int, out_features: int, index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        scale = np.sqrt(6.0 / (in_features + out_features))
        indices = np.arange(1, in_features * out_features + 1, dtype=np.float64)
        phases = MLP_INITIALIZATION_SEED + 1009 * index + 9176 * indices
        weight = (scale * np.sin(phases)).reshape(
            (in_features, out_features), order="F"
        )
        bias_indices = np.arange(1, out_features + 1, dtype=np.float64)
        bias = (
            0.01
            * scale
            * np.sin(MLP_INITIALIZATION_SEED + 1009 * index + 7919 * bias_indices)
        )
        return weight, bias

    weight_1, bias_1 = layer(N_FEATURES, N_HIDDEN, 1)
    weight_2, bias_2 = layer(N_HIDDEN, N_CLASSES, 2)
    return weight_1, bias_1, weight_2, bias_2


def mlp_loss_gradient(
    x: np.ndarray,
    encoded: np.ndarray,
    weights: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    l2: float,
) -> tuple[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    weight_1, bias_1, weight_2, bias_2 = weights
    hidden = np.tanh(x @ weight_1 + bias_1)
    logits = hidden @ weight_2 + bias_2
    probabilities = stable_softmax(logits)
    residual = probabilities.copy()
    residual[np.arange(x.shape[0]), encoded] -= 1.0
    loss = float(
        -np.mean(
            np.log(np.maximum(probabilities[np.arange(x.shape[0]), encoded], 1.0e-300))
        )
        + 0.5 * l2 * sum(np.sum(value * value) for value in weights)
    )
    residual /= x.shape[0]
    weight_2_bar = hidden.T @ residual + l2 * weight_2
    bias_2_bar = np.sum(residual, axis=0) + l2 * bias_2
    hidden_bar = residual @ weight_2.T
    preactivation_bar = hidden_bar * (1.0 - hidden * hidden)
    weight_1_bar = x.T @ preactivation_bar + l2 * weight_1
    bias_1_bar = np.sum(preactivation_bar, axis=0) + l2 * bias_1
    return loss, (weight_1_bar, bias_1_bar, weight_2_bar, bias_2_bar), probabilities


def mlp_oracle(x: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    encoded = np.searchsorted(CLASS_LABELS, labels)
    weights = mlp_initialize()
    initial_weights = tuple(value.copy() for value in weights)
    moments = tuple(np.zeros_like(value) for value in weights)
    second = tuple(np.zeros_like(value) for value in weights)
    initial_loss, _, _ = mlp_loss_gradient(x, encoded, weights, MLP_L2)
    for step in range(1, MLP_EPOCHS + 1):
        loss, gradient, _ = mlp_loss_gradient(x, encoded, weights, MLP_L2)
        updated_weights: list[np.ndarray] = []
        updated_moments: list[np.ndarray] = []
        updated_second: list[np.ndarray] = []
        for value, derivative, first, second_moment in zip(
            weights, gradient, moments, second
        ):
            first = 0.9 * first + 0.1 * derivative
            second_moment = 0.999 * second_moment + 0.001 * derivative * derivative
            first_hat = first / (1.0 - 0.9**step)
            second_hat = second_moment / (1.0 - 0.999**step)
            updated_weights.append(
                value - MLP_LEARNING_RATE * first_hat / (np.sqrt(second_hat) + 1.0e-8)
            )
            updated_moments.append(first)
            updated_second.append(second_moment)
        weights = tuple(updated_weights)  # type: ignore[assignment]
        moments = tuple(updated_moments)
        second = tuple(updated_second)
    final_loss, _, probabilities = mlp_loss_gradient(x, encoded, weights, MLP_L2)
    predicted = CLASS_LABELS[np.argmax(probabilities, axis=1)]
    return {
        "initial_weights": initial_weights,
        "weights": weights,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "probabilities": probabilities,
        "predicted": predicted,
    }


def checked_metrics(
    labels: np.ndarray, predicted: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    if predicted.shape != labels.shape or probabilities.shape != (N_SAMPLES, N_CLASSES):
        raise RuntimeError("multiclass output shape does not match the fixture")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise RuntimeError("multiclass probabilities are not finite/nonnegative")
    normalization_error = float(np.max(np.abs(probabilities.sum(axis=1) - 1.0)))
    if normalization_error > 2.0e-13:
        raise RuntimeError(f"probability normalization error {normalization_error:.3e}")
    selected = probabilities[
        np.arange(N_SAMPLES), np.searchsorted(CLASS_LABELS, labels)
    ]
    return {
        "accuracy": float(np.mean(predicted == labels)),
        "log_loss": float(-np.mean(np.log(np.maximum(selected, 1.0e-300)))),
        "probability_normalization_error": normalization_error,
    }


def parse_fortran(stdout: str) -> dict[str, list[str]]:
    records: dict[str, list[str]] = {}
    pattern = re.compile(
        r"^(softmax_fit|softmax_predict|mlp_classifier_fit|mlp_classifier_predict),(.*)$"
    )
    for line in stdout.splitlines():
        match = pattern.match(line.strip())
        if match:
            records[match.group(1)] = [
                part.strip() for part in match.group(2).split(",")
            ]
    return records


def read_fortran_oracle(path: Path) -> dict[str, np.ndarray]:
    arrays = {
        "label": np.full(N_SAMPLES, np.iinfo(np.int64).min, dtype=np.int64),
        "softmax_probability": np.full((N_SAMPLES, N_CLASSES), np.nan),
        "softmax_prediction": np.full(
            N_SAMPLES, np.iinfo(np.int64).min, dtype=np.int64
        ),
        "mlp_probability": np.full((N_SAMPLES, N_CLASSES), np.nan),
        "mlp_prediction": np.full(N_SAMPLES, np.iinfo(np.int64).min, dtype=np.int64),
    }
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            quantity = row["quantity"]
            if quantity not in arrays:
                raise RuntimeError(
                    f"unknown FortML classifier oracle quantity {quantity!r}"
                )
            first = int(row["row"]) - 1
            second = int(row.get("column", "1")) - 1
            if not 0 <= first < N_SAMPLES:
                raise RuntimeError("FortML classifier oracle row is out of range")
            target = arrays[quantity]
            value = float(row["value"])
            if target.ndim == 1:
                target[first] = int(value)
            else:
                if not 0 <= second < N_CLASSES:
                    raise RuntimeError(
                        "FortML classifier oracle column is out of range"
                    )
                target[first, second] = value
    if np.any(arrays["label"] == np.iinfo(np.int64).min):
        raise RuntimeError("FortML classifier oracle omitted labels")
    for quantity in ("softmax_probability", "mlp_probability"):
        if not np.isfinite(arrays[quantity]).all():
            raise RuntimeError(f"FortML classifier oracle omitted {quantity}")
    for quantity in ("softmax_prediction", "mlp_prediction"):
        if np.any(arrays[quantity] == np.iinfo(np.int64).min):
            raise RuntimeError(f"FortML classifier oracle omitted {quantity}")
    return arrays


def metadata(root: Path, fortml: Path) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": git_revision(fortml),
        "fortnum_revision": git_revision(fortml.parent / "fortnum"),
        "benchmark_revision": git_revision(root),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }


def unavailable_rows(
    details: dict[str, str], backend: str, status: str, note: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for workload, phases in (
        ("softmax_regression", ("fit", "predict")),
        ("mlp_classifier", ("fit", "predict")),
    ):
        for phase in phases:
            row: dict[str, Any] = dict(details)
            row.update(
                {
                    "workload": workload,
                    "phase": phase,
                    "backend": backend,
                    "status": status,
                    "n_samples": N_SAMPLES,
                    "n_features": N_FEATURES,
                    "n_hidden": N_HIDDEN if workload == "mlp_classifier" else 0,
                    "n_classes": N_CLASSES,
                    "notes": note,
                }
            )
            rows.append(row)
    return rows


def run_numpy_reference(
    x: np.ndarray, labels: np.ndarray, details: dict[str, str]
) -> list[dict[str, Any]]:
    """Time the independent references and retain their checked metrics."""
    started = time.perf_counter()
    softmax = softmax_oracle(x, labels)
    softmax_fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    softmax_predicted = softmax["predicted"]
    softmax_probabilities = softmax["probabilities"]
    softmax_predict_seconds = time.perf_counter() - started
    softmax_metrics = checked_metrics(labels, softmax_predicted, softmax_probabilities)
    started = time.perf_counter()
    mlp = mlp_oracle(x, labels)
    mlp_fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    mlp_predicted = mlp["predicted"]
    mlp_probabilities = mlp["probabilities"]
    mlp_predict_seconds = time.perf_counter() - started
    mlp_metrics = checked_metrics(labels, mlp_predicted, mlp_probabilities)
    rows: list[dict[str, Any]] = []
    for workload, metrics, fit_seconds, predict_seconds, note in (
        (
            "softmax_regression",
            softmax_metrics,
            softmax_fit_seconds,
            softmax_predict_seconds,
            f"damped Newton; l2={SOFTMAX_L2:g}; gradient norm={softmax['gradient_norm']:.3e}",
        ),
        (
            "mlp_classifier",
            mlp_metrics,
            mlp_fit_seconds,
            mlp_predict_seconds,
            f"full-batch Adam; epochs={MLP_EPOCHS}; l2={MLP_L2:g}; loss {mlp['initial_loss']:.6g}->{mlp['final_loss']:.6g}",
        ),
    ):
        for phase, seconds in (("fit", fit_seconds), ("predict", predict_seconds)):
            row: dict[str, Any] = dict(details)
            row.update(
                {
                    "workload": workload,
                    "phase": phase,
                    "backend": "numpy_oracle",
                    "status": "pass",
                    "n_samples": N_SAMPLES,
                    "n_features": N_FEATURES,
                    "n_hidden": N_HIDDEN if workload == "mlp_classifier" else 0,
                    "n_classes": N_CLASSES,
                    "seconds_per_operation": seconds,
                    "max_abs_error": 0.0,
                    **metrics,
                    "oracle": "independent NumPy implementation (behavioral reference)",
                    "notes": note,
                }
            )
            rows.append(row)
    return rows


def run_sklearn(
    x: np.ndarray, labels: np.ndarray, details: dict[str, str]
) -> list[dict[str, Any]]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
    except ImportError as error:
        return unavailable_rows(
            details, "sklearn", "unavailable", f"optional dependency missing: {error}"
        )
    rows: list[dict[str, Any]] = []
    softmax_started = time.perf_counter()
    softmax = LogisticRegression(
        C=1.0 / SOFTMAX_L2,
        fit_intercept=True,
        solver="lbfgs",
        max_iter=SOFTMAX_MAX_ITERATIONS,
        tol=SOFTMAX_TOLERANCE,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        softmax.fit(x, labels)
    softmax_fit_seconds = time.perf_counter() - softmax_started
    started = time.perf_counter()
    softmax_probabilities = softmax.predict_proba(x)
    softmax_predicted = softmax.predict(x)
    softmax_predict_seconds = time.perf_counter() - started
    softmax_metrics = checked_metrics(labels, softmax_predicted, softmax_probabilities)
    for phase, seconds in (
        ("fit", softmax_fit_seconds),
        ("predict", softmax_predict_seconds),
    ):
        row: dict[str, Any] = dict(details)
        row.update(
            {
                "workload": "softmax_regression",
                "phase": phase,
                "backend": "sklearn",
                "status": "pass",
                "n_samples": N_SAMPLES,
                "n_features": N_FEATURES,
                "n_hidden": 0,
                "n_classes": N_CLASSES,
                "seconds_per_operation": seconds,
                **softmax_metrics,
                "oracle": "independent NumPy softmax objective and generated labels",
                "notes": "scikit-learn multinomial lbfgs; CPU single process",
            }
        )
        rows.append(row)
    mlp_started = time.perf_counter()
    mlp = MLPClassifier(
        hidden_layer_sizes=(N_HIDDEN,),
        activation="tanh",
        solver="adam",
        batch_size=N_SAMPLES,
        learning_rate_init=MLP_LEARNING_RATE,
        alpha=MLP_L2,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1.0e-8,
        max_iter=MLP_EPOCHS,
        shuffle=False,
        random_state=MLP_INITIALIZATION_SEED,
        tol=0.0,
        n_iter_no_change=MLP_EPOCHS + 1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlp.fit(x, labels)
    mlp_fit_seconds = time.perf_counter() - mlp_started
    started = time.perf_counter()
    mlp_probabilities = mlp.predict_proba(x)
    mlp_predicted = mlp.predict(x)
    mlp_predict_seconds = time.perf_counter() - started
    mlp_metrics = checked_metrics(labels, mlp_predicted, mlp_probabilities)
    for phase, seconds in (("fit", mlp_fit_seconds), ("predict", mlp_predict_seconds)):
        row = dict(details)
        row.update(
            {
                "workload": "mlp_classifier",
                "phase": phase,
                "backend": "sklearn",
                "status": "pass",
                "n_samples": N_SAMPLES,
                "n_features": N_FEATURES,
                "n_hidden": N_HIDDEN,
                "n_classes": N_CLASSES,
                "seconds_per_operation": seconds,
                **mlp_metrics,
                "oracle": "independent NumPy full-batch Adam and generated labels",
                "notes": "scikit-learn tanh MLPClassifier; optimizer settings matched where exposed",
            }
        )
        rows.append(row)
    return rows


def run_torch(
    x: np.ndarray, labels: np.ndarray, details: dict[str, str]
) -> list[dict[str, Any]]:
    """Run the same MLP update on resident PyTorch CPU/CUDA tensors."""
    try:
        import torch
    except ImportError as error:
        return [
            row
            for row in unavailable_rows(
                details,
                "pytorch",
                "unavailable",
                f"optional dependency missing: {error}",
            )
            if row["workload"] == "mlp_classifier"
        ]
    torch.set_num_threads(1)
    expected = mlp_oracle(x, labels)
    rows: list[dict[str, Any]] = []
    for device_name in ("cpu", "cuda"):
        device_details = dict(details)
        device_details.update(
            {
                "device": device_name,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda or "unavailable",
            }
        )
        if device_name == "cuda" and not torch.cuda.is_available():
            rows.extend(
                row
                for row in unavailable_rows(
                    device_details,
                    "pytorch_cuda",
                    "unavailable",
                    "torch.cuda.is_available() is false",
                )
                if row["workload"] == "mlp_classifier"
            )
            continue
        target = torch.as_tensor(labels, dtype=torch.long, device=device_name)
        features = torch.as_tensor(x, dtype=torch.float64, device=device_name)
        model = torch.nn.Sequential(
            torch.nn.Linear(N_FEATURES, N_HIDDEN, dtype=torch.float64),
            torch.nn.Tanh(),
            torch.nn.Linear(N_HIDDEN, N_CLASSES, dtype=torch.float64),
        ).to(device=device_name)
        with torch.no_grad():
            weight_1, bias_1, weight_2, bias_2 = expected["initial_weights"]
            model[0].weight.copy_(
                torch.as_tensor(weight_1.T, dtype=torch.float64, device=device_name)
            )
            model[0].bias.copy_(
                torch.as_tensor(bias_1, dtype=torch.float64, device=device_name)
            )
            model[2].weight.copy_(
                torch.as_tensor(weight_2.T, dtype=torch.float64, device=device_name)
            )
            model[2].bias.copy_(
                torch.as_tensor(bias_2, dtype=torch.float64, device=device_name)
            )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=MLP_LEARNING_RATE, eps=1.0e-8, foreach=False
        )
        started = time.perf_counter()
        for _ in range(MLP_EPOCHS):
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(logits, target)
            loss = loss + 0.5 * MLP_L2 * sum(
                torch.sum(parameter * parameter) for parameter in model.parameters()
            )
            loss.backward()
            optimizer.step()
        if device_name == "cuda":
            torch.cuda.synchronize()
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        with torch.no_grad():
            probabilities = torch.softmax(model(features), dim=1)
            predicted = torch.argmax(probabilities, dim=1)
        if device_name == "cuda":
            torch.cuda.synchronize()
        predict_seconds = time.perf_counter() - started
        probability_array = probabilities.detach().cpu().numpy()
        predicted_array = CLASS_LABELS[predicted.detach().cpu().numpy()]
        metrics = checked_metrics(labels, predicted_array, probability_array)
        error = float(np.max(np.abs(probability_array - expected["probabilities"])))
        if error > 2.0e-10:
            raise RuntimeError(
                f"PyTorch {device_name} MLP oracle mismatch: {error:.3e}"
            )
        for phase, seconds in (("fit", fit_seconds), ("predict", predict_seconds)):
            row: dict[str, Any] = dict(device_details)
            row.update(
                {
                    "workload": "mlp_classifier",
                    "phase": phase,
                    "backend": f"pytorch_{device_name}",
                    "status": "pass",
                    "n_samples": N_SAMPLES,
                    "n_features": N_FEATURES,
                    "n_hidden": N_HIDDEN,
                    "n_classes": N_CLASSES,
                    "seconds_per_operation": seconds,
                    "max_abs_error": error,
                    **metrics,
                    "oracle": "independent NumPy full-batch Adam MLP",
                    "notes": "resident float64 tensors; explicit L2 term; foreach=False",
                }
            )
            rows.append(row)
    return rows


def run_fortml(
    fortml: Path, root: Path, details: dict[str, str], target: str
) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update(
        {"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"}
    )
    build = subprocess.run(
        ["fo", "build", "--flag", "-O3"],
        cwd=fortml,
        env=environment,
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        note = (
            build.stderr.strip().splitlines()[-1]
            if build.stderr.strip()
            else "fo build failed"
        )
        return unavailable_rows(details, "fortml", "build_failed", note)
    x, labels = fixture()
    softmax_expected = softmax_oracle(x, labels)
    mlp_expected = mlp_oracle(x, labels)
    with tempfile.TemporaryDirectory() as directory:
        oracle_path = Path(directory) / "classifier_oracle.csv"
        run_environment = environment.copy()
        run_environment["FORTML_BENCH_CLASSIFIER_ORACLE"] = str(oracle_path)
        started = time.perf_counter()
        completed = subprocess.run(
            ["fo", "exec", "--no-build", target],
            cwd=fortml,
            env=run_environment,
            capture_output=True,
            text=True,
        )
        wall = time.perf_counter() - started
        if completed.returncode != 0:
            stderr_lines = completed.stderr.strip().splitlines()
            note = next(
                (
                    line.strip()
                    for line in reversed(stderr_lines)
                    if "fo exec:" in line.lower() or "no such target" in line.lower()
                ),
                stderr_lines[-1].strip() if stderr_lines else "target unavailable",
            )
            if (
                "not found" in note.lower()
                or "unknown" in note.lower()
                or "no such" in note.lower()
                or "no target" in note.lower()
            ):
                return unavailable_rows(
                    details, "fortml", "unavailable", f"{target}: {note}"
                )
            return unavailable_rows(
                details, "fortml", "execution_failed", f"{target}: {note}"
            )
        if not oracle_path.is_file():
            return unavailable_rows(
                details,
                "fortml",
                "parse_failed",
                "classifier app did not write FORTML_BENCH_CLASSIFIER_ORACLE",
            )
        actual = read_fortran_oracle(oracle_path)
    if not np.array_equal(actual["label"], labels):
        raise RuntimeError(
            "FortML classifier oracle labels differ from the NumPy fixture"
        )
    softmax_error = max(
        float(
            np.max(
                np.abs(
                    actual["softmax_probability"] - softmax_expected["probabilities"]
                )
            )
        ),
        float(
            np.max(np.abs(actual["softmax_prediction"] - softmax_expected["predicted"]))
        ),
    )
    mlp_error = max(
        float(
            np.max(np.abs(actual["mlp_probability"] - mlp_expected["probabilities"]))
        ),
        float(np.max(np.abs(actual["mlp_prediction"] - mlp_expected["predicted"]))),
    )
    # The optimizers are independent, so probability checks use a practical
    # convergence envelope.  Integer predictions remain an exact check.
    if softmax_error > 2.0e-5 or mlp_error > 2.0e-10:
        raise RuntimeError(
            f"FortML classifier oracle mismatch: softmax={softmax_error:.3e}, mlp={mlp_error:.3e}"
        )
    rows: list[dict[str, Any]] = []
    for workload, probabilities, predicted, error, oracle_name in (
        (
            "softmax_regression",
            actual["softmax_probability"],
            actual["softmax_prediction"],
            softmax_error,
            "independent NumPy damped-Newton softmax",
        ),
        (
            "mlp_classifier",
            actual["mlp_probability"],
            actual["mlp_prediction"],
            mlp_error,
            "independent NumPy full-batch Adam MLP",
        ),
    ):
        metrics = checked_metrics(labels, predicted, probabilities)
        for phase in ("fit", "predict"):
            record: dict[str, Any] = dict(details)
            record.update(
                {
                    "workload": workload,
                    "phase": phase,
                    "backend": "fortml",
                    "status": "pass",
                    "n_samples": N_SAMPLES,
                    "n_features": N_FEATURES,
                    "n_hidden": N_HIDDEN if workload == "mlp_classifier" else 0,
                    "n_classes": N_CLASSES,
                    "max_abs_error": error,
                    **metrics,
                    "oracle": oracle_name,
                    "notes": f"{target}; fo wall including process startup={wall:.6e}s; parse timings from stdout records",
                }
            )
            rows.append(record)
    records = parse_fortran(completed.stdout)
    for row in rows:
        # The accepted protocol uses softmax_fit/softmax_predict and
        # mlp_classifier_fit/mlp_classifier_predict exactly.
        protocol_key = (
            f"softmax_{row['phase']}"
            if row["workload"] == "softmax_regression"
            else f"mlp_classifier_{row['phase']}"
        )
        fields = records.get(protocol_key)
        if fields is None:
            row["status"] = "parse_failed"
            row["notes"] = f"missing stdout record {protocol_key}"
        elif fields:
            try:
                row["seconds_per_operation"] = float(fields[-1])
            except ValueError:
                row["status"] = "parse_failed"
                row["notes"] = f"invalid timing in stdout record {protocol_key}"
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--target", default="fortml_bench_classifiers")
    parser.add_argument(
        "--output", type=Path, default=Path("results/classification_models.csv")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    x, labels = fixture()
    details = metadata(root, fortml)
    rows = run_numpy_reference(x, labels, details)
    rows.extend(run_sklearn(x, labels, details))
    rows.extend(run_torch(x, labels, details))
    rows.extend(run_fortml(fortml, root, details, args.target))
    fields = [
        "workload",
        "phase",
        "backend",
        "device",
        "status",
        "n_samples",
        "n_features",
        "n_hidden",
        "n_classes",
        "seconds_per_operation",
        "accuracy",
        "log_loss",
        "probability_normalization_error",
        "max_abs_error",
        "oracle",
        "compiler",
        "flags",
        "python_version",
        "numpy_version",
        "torch_version",
        "cuda_version",
        "fortml_revision",
        "fortnum_revision",
        "benchmark_revision",
        "notes",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
