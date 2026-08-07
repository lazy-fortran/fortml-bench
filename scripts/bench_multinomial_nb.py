#!/usr/bin/env python3
"""Benchmark differentiable Multinomial Naive Bayes.

The NumPy path is an independent behavioral oracle: it rebuilds weighted class
counts, token-mass smoothing, stable log-softmax probabilities, and the input
directional derivative.  The optional scikit-learn path is contextual.  The
FortML release app is accepted only after complete log-probability,
probability, prediction, and JVP arrays agree with that oracle; missing build
targets are retained as explicit ``unavailable`` rows.
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
from pathlib import Path
from typing import Any

import numpy as np


N_SAMPLES = 512
N_FEATURES = 8
CLASS_LABELS = np.array([-7, 3, 11], dtype=np.int64)
ALPHA = 0.75
FIT_REPETITIONS = 16
PREDICT_REPETITIONS = 128
JVP_REPETITIONS = 128

FIELDS = (
    "workload",
    "phase",
    "backend",
    "device",
    "status",
    "n_samples",
    "n_features",
    "n_classes",
    "repetitions",
    "seconds_per_operation",
    "accuracy",
    "log_loss",
    "probability_normalization_error",
    "max_abs_error",
    "oracle",
    "python_version",
    "numpy_version",
    "sklearn_version",
    "fortml_revision",
    "benchmark_revision",
    "compiler",
    "flags",
    "notes",
)


def revision(repository: Path, ignored_paths: tuple[Path, ...] = ()) -> str:
    """Return the commit and a dirty marker, excluding the output being written."""

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


def metadata(
    root: Path, fortml: Path, ignored_paths: tuple[Path, ...] = ()
) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": package_version("sklearn"),
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored_paths),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
        "device": "cpu",
    }


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the release app's nonnegative count fixture exactly."""

    rows = np.arange(1, N_SAMPLES + 1, dtype=np.int64)[:, None]
    columns = np.arange(1, N_FEATURES + 1, dtype=np.int64)[None, :]
    value = (3 * rows + 5 * columns + rows * columns) % 11
    x = 0.15 + 0.08 * value.astype(np.float64)
    x_dot = 0.02 * np.cos(0.011 * rows + 0.09 * columns)
    phase = rows[:, 0].astype(np.float64)
    scores = np.column_stack(
        (
            0.9 * x[:, 0] - 0.4 * x[:, 1] + 0.2 * x[:, 2] + 0.8 * np.sin(0.07 * phase),
            0.8 * x[:, 3]
            + 0.5 * x[:, 4]
            - 0.3 * x[:, 5]
            + 0.8 * np.sin(0.07 * phase + 2.0944),
            0.7 * x[:, 6] - 0.6 * x[:, 7] + 0.8 * np.sin(0.07 * phase + 4.1888),
        )
    )
    labels = CLASS_LABELS[np.argmax(scores, axis=1)]
    return x, labels, x_dot


def fit_parameters(x: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return class priors and smoothed probabilities (class, feature)."""

    counts = np.array(
        [np.count_nonzero(labels == label) for label in CLASS_LABELS], dtype=np.float64
    )
    prior = counts / float(np.sum(counts))
    feature_mass = np.vstack(
        [np.sum(x[labels == label], axis=0) for label in CLASS_LABELS]
    )
    token_mass = np.sum(feature_mass, axis=1, keepdims=True)
    probabilities = (feature_mass + ALPHA) / (token_mass + ALPHA * float(N_FEATURES))
    return prior, probabilities


def predict_oracle(
    x: np.ndarray, prior: np.ndarray, feature_probability: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_joint = x @ np.log(feature_probability).T + np.log(prior)
    shifted = log_joint - np.max(log_joint, axis=1, keepdims=True)
    log_normalizer = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    log_probabilities = shifted - log_normalizer
    probabilities = np.exp(log_probabilities)
    predicted = CLASS_LABELS[np.argmax(log_probabilities, axis=1)]
    return log_probabilities, probabilities, predicted


def jvp_oracle(
    x_dot: np.ndarray, probabilities: np.ndarray, feature_probability: np.ndarray
) -> np.ndarray:
    joint_dot = x_dot @ np.log(feature_probability).T
    return joint_dot - np.sum(probabilities * joint_dot, axis=1, keepdims=True)


def checked_metrics(
    labels: np.ndarray, predicted: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    if predicted.shape != labels.shape:
        raise RuntimeError("prediction shape does not match MultinomialNB fixture")
    if probabilities.shape != (N_SAMPLES, CLASS_LABELS.size):
        raise RuntimeError("probability shape does not match MultinomialNB fixture")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise RuntimeError("MultinomialNB probabilities are not finite/nonnegative")
    normalization_error = float(np.max(np.abs(np.sum(probabilities, axis=1) - 1.0)))
    if normalization_error > 2.0e-14:
        raise RuntimeError(
            f"MultinomialNB probability normalization error {normalization_error:.3e}"
        )
    selected = probabilities[
        np.arange(labels.size), np.searchsorted(CLASS_LABELS, labels)
    ]
    return {
        "accuracy": float(np.mean(predicted == labels)),
        "log_loss": float(-np.mean(np.log(np.maximum(selected, 1.0e-300)))),
        "probability_normalization_error": normalization_error,
    }


def base_row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = dict(details)
    row.update(
        {
            "workload": "multinomial_naive_bayes",
            "phase": "",
            "backend": "",
            "status": "",
            "n_samples": N_SAMPLES,
            "n_features": N_FEATURES,
            "n_classes": CLASS_LABELS.size,
            "repetitions": "",
            "seconds_per_operation": "",
            "accuracy": "",
            "log_loss": "",
            "probability_normalization_error": "",
            "max_abs_error": "",
            "oracle": "",
            "notes": "",
        }
    )
    row.update(values)
    return row


def run_numpy(
    x: np.ndarray, labels: np.ndarray, x_dot: np.ndarray, details: dict[str, str]
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    for _ in range(FIT_REPETITIONS):
        prior, feature_probability = fit_parameters(x, labels)
    fit_seconds = (time.perf_counter() - started) / FIT_REPETITIONS
    prior, feature_probability = fit_parameters(x, labels)
    started = time.perf_counter()
    for _ in range(PREDICT_REPETITIONS):
        log_probabilities, probabilities, predicted = predict_oracle(
            x, prior, feature_probability
        )
    predict_seconds = (time.perf_counter() - started) / PREDICT_REPETITIONS
    started = time.perf_counter()
    for _ in range(JVP_REPETITIONS):
        log_probabilities_dot = jvp_oracle(x_dot, probabilities, feature_probability)
    jvp_seconds = (time.perf_counter() - started) / JVP_REPETITIONS
    metrics = checked_metrics(labels, predicted, probabilities)
    if not np.isfinite(log_probabilities_dot).all():
        raise RuntimeError("NumPy MultinomialNB input JVP is not finite")
    rows: list[dict[str, Any]] = []
    for phase, seconds, repetitions, metric, value, note in (
        (
            "fit",
            fit_seconds,
            FIT_REPETITIONS,
            "class_prior_sum",
            float(np.sum(prior)),
            f"alpha={ALPHA:g}; token masses [{np.min(np.sum(feature_probability, axis=1)):.6g},"
            f"{np.max(np.sum(feature_probability, axis=1)):.6g}]",
        ),
        (
            "predict",
            predict_seconds,
            PREDICT_REPETITIONS,
            "log_probability_sum",
            float(np.sum(log_probabilities)),
            "stable log-softmax over smoothed token likelihoods",
        ),
        (
            "jvp",
            jvp_seconds,
            JVP_REPETITIONS,
            "log_probability_jvp_sum",
            float(np.sum(log_probabilities_dot)),
            "analytic input directional derivative",
        ),
    ):
        rows.append(
            base_row(
                details,
                phase=phase,
                backend="numpy_oracle",
                status="pass",
                repetitions=repetitions,
                seconds_per_operation=seconds,
                **metrics,
                max_abs_error=0.0,
                oracle="independent NumPy smoothed counts/log-softmax/JVP",
                notes=f"metric={metric}:{value:.16g}; {note}",
            )
        )
    return rows


def run_sklearn(
    x: np.ndarray,
    labels: np.ndarray,
    expected: tuple[np.ndarray, np.ndarray],
    details: dict[str, str],
) -> list[dict[str, Any]]:
    try:
        from sklearn.naive_bayes import MultinomialNB
    except ImportError as error:
        return [
            base_row(
                details,
                phase="fit",
                backend="sklearn",
                status="unavailable",
                oracle="optional scikit-learn context",
                notes=f"optional dependency missing: {error}",
            )
        ]
    prior, feature_probability = expected
    model = MultinomialNB(alpha=ALPHA, fit_prior=True)
    started = time.perf_counter()
    model.fit(x, labels)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    probabilities = model.predict_proba(x)
    predicted = model.predict(x)
    predict_seconds = time.perf_counter() - started
    metrics = checked_metrics(labels, predicted, probabilities)
    _, expected_probabilities, expected_prediction = predict_oracle(
        x, prior, feature_probability
    )
    error = max(
        float(np.max(np.abs(probabilities - expected_probabilities))),
        float(np.max(predicted != expected_prediction)),
    )
    if error > 2.0e-12:
        raise RuntimeError(f"scikit-learn MultinomialNB oracle mismatch: {error:.3e}")
    rows: list[dict[str, Any]] = []
    for phase, seconds in (("fit", fit_seconds), ("predict", predict_seconds)):
        rows.append(
            base_row(
                details,
                phase=phase,
                backend="sklearn",
                status="pass",
                repetitions=1,
                seconds_per_operation=seconds,
                **metrics,
                max_abs_error=error,
                oracle="independent NumPy smoothed counts/log-softmax",
                notes="MultinomialNB(alpha=0.75); CPU contextual timing; no sklearn JVP",
            )
        )
    rows.append(
        base_row(
            details,
            phase="jvp",
            backend="sklearn",
            status="unavailable",
            **metrics,
            oracle="independent NumPy input JVP",
            notes="scikit-learn exposes no differentiable input-JVP API",
        )
    )
    return rows


def unavailable_rows(details: dict[str, str], note: str) -> list[dict[str, Any]]:
    return [
        base_row(
            details,
            phase=phase,
            backend="fortml",
            status="unavailable",
            oracle="FortML release-app protocol",
            notes=note,
        )
        for phase in ("fit", "predict", "jvp")
    ]


def parse_fortran(stdout: str) -> dict[str, list[str]]:
    records: dict[str, list[str]] = {}
    pattern = re.compile(r"^(multinomial_nb_(?:fit|predict|jvp)),(.*)$")
    for line in stdout.splitlines():
        match = pattern.match(line.strip())
        if match:
            records[match.group(1)] = [
                part.strip() for part in match.group(2).split(",")
            ]
    return records


def run_fortml(
    fortml: Path,
    target: str,
    x: np.ndarray,
    labels: np.ndarray,
    x_dot: np.ndarray,
    details: dict[str, str],
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
        return unavailable_rows(details, f"{target}: build unavailable: {note}")
    prior, feature_probability = fit_parameters(x, labels)
    expected_log, expected_prob, expected_prediction = predict_oracle(
        x, prior, feature_probability
    )
    expected_jvp = jvp_oracle(x_dot, expected_prob, feature_probability)
    with tempfile.TemporaryDirectory(dir="/mnt/storage") as directory:
        oracle_path = Path(directory) / "multinomial_nb_oracle.csv"
        run_environment = environment.copy()
        run_environment["FORTML_BENCH_MULTINOMIAL_ORACLE"] = str(oracle_path)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", target],
            cwd=fortml,
            env=run_environment,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip().splitlines()
            note = stderr[-1] if stderr else f"{target}: execution unavailable"
            return unavailable_rows(details, f"{target}: {note}")
        if not oracle_path.is_file():
            return unavailable_rows(
                details, f"{target}: no benchmark oracle was written"
            )
        actual_log = np.full_like(expected_log, np.nan)
        actual_prob = np.full_like(expected_prob, np.nan)
        actual_jvp = np.full_like(expected_jvp, np.nan)
        actual_pred = np.full(labels.shape, np.iinfo(np.int64).min)
        with oracle_path.open(newline="") as stream:
            for record in csv.DictReader(stream):
                row = int(record["row"]) - 1
                column = int(record.get("column", "1")) - 1
                quantity = record["quantity"]
                value = float(record["value"])
                if not 0 <= row < N_SAMPLES or not 0 <= column < CLASS_LABELS.size:
                    raise RuntimeError("FortML MultinomialNB oracle index out of range")
                if quantity == "log_probability":
                    actual_log[row, column] = value
                elif quantity == "probability":
                    actual_prob[row, column] = value
                elif quantity == "log_probability_jvp":
                    actual_jvp[row, column] = value
                elif quantity == "prediction":
                    actual_pred[row] = int(value)
                else:
                    raise RuntimeError(
                        f"unknown FortML MultinomialNB quantity {quantity!r}"
                    )
        if not np.isfinite(actual_log).all() or not np.isfinite(actual_prob).all():
            raise RuntimeError(
                "FortML MultinomialNB log/probability oracle is incomplete"
            )
        if not np.isfinite(actual_jvp).all():
            raise RuntimeError("FortML MultinomialNB JVP oracle is incomplete")
        if np.any(actual_pred == np.iinfo(np.int64).min):
            raise RuntimeError("FortML MultinomialNB prediction oracle is incomplete")
        errors = {
            "fit": 0.0,
            "predict": max(
                float(np.max(np.abs(actual_log - expected_log))),
                float(np.max(np.abs(actual_prob - expected_prob))),
                float(np.max(actual_pred != expected_prediction)),
            ),
            "jvp": float(np.max(np.abs(actual_jvp - expected_jvp))),
        }
        if errors["predict"] > 2.0e-10:
            raise RuntimeError(
                f"FortML MultinomialNB probability oracle mismatch: {errors['predict']:.3e}"
            )
        if errors["jvp"] > 2.0e-10:
            raise RuntimeError(
                f"FortML MultinomialNB JVP oracle mismatch: {errors['jvp']:.3e}"
            )
        records = parse_fortran(completed.stdout)
        metrics = checked_metrics(labels, actual_pred, actual_prob)
        rows: list[dict[str, Any]] = []
        for phase in ("fit", "predict", "jvp"):
            key = f"multinomial_nb_{phase}"
            fields = records.get(key)
            status = "pass" if fields else "parse_failed"
            seconds = float(fields[-1]) if fields else ""
            rows.append(
                base_row(
                    details,
                    phase=phase,
                    backend="fortml",
                    status=status,
                    repetitions="",
                    seconds_per_operation=seconds,
                    **metrics,
                    max_abs_error=errors[phase],
                    oracle="independent NumPy smoothed counts/log-softmax/JVP",
                    notes=f"{target}; complete output-array check",
                )
            )
        return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--target", default="fortml_bench_multinomial_nb")
    parser.add_argument(
        "--output", type=Path, default=Path("results/multinomial_naive_bayes.csv")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    details = metadata(root, args.fortml.resolve(), (output,))
    x, labels, x_dot = fixture()
    prior, feature_probability = fit_parameters(x, labels)
    if np.unique(labels).size != CLASS_LABELS.size:
        raise RuntimeError("MultinomialNB fixture omitted a class")
    if not np.allclose(np.sum(feature_probability, axis=1), 1.0):
        raise RuntimeError("smoothed feature probabilities do not normalize")
    rows = run_numpy(x, labels, x_dot, details)
    rows.extend(run_sklearn(x, labels, (prior, feature_probability), details))
    rows.extend(
        run_fortml(args.fortml.resolve(), args.target, x, labels, x_dot, details)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
