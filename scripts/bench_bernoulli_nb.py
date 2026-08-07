#!/usr/bin/env python3
"""Benchmark a relaxed Bernoulli Naive Bayes fixture.

The NumPy implementation is the behavioral oracle.  It evaluates the
smoothed Bernoulli log likelihood directly, including the input JVP, so a
successful process exit is not enough to retain a row.  The optional
scikit-learn row is contextual.  The FortML row follows the release-app
protocol used by the other classifier lanes and is an explicit refusal until
``fortml_bench_bernoulli_nb`` is available.
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
ALPHA = 0.5
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
    """Return interior relaxed Bernoulli features, labels, and an input JVP."""

    rows = np.arange(1, N_SAMPLES + 1, dtype=np.int64)[:, None]
    columns = np.arange(1, N_FEATURES + 1, dtype=np.int64)[None, :]
    bits = ((7 * rows + 11 * columns + rows * columns) % 17 >= 8).astype(np.float64)
    x = 0.1 + 0.8 * bits
    phase = rows[:, 0].astype(np.float64)
    scores = np.column_stack(
        (
            1.7 * x[:, 0] - 0.6 * x[:, 1] + 0.2 * np.sin(0.07 * phase),
            1.4 * x[:, 2] + 0.8 * x[:, 3] - 0.4 * x[:, 4],
            1.2 * x[:, 5] - 0.7 * x[:, 6] + 0.3 * x[:, 7] + 0.1 * np.cos(0.11 * phase),
        )
    )
    labels = CLASS_LABELS[np.argmax(scores, axis=1)]
    x_dot = 0.03 * np.column_stack(
        [
            np.sin(0.013 * phase + 0.17 * float(column))
            for column in range(1, N_FEATURES + 1)
        ]
    )
    return x, labels, x_dot


def fit_parameters(x: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.array(
        [np.count_nonzero(labels == label) for label in CLASS_LABELS], dtype=np.float64
    )
    prior = counts / float(np.sum(counts))
    feature_counts = np.vstack(
        [np.sum(x[labels == label], axis=0) for label in CLASS_LABELS]
    )
    q = (ALPHA + feature_counts) / (2.0 * ALPHA + counts[:, None])
    return prior, q


def predict_oracle(
    x: np.ndarray, prior: np.ndarray, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_q = np.log(q)
    log_one_minus_q = np.log1p(-q)
    joint = x @ log_q.T + (1.0 - x) @ log_one_minus_q.T + np.log(prior)
    shifted = joint - np.max(joint, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    log_probabilities = np.log(probabilities)
    predicted = CLASS_LABELS[np.argmax(probabilities, axis=1)]
    return log_probabilities, probabilities, predicted


def jvp_oracle(
    x_dot: np.ndarray, probabilities: np.ndarray, q: np.ndarray
) -> np.ndarray:
    joint_dot = x_dot @ (np.log(q) - np.log1p(-q)).T
    return joint_dot - np.sum(probabilities * joint_dot, axis=1, keepdims=True)


def checked_metrics(
    labels: np.ndarray, predicted: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    if predicted.shape != labels.shape:
        raise RuntimeError("prediction shape does not match Bernoulli fixture")
    if probabilities.shape != (N_SAMPLES, CLASS_LABELS.size):
        raise RuntimeError("probability shape does not match Bernoulli fixture")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise RuntimeError("Bernoulli probabilities are not finite/nonnegative")
    normalization_error = float(np.max(np.abs(np.sum(probabilities, axis=1) - 1.0)))
    if normalization_error > 2.0e-14:
        raise RuntimeError(
            f"Bernoulli probability normalization error {normalization_error:.3e}"
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
            "workload": "bernoulli_naive_bayes",
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
        prior, q = fit_parameters(x, labels)
    fit_seconds = (time.perf_counter() - started) / FIT_REPETITIONS
    prior, q = fit_parameters(x, labels)
    started = time.perf_counter()
    for _ in range(PREDICT_REPETITIONS):
        log_probabilities, probabilities, predicted = predict_oracle(x, prior, q)
    predict_seconds = (time.perf_counter() - started) / PREDICT_REPETITIONS
    started = time.perf_counter()
    for _ in range(JVP_REPETITIONS):
        log_probabilities_dot = jvp_oracle(x_dot, probabilities, q)
    jvp_seconds = (time.perf_counter() - started) / JVP_REPETITIONS
    metrics = checked_metrics(labels, predicted, probabilities)
    # Keep this behavioral check independent of the timing loop: the derivative
    # is the closed-form log-softmax JVP, not a copied checksum.
    if not np.isfinite(log_probabilities_dot).all():
        raise RuntimeError("NumPy Bernoulli input JVP is not finite")
    rows: list[dict[str, Any]] = []
    for phase, seconds, repetitions, metric, value, note in (
        (
            "fit",
            fit_seconds,
            FIT_REPETITIONS,
            "class_prior_sum",
            float(np.sum(prior)),
            f"alpha={ALPHA:g}; q range [{np.min(q):.6g},{np.max(q):.6g}]",
        ),
        (
            "predict",
            predict_seconds,
            PREDICT_REPETITIONS,
            "log_probability_sum",
            float(np.sum(log_probabilities)),
            "relaxed x in [0.1,0.9]; stable log-softmax",
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
                oracle="independent NumPy Bernoulli likelihood/log-softmax/JVP",
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
        from sklearn.naive_bayes import BernoulliNB
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
    prior, q = expected
    model = BernoulliNB(alpha=ALPHA, binarize=None, fit_prior=True)
    started = time.perf_counter()
    model.fit(x, labels)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    probabilities = model.predict_proba(x)
    predicted = model.predict(x)
    predict_seconds = time.perf_counter() - started
    metrics = checked_metrics(labels, predicted, probabilities)
    expected_log, expected_prob, expected_prediction = predict_oracle(x, prior, q)
    error = max(
        float(np.max(np.abs(probabilities - expected_prob))),
        float(np.max(predicted != expected_prediction)),
    )
    if error > 2.0e-12:
        raise RuntimeError(f"scikit-learn BernoulliNB oracle mismatch: {error:.3e}")
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
                oracle="independent NumPy Bernoulli likelihood",
                notes="BernoulliNB(alpha=0.5,binarize=None); CPU contextual timing",
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
    pattern = re.compile(r"^(bernoulli_nb_(?:fit|predict|jvp)),(.*)$")
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
    prior, q = fit_parameters(x, labels)
    expected_log, expected_prob, expected_prediction = predict_oracle(x, prior, q)
    with tempfile.TemporaryDirectory() as directory:
        oracle_path = Path(directory) / "bernoulli_nb_oracle.csv"
        run_environment = environment.copy()
        run_environment["FORTML_BENCH_BERNOULLI_ORACLE"] = str(oracle_path)
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
        # The app protocol is deliberately strict: full arrays are required.
        actual_log = np.full_like(expected_log, np.nan)
        actual_prob = np.full_like(expected_prob, np.nan)
        actual_pred = np.full(labels.shape, np.iinfo(np.int64).min)
        with oracle_path.open(newline="") as stream:
            for record in csv.DictReader(stream):
                row = int(record["row"]) - 1
                column = int(record.get("column", "1")) - 1
                quantity = record["quantity"]
                value = float(record["value"])
                if not 0 <= row < N_SAMPLES or not 0 <= column < CLASS_LABELS.size:
                    raise RuntimeError("FortML Bernoulli oracle index out of range")
                if quantity == "log_probability":
                    actual_log[row, column] = value
                elif quantity == "probability":
                    actual_prob[row, column] = value
                elif quantity == "prediction":
                    actual_pred[row] = int(value)
                else:
                    raise RuntimeError(
                        f"unknown FortML Bernoulli quantity {quantity!r}"
                    )
        error = max(
            float(np.nanmax(np.abs(actual_log - expected_log))),
            float(np.nanmax(np.abs(actual_prob - expected_prob))),
            float(np.max(actual_pred != expected_prediction)),
        )
        if not np.isfinite(actual_log).all() or not np.isfinite(actual_prob).all():
            raise RuntimeError("FortML Bernoulli oracle is incomplete")
        if error > 2.0e-10:
            raise RuntimeError(f"FortML Bernoulli oracle mismatch: {error:.3e}")
        records = parse_fortran(completed.stdout)
        metrics = checked_metrics(labels, actual_pred, actual_prob)
        rows: list[dict[str, Any]] = []
        for phase in ("fit", "predict", "jvp"):
            key = f"bernoulli_nb_{phase}"
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
                    max_abs_error=error,
                    oracle="independent NumPy Bernoulli likelihood/log-softmax/JVP",
                    notes=f"{target}; complete output-array check",
                )
            )
        return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--target", default="fortml_bench_bernoulli_nb")
    parser.add_argument(
        "--output", type=Path, default=Path("results/bernoulli_naive_bayes.csv")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    details = metadata(root, args.fortml.resolve(), (output,))
    x, labels, x_dot = fixture()
    prior, q = fit_parameters(x, labels)
    if np.unique(labels).size != CLASS_LABELS.size:
        raise RuntimeError("Bernoulli fixture omitted a class")
    rows = run_numpy(x, labels, x_dot, details)
    rows.extend(run_sklearn(x, labels, (prior, q), details))
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
