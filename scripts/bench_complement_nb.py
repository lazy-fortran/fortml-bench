#!/usr/bin/env python3
"""Benchmark the differentiable FortML Complement Naive Bayes contract.

The NumPy path is an independent behavioral oracle.  It reconstructs the
weighted complement counts, smoothed complement probabilities, positive
ComplementNB weights, stable softmax, and input JVP without importing FortML.
The scikit-learn row is contextual: its multiclass implementation deliberately
omits the class-prior intercept, while FortML retains it.  A release app is
accepted only after complete output arrays agree with this oracle; an absent
target remains an explicit ``unavailable`` record.
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
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_classes", "repetitions", "seconds_per_operation",
    "accuracy", "log_loss", "probability_normalization_error",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "sklearn_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
)


def revision(repository: Path, ignored_paths: tuple[Path, ...] = ()) -> str:
    value = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    ).splitlines()
    ignored = {path.resolve() for path in ignored_paths}
    dirty = []
    for line in status:
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


def metadata(root: Path, fortml: Path, ignored: tuple[Path, ...] = ()) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": package_version("sklearn"),
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
        "device": "cpu",
    }


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(1, N_SAMPLES + 1, dtype=np.int64)[:, None]
    columns = np.arange(1, N_FEATURES + 1, dtype=np.int64)[None, :]
    value = (3 * rows + 5 * columns + rows * columns) % 11
    x = 0.15 + 0.08 * value.astype(np.float64)
    x_dot = 0.02 * np.cos(0.011 * rows + 0.09 * columns)
    phase = rows[:, 0].astype(np.float64)
    scores = np.column_stack(
        (
            0.9 * x[:, 0] - 0.4 * x[:, 1] + 0.2 * x[:, 2]
            + 0.8 * np.sin(0.07 * phase),
            0.8 * x[:, 3] + 0.5 * x[:, 4] - 0.3 * x[:, 5]
            + 0.8 * np.sin(0.07 * phase + 2.0944),
            0.7 * x[:, 6] - 0.6 * x[:, 7]
            + 0.8 * np.sin(0.07 * phase + 4.1888),
        )
    )
    labels = CLASS_LABELS[np.argmax(scores, axis=1)]
    return x, labels, x_dot


def fit_oracle(x: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.array(
        [np.count_nonzero(labels == label) for label in CLASS_LABELS], dtype=np.float64
    )
    masses = np.vstack([np.sum(x[labels == label], axis=0) for label in CLASS_LABELS])
    complement = masses.sum(axis=0)[None, :] - masses
    q = (complement + ALPHA) / (
        complement.sum(axis=1, keepdims=True) + ALPHA * N_FEATURES
    )
    return counts / counts.sum(), -np.log(q)


def predict_oracle(
    x: np.ndarray, prior: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint = x @ weights.T + np.log(prior)
    shifted = joint - np.max(joint, axis=1, keepdims=True)
    log_prob = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    probability = np.exp(log_prob)
    prediction = CLASS_LABELS[np.argmax(log_prob, axis=1)]
    return log_prob, probability, prediction


def jvp_oracle(x_dot: np.ndarray, probability: np.ndarray, weights: np.ndarray) -> np.ndarray:
    joint_dot = x_dot @ weights.T
    return joint_dot - np.sum(probability * joint_dot, axis=1, keepdims=True)


def metrics(labels: np.ndarray, prediction: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    if prediction.shape != labels.shape or probability.shape != (N_SAMPLES, CLASS_LABELS.size):
        raise RuntimeError("ComplementNB output shape does not match fixture")
    if not np.isfinite(probability).all() or np.any(probability < 0.0):
        raise RuntimeError("ComplementNB probabilities are not finite/nonnegative")
    normalization = float(np.max(np.abs(probability.sum(axis=1) - 1.0)))
    if normalization > 2.0e-14:
        raise RuntimeError(f"ComplementNB probability normalization error {normalization:.3e}")
    selected = probability[np.arange(labels.size), np.searchsorted(CLASS_LABELS, labels)]
    return {
        "accuracy": float(np.mean(prediction == labels)),
        "log_loss": float(-np.mean(np.log(np.maximum(selected, 1.0e-300)))),
        "probability_normalization_error": normalization,
    }


def base_row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = dict(details)
    row.update({
        "workload": "complement_naive_bayes", "phase": "", "backend": "",
        "status": "", "n_samples": N_SAMPLES, "n_features": N_FEATURES,
        "n_classes": CLASS_LABELS.size, "repetitions": "",
        "seconds_per_operation": "", "accuracy": "", "log_loss": "",
        "probability_normalization_error": "", "max_abs_error": "",
        "oracle": "", "notes": "",
    })
    row.update(values)
    return row


def run_numpy(x: np.ndarray, labels: np.ndarray, x_dot: np.ndarray,
              details: dict[str, str]) -> list[dict[str, Any]]:
    started = time.perf_counter()
    for _ in range(FIT_REPETITIONS):
        prior, weights = fit_oracle(x, labels)
    fit_seconds = (time.perf_counter() - started) / FIT_REPETITIONS
    prior, weights = fit_oracle(x, labels)
    started = time.perf_counter()
    for _ in range(PREDICT_REPETITIONS):
        log_probability, probability, prediction = predict_oracle(x, prior, weights)
    predict_seconds = (time.perf_counter() - started) / PREDICT_REPETITIONS
    started = time.perf_counter()
    for _ in range(JVP_REPETITIONS):
        log_probability_dot = jvp_oracle(x_dot, probability, weights)
    jvp_seconds = (time.perf_counter() - started) / JVP_REPETITIONS
    checked = metrics(labels, prediction, probability)
    if not np.isfinite(log_probability_dot).all():
        raise RuntimeError("NumPy ComplementNB JVP is not finite")
    records = []
    for phase, seconds, repetitions, note in (
        ("fit", fit_seconds, FIT_REPETITIONS, f"alpha={ALPHA:g}; prior sum={prior.sum():.16g}"),
        ("predict", predict_seconds, PREDICT_REPETITIONS, "positive -log complement weights; stable log-softmax"),
        ("jvp", jvp_seconds, JVP_REPETITIONS, "analytic input directional derivative"),
    ):
        records.append(base_row(
            details, phase=phase, backend="numpy_oracle", status="pass",
            repetitions=repetitions, seconds_per_operation=seconds, **checked,
            max_abs_error=0.0,
            oracle="independent NumPy complement counts/log-softmax/JVP", notes=note,
        ))
    return records


def run_sklearn(x: np.ndarray, labels: np.ndarray, details: dict[str, str]) -> list[dict[str, Any]]:
    try:
        from sklearn.naive_bayes import ComplementNB
    except ImportError as error:
        return [base_row(details, phase="fit", backend="sklearn", status="unavailable",
                         oracle="optional scikit-learn context", notes=f"optional dependency missing: {error}")]
    model = ComplementNB(alpha=ALPHA, norm=False, fit_prior=True)
    started = time.perf_counter()
    model.fit(x, labels)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    probability = model.predict_proba(x)
    prediction = model.predict(x)
    predict_seconds = time.perf_counter() - started
    checked = metrics(labels, prediction, probability)
    # ComplementNB's multiclass _joint_log_likelihood intentionally omits the
    # class-prior intercept.  Keep this contextual timing, but do not claim it
    # is bitwise equivalent to FortML's documented prior-inclusive contract.
    notes = "ComplementNB(alpha=0.75,norm=False); sklearn multiclass score omits class prior"
    rows = []
    for phase, seconds in (("fit", fit_seconds), ("predict", predict_seconds)):
        rows.append(base_row(details, phase=phase, backend="sklearn", status="pass",
                             repetitions=1, seconds_per_operation=seconds, **checked,
                             max_abs_error="", oracle="optional scikit-learn contextual row",
                             notes=notes))
    rows.append(base_row(details, phase="jvp", backend="sklearn", status="unavailable",
                         **checked, oracle="independent NumPy input JVP",
                         notes="scikit-learn exposes no differentiable input-JVP API"))
    return rows


def unavailable_rows(details: dict[str, str], note: str) -> list[dict[str, Any]]:
    return [base_row(details, phase=phase, backend="fortml", status="unavailable",
                     oracle="FortML release-app protocol", notes=note)
            for phase in ("fit", "predict", "jvp")]


def run_fortml(fortml: Path, target: str, x: np.ndarray, labels: np.ndarray,
               x_dot: np.ndarray, details: dict[str, str]) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                           env=environment, capture_output=True, text=True)
    if build.returncode != 0:
        note = build.stderr.strip().splitlines()[-1] if build.stderr.strip() else "fo build failed"
        return unavailable_rows(details, f"{target}: build unavailable: {note}")
    prior, weights = fit_oracle(x, labels)
    expected_log, expected_probability, expected_prediction = predict_oracle(x, prior, weights)
    expected_jvp = jvp_oracle(x_dot, expected_probability, weights)
    with tempfile.TemporaryDirectory(dir="/mnt/storage") as directory:
        oracle_path = Path(directory) / "complement_nb_oracle.csv"
        run_environment = environment.copy()
        run_environment["FORTML_BENCH_COMPLEMENT_NB_ORACLE"] = str(oracle_path)
        completed = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                                   env=run_environment, capture_output=True, text=True)
        if completed.returncode != 0:
            stderr = completed.stderr.strip().splitlines()
            return unavailable_rows(details, f"{target}: {stderr[-1] if stderr else 'execution unavailable'}")
        if not oracle_path.is_file():
            return unavailable_rows(details, f"{target}: no benchmark oracle was written")
        actual_log = np.full_like(expected_log, np.nan)
        actual_probability = np.full_like(expected_probability, np.nan)
        actual_jvp = np.full_like(expected_jvp, np.nan)
        actual_prediction = np.full(labels.shape, np.iinfo(np.int64).min)
        with oracle_path.open(newline="") as stream:
            for record in csv.DictReader(stream):
                row = int(record["row"]) - 1
                column = int(record.get("column", "1")) - 1
                if not 0 <= row < N_SAMPLES or not 0 <= column < CLASS_LABELS.size:
                    raise RuntimeError("FortML ComplementNB oracle index out of range")
                quantity = record["quantity"]
                value = float(record["value"])
                if quantity == "log_probability": actual_log[row, column] = value
                elif quantity == "probability": actual_probability[row, column] = value
                elif quantity == "log_probability_jvp": actual_jvp[row, column] = value
                elif quantity == "prediction": actual_prediction[row] = int(value)
                else: raise RuntimeError(f"unknown FortML ComplementNB quantity {quantity!r}")
        if not np.isfinite(actual_log).all() or not np.isfinite(actual_probability).all():
            raise RuntimeError("FortML ComplementNB log/probability oracle is incomplete")
        if not np.isfinite(actual_jvp).all() or np.any(actual_prediction == np.iinfo(np.int64).min):
            raise RuntimeError("FortML ComplementNB JVP/prediction oracle is incomplete")
        errors = {
            "fit": 0.0,
            "predict": max(float(np.max(np.abs(actual_log - expected_log))),
                           float(np.max(np.abs(actual_probability - expected_probability))),
                           float(np.max(actual_prediction != expected_prediction))),
            "jvp": float(np.max(np.abs(actual_jvp - expected_jvp))),
        }
        if errors["predict"] > 2.0e-10 or errors["jvp"] > 2.0e-10:
            raise RuntimeError(f"FortML ComplementNB oracle mismatch: {errors}")
        records: dict[str, list[str]] = {}
        pattern = re.compile(r"^(complement_nb_(?:fit|predict|jvp)),(.*)$")
        for line in completed.stdout.splitlines():
            match = pattern.match(line.strip())
            if match: records[match.group(1)] = [part.strip() for part in match.group(2).split(",")]
        checked = metrics(labels, actual_prediction, actual_probability)
        rows = []
        for phase in ("fit", "predict", "jvp"):
            fields = records.get(f"complement_nb_{phase}")
            rows.append(base_row(details, phase=phase, backend="fortml",
                                 status="pass" if fields else "parse_failed",
                                 seconds_per_operation=float(fields[-1]) if fields else "",
                                 **checked, max_abs_error=errors[phase],
                                 oracle="independent NumPy complement counts/log-softmax/JVP",
                                 notes=f"{target}; complete output-array check"))
        return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--target", default="fortml_bench_complement_nb")
    parser.add_argument("--output", type=Path, default=Path("results/complement_naive_bayes.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    details = metadata(root, args.fortml.resolve(), (output,))
    x, labels, x_dot = fixture()
    prior, weights = fit_oracle(x, labels)
    if np.unique(labels).size != CLASS_LABELS.size or not np.isfinite(weights).all():
        raise RuntimeError("invalid ComplementNB fixture/oracle")
    rows = run_numpy(x, labels, x_dot, details)
    rows.extend(run_sklearn(x, labels, details))
    rows.extend(run_fortml(args.fortml.resolve(), args.target, x, labels, x_dot, details))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
