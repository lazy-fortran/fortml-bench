#!/usr/bin/env python3
"""Benchmark the integer categorical FortML one-hot encoder.

The dense NumPy implementation is an independent oracle for sorted categories,
packed offsets, ``drop_first``, missing-category handling, and ignored unknowns.
Scikit-learn is a contextual reference.  Integer categories have no canonical
tangent space, so JVP/VJP rows are explicit refusals rather than fake zeros.
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
N_FEATURES = 5
N_QUERY = 128
MISSING_VALUE = -99
FIT_REPETITIONS = 16
TRANSFORM_REPETITIONS = 128

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


def fixture() -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(1, N_SAMPLES + 1, dtype=np.int64)[:, None]
    columns = np.arange(1, N_FEATURES + 1, dtype=np.int64)[None, :]
    train = ((7 * rows + 3 * columns + rows * columns) % (columns + 2)).astype(np.int64)
    # Keep an explicit missing category in every feature; it is part of the
    # fitted sorted category list rather than an all-zero missing value.
    train[(rows[:, 0] % 29) == 0, :] = MISSING_VALUE
    query_rows = np.arange(1, N_QUERY + 1, dtype=np.int64)[:, None]
    query_columns = np.arange(1, N_FEATURES + 1, dtype=np.int64)[None, :]
    query = ((11 * query_rows + 5 * query_columns + query_rows * query_columns) %
             (query_columns + 3)).astype(np.int64)
    query[(query_rows[:, 0] % 17) == 0, :] = MISSING_VALUE
    # 1001 is not present in the training categories and exercises ignore.
    query[::19, 2] = 1001
    return train, query


def fit_oracle(train: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, int]:
    categories = [np.unique(train[:, feature]) for feature in range(N_FEATURES)]
    category_offsets = np.zeros(N_FEATURES + 1, dtype=np.int64)
    output_offsets = np.zeros(N_FEATURES + 1, dtype=np.int64)
    for feature, values in enumerate(categories):
        category_offsets[feature + 1] = category_offsets[feature] + values.size
        output_offsets[feature + 1] = output_offsets[feature] + max(0, values.size - 1)
    return categories, category_offsets, output_offsets, int(output_offsets[-1])


def transform_oracle(query: np.ndarray, categories: list[np.ndarray],
                    output_offsets: np.ndarray, output_count: int) -> np.ndarray:
    result = np.zeros((query.shape[0], output_count), dtype=np.float64)
    for feature, values in enumerate(categories):
        for row, value in enumerate(query[:, feature]):
            matches = np.flatnonzero(values == value)
            if matches.size == 0:
                continue
            index = int(matches[0])
            # drop_first removes the first sorted category from every block.
            if index > 0:
                result[row, output_offsets[feature] + index - 1] = 1.0
    return result


def base_row(details: dict[str, str], **values: Any) -> dict[str, Any]:
    row: dict[str, Any] = dict(details)
    row.update({
        "workload": "one_hot_encoder", "phase": "", "backend": "",
        "status": "", "n_samples": N_SAMPLES, "n_features": N_FEATURES,
        "n_classes": "", "repetitions": "", "seconds_per_operation": "",
        "accuracy": "", "log_loss": "", "probability_normalization_error": "",
        "max_abs_error": "", "oracle": "", "notes": "",
    })
    row.update(values)
    return row


def run_numpy(train: np.ndarray, query: np.ndarray, details: dict[str, str]) -> list[dict[str, Any]]:
    started = time.perf_counter()
    for _ in range(FIT_REPETITIONS):
        categories, category_offsets, output_offsets, output_count = fit_oracle(train)
    fit_seconds = (time.perf_counter() - started) / FIT_REPETITIONS
    started = time.perf_counter()
    for _ in range(TRANSFORM_REPETITIONS):
        transformed = transform_oracle(query, categories, output_offsets, output_count)
    transform_seconds = (time.perf_counter() - started) / TRANSFORM_REPETITIONS
    if not np.isfinite(transformed).all() or not np.all((transformed == 0.0) | (transformed == 1.0)):
        raise RuntimeError("NumPy one-hot output is not finite binary data")
    rows = [base_row(
        details, phase="fit", backend="numpy_oracle", status="pass",
        repetitions=FIT_REPETITIONS, seconds_per_operation=fit_seconds,
        max_abs_error=0.0, oracle="independent NumPy sorted-category/offset oracle",
        notes=(f"sorted categories; handle_unknown=ignore; handle_missing=category; "
               f"drop_first=True; output_count={output_count}"),
    ), base_row(
        details, phase="transform", backend="numpy_oracle", status="pass",
        repetitions=TRANSFORM_REPETITIONS, seconds_per_operation=transform_seconds,
        max_abs_error=0.0, oracle="independent NumPy dense one-hot oracle",
        notes=f"query_rows={N_QUERY}; unknown values map to all-zero blocks",
    ), base_row(
        details, phase="jvp", backend="numpy_oracle", status="refused",
        oracle="categorical derivative contract", notes="integer categories have no canonical tangent space",
    ), base_row(
        details, phase="vjp", backend="numpy_oracle", status="refused",
        oracle="categorical derivative contract", notes="integer categories have no canonical cotangent space",
    )]
    return rows


def run_sklearn(train: np.ndarray, query: np.ndarray, expected: np.ndarray,
                details: dict[str, str]) -> list[dict[str, Any]]:
    try:
        from sklearn.preprocessing import OneHotEncoder
    except ImportError as error:
        return [base_row(details, phase="fit", backend="sklearn", status="unavailable",
                         oracle="optional scikit-learn context", notes=f"optional dependency missing: {error}")]
    try:
        model = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
    except TypeError:  # sklearn < 1.2
        model = OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)
    started = time.perf_counter()
    model.fit(train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    transformed = model.transform(query)
    transform_seconds = time.perf_counter() - started
    if transformed.shape != expected.shape:
        raise RuntimeError(f"scikit-learn one-hot shape {transformed.shape} != {expected.shape}")
    error = float(np.max(np.abs(transformed - expected)))
    if error > 1.0e-14:
        raise RuntimeError(f"scikit-learn one-hot oracle mismatch: {error:.3e}")
    notes = "OneHotEncoder(handle_unknown=ignore,drop=first,sparse_output=False)"
    return [base_row(details, phase="fit", backend="sklearn", status="pass",
                     repetitions=1, seconds_per_operation=fit_seconds,
                     max_abs_error=error, oracle="independent NumPy sorted-category/offset oracle",
                     notes=notes),
            base_row(details, phase="transform", backend="sklearn", status="pass",
                     repetitions=1, seconds_per_operation=transform_seconds,
                     max_abs_error=error, oracle="independent NumPy dense one-hot oracle",
                     notes=notes),
            base_row(details, phase="jvp", backend="sklearn", status="refused",
                     oracle="categorical derivative contract", notes="scikit-learn exposes no categorical JVP/VJP API")]


def unavailable_rows(details: dict[str, str], note: str) -> list[dict[str, Any]]:
    return [base_row(details, phase=phase, backend="fortml", status="unavailable",
                     oracle="FortML release-app protocol", notes=note)
            for phase in ("fit", "transform", "jvp", "vjp")]


def run_fortml(fortml: Path, target: str, train: np.ndarray, query: np.ndarray,
               expected: np.ndarray, categories: list[np.ndarray], category_offsets: np.ndarray,
               output_offsets: np.ndarray, details: dict[str, str]) -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment.update({"FO_FC": environment.get("FO_FC", "gfortran"), "OMP_NUM_THREADS": "1"})
    build = subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                           env=environment, capture_output=True, text=True)
    if build.returncode != 0:
        note = build.stderr.strip().splitlines()[-1] if build.stderr.strip() else "fo build failed"
        return unavailable_rows(details, f"{target}: build unavailable: {note}")
    with tempfile.TemporaryDirectory(dir="/mnt/storage") as directory:
        oracle_path = Path(directory) / "one_hot_oracle.csv"
        run_environment = environment.copy()
        run_environment["FORTML_BENCH_ONE_HOT_ORACLE"] = str(oracle_path)
        completed = subprocess.run(["fo", "exec", "--no-build", target], cwd=fortml,
                                   env=run_environment, capture_output=True, text=True)
        if completed.returncode != 0:
            stderr = completed.stderr.strip().splitlines()
            return unavailable_rows(details, f"{target}: {stderr[-1] if stderr else 'execution unavailable'}")
        if not oracle_path.is_file():
            return unavailable_rows(details, f"{target}: no benchmark oracle was written")
        actual = np.full_like(expected, np.nan)
        actual_categories = np.full(sum(values.size for values in categories), np.iinfo(np.int64).min)
        actual_category_offsets = np.full(category_offsets.shape, np.iinfo(np.int64).min)
        actual_output_offsets = np.full(output_offsets.shape, np.iinfo(np.int64).min)
        with oracle_path.open(newline="") as stream:
            for record in csv.DictReader(stream):
                quantity = record["quantity"]
                row = int(record.get("row", "1")) - 1
                column = int(record.get("column", "1")) - 1
                value = float(record["value"])
                if quantity == "transformed":
                    if not 0 <= row < N_QUERY or not 0 <= column < expected.shape[1]:
                        raise RuntimeError("FortML one-hot transform index out of range")
                    actual[row, column] = value
                elif quantity == "category":
                    if not 0 <= row < actual_categories.size: raise RuntimeError("FortML category index out of range")
                    actual_categories[row] = int(value)
                elif quantity == "category_offset":
                    if not 0 <= row < category_offsets.size: raise RuntimeError("FortML category offset index out of range")
                    actual_category_offsets[row] = int(value)
                elif quantity == "output_offset":
                    if not 0 <= row < output_offsets.size: raise RuntimeError("FortML output offset index out of range")
                    actual_output_offsets[row] = int(value)
                else: raise RuntimeError(f"unknown FortML one-hot quantity {quantity!r}")
        if not np.isfinite(actual).all() or np.any(actual_categories == np.iinfo(np.int64).min):
            raise RuntimeError("FortML one-hot oracle is incomplete")
        if np.any(actual_category_offsets == np.iinfo(np.int64).min) or np.any(actual_output_offsets == np.iinfo(np.int64).min):
            raise RuntimeError("FortML one-hot offsets are incomplete")
        error = max(float(np.max(np.abs(actual - expected))),
                    float(np.max(actual_categories != np.concatenate(categories))),
                    float(np.max(actual_category_offsets != category_offsets)),
                    float(np.max(actual_output_offsets != output_offsets)))
        if error > 1.0e-14:
            raise RuntimeError(f"FortML one-hot oracle mismatch: {error:.3e}")
        records: dict[str, list[str]] = {}
        pattern = re.compile(r"^(one_hot_(?:fit|transform)),(.*)$")
        for line in completed.stdout.splitlines():
            match = pattern.match(line.strip())
            if match: records[match.group(1)] = [part.strip() for part in match.group(2).split(",")]
        rows = []
        for phase in ("fit", "transform"):
            fields = records.get(f"one_hot_{phase}")
            rows.append(base_row(details, phase=phase, backend="fortml",
                                 status="pass" if fields else "parse_failed",
                                 seconds_per_operation=float(fields[-1]) if fields else "",
                                 max_abs_error=error, oracle="independent NumPy sorted-category/offset oracle",
                                 notes=f"{target}; complete category/offset/output check"))
        rows.extend([base_row(details, phase="jvp", backend="fortml", status="refused",
                              oracle="categorical derivative contract", notes="FortML returns FORTNUM_NOT_IMPLEMENTED"),
                     base_row(details, phase="vjp", backend="fortml", status="refused",
                              oracle="categorical derivative contract", notes="FortML returns FORTNUM_NOT_IMPLEMENTED")])
        return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--target", default="fortml_bench_one_hot_encoder")
    parser.add_argument("--output", type=Path, default=Path("results/one_hot_encoder.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    details = metadata(root, args.fortml.resolve(), (output,))
    train, query = fixture()
    categories, category_offsets, output_offsets, output_count = fit_oracle(train)
    expected = transform_oracle(query, categories, output_offsets, output_count)
    rows = run_numpy(train, query, details)
    rows.extend(run_sklearn(train, query, expected, details))
    rows.extend(run_fortml(args.fortml.resolve(), args.target, train, query, expected,
                           categories, category_offsets, output_offsets, details))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
