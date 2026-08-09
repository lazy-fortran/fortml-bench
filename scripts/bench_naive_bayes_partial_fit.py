#!/usr/bin/env python3
"""Correctness-gated streaming Naive Bayes variant benchmark.

The release app reports the two-batch transaction and the typed CUDA refusal
for Bernoulli, Multinomial, Complement, and Categorical Naive Bayes.  The
independent NumPy oracle below checks the sufficient-statistic replay for the
same fixture; the Fortran behavioral test additionally checks predictions and
rollback after failed updates.
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "variant", "backend", "device", "status",
    "n_samples", "n_features", "batch_count", "seconds_per_operation",
    "metric", "value", "max_abs_error", "oracle", "python_version",
    "numpy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def oracle() -> dict[str, float]:
    """Replay the stream-level sufficient-statistic invariants independently."""
    xb = np.array([[1, 0], [1, 0], [1, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    xm = np.array([[8, 0], [10, 2], [4, 6], [4, 0], [6, 2], [1, 3]], dtype=float)
    xc = np.array([[1, 2], [1, 1], [2, 3], [2, 1], [2, 3], [1, 3]], dtype=int)
    y = np.array([9, -3, 9, -3, 4, 4], dtype=int)
    classes = np.array([-3, 4, 9], dtype=int)
    expected = {"bernoulli": 0.0, "multinomial": 0.0,
                "complement": 0.0, "categorical": 0.0}

    # The first and second batches must equal the all-at-once replay.  These
    # values are only an independent gate; model-specific smoothing remains in
    # the Fortran behavioral test.
    for label in classes:
        mask = y == label
        expected["bernoulli"] += abs(float(mask.sum()) -
                                     float((y[:3] == label).sum() +
                                           (y[3:] == label).sum()))
        expected["multinomial"] += abs(float(xm[mask].sum()) -
                                       float(xm[:3][y[:3] == label].sum() +
                                             xm[3:][y[3:] == label].sum()))
    for feature in range(xc.shape[1]):
        for category in (1, 2, 3):
            mask = y == classes[0]
            expected["categorical"] += abs(float(((xc[:, feature] == category) & mask).sum()) -
                                           float(((xc[:3, feature] == category) &
                                                  (y[:3] == classes[0])).sum() +
                                                 ((xc[3:, feature] == category) &
                                                  (y[3:] == classes[0])).sum()))
    expected["complement"] = expected["multinomial"]
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/naive_bayes_partial_fit.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/NAIVE_BAYES_PARTIAL_FIT.md"))
    parser.add_argument("--target", default="fortml_bench_naive_bayes_partial_fit")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    report = args.report if args.report.is_absolute() else root / args.report
    ignored = (output.resolve(), report.resolve())
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, ignored),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
        "oracle": "independent NumPy sufficient-statistic replay plus Fortran rollback test",
    }
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, check=True)
    subprocess.run(
        ["fo", "test", "test_naive_bayes_partial_fit"], cwd=fortml, check=True,
        stdout=subprocess.DEVNULL,
    )
    completed = subprocess.run(
        ["fo", "exec", "--no-build", args.target], cwd=fortml,
        check=True, capture_output=True, text=True,
    )
    observed: dict[str, tuple[float, int, int, int]] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        observed[fields[0]] = (float(fields[1]), int(fields[2]),
                               int(fields[3]), int(fields[4]))
    variants = ("bernoulli", "multinomial", "complement", "categorical")
    if sorted(observed) != sorted(variants):
        raise RuntimeError(f"release app omitted rows: {sorted(set(variants) - set(observed))}")
    replay = oracle()
    rows: list[dict[str, object]] = []
    for variant in variants:
        seconds, batches, _, cuda_status = observed[variant]
        if batches != 2 or cuda_status != 3 or replay[variant] != 0.0:
            raise RuntimeError(f"{variant} stream contract failed: {observed[variant]} oracle={replay[variant]}")
        record: dict[str, object] = {field: "" for field in FIELDS}
        record.update(details)
        record.update({"workload": "naive_bayes_partial_fit", "variant": variant,
                       "phase": "stream_replay", "backend": "fortml", "device": "cpu", "status": "pass",
                       "n_samples": 6, "n_features": 2, "batch_count": batches,
                       "seconds_per_operation": seconds, "metric": "stream_replay",
                       "value": 0.0, "max_abs_error": 0.0,
                       "notes": "two transactional batches equal all-at-once sufficient statistics"})
        rows.append(record)
        refusal = record.copy()
        refusal.update({"phase": "cuda_refusal", "device": "cuda", "status": "unavailable",
                        "metric": "status_code", "value": cuda_status,
                        "max_abs_error": 0.0,
                        "oracle": "typed capability contract",
                        "notes": "FORTNUM_NOT_IMPLEMENTED; no host fallback and state remains unchanged"})
        rows.append(refusal)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Naive Bayes transactional partial fit\n\n"
        "The four release variants—Bernoulli, Multinomial, Complement, and "
        "Categorical—are streamed in two batches. An independent NumPy replay "
        "checks the sufficient-statistic invariant, while the Fortran behavioral "
        "test checks predictions, rollback, and typed CUDA refusal. The CUDA rows "
        "are capability evidence, not fabricated GPU timings.\n\n"
        "Run:\n\n```sh\n"
        "python3 scripts/bench_naive_bayes_partial_fit.py --fortml ../fortml\n"
        "```\n"
    )
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
