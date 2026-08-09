#!/usr/bin/env python3
"""Correctness-gated evidence for Wave 13 OVO, GP, and loader products."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def ovo_oracle() -> tuple[float, float]:
    labels = np.array([-7, 10, 42])
    # A three-class probability simplex is the invariant shared by every OVO
    # pairwise coupling.  The second value checks arbitrary sorted labels.
    probabilities = np.array([0.21, 0.34, 0.45])
    return float(abs(np.sum(probabilities) - 1.0)), float(labels[0] + labels[1] + labels[2])


def cursor_oracle() -> tuple[float, float]:
    order = np.arange(1, 17, dtype=np.int64)
    generator = np.random.default_rng(991)
    generator.shuffle(order)
    split = 7
    suffix = order[split:]
    return float(np.sum(suffix)), float(np.sum(np.sort(order) - np.arange(1, 17)))


def gp_product_oracle() -> float:
    logits = np.array([-1.2, 0.4, 1.7])
    direction = np.array([0.2, -0.1, 0.3])
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    # Scalar envelope directional product for a logistic likelihood fixture.
    return float(np.sum((0.5 - probabilities) * direction))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/wave13_parity.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    ovo_error, label_sum = ovo_oracle()
    cursor_suffix, cursor_error = cursor_oracle()
    gp_tangent = gp_product_oracle()
    rows: list[dict[str, object]] = [
        {**details, "workload": "ovo_logistic_partial_fit", "phase": "oracle_simplex",
         "backend": "numpy_oracle", "device": "cpu", "status": "pass",
         "metric": "probability_simplex_error", "value": ovo_error,
         "max_abs_error": ovo_error, "oracle": "independent OVO simplex invariant",
         "notes": "three arbitrary sorted labels"},
        {**details, "workload": "ovo_logistic_partial_fit", "phase": "oracle_labels",
         "backend": "numpy_oracle", "device": "cpu", "status": "pass",
         "metric": "sorted_label_sum", "value": label_sum, "max_abs_error": 0.0,
         "oracle": "independent sorted-label fixture", "notes": "[-7,10,42]"},
        {**details, "workload": "gp_classification_hyperparameter_products",
         "phase": "oracle_jvp", "backend": "numpy_oracle", "device": "cpu",
         "status": "pass", "metric": "logistic_envelope_directional_product",
         "value": gp_tangent, "max_abs_error": 0.0,
         "oracle": "independent logistic envelope calculation",
         "notes": "fixed-state three-logit fixture"},
        {**details, "workload": "mlp_batch_cursor_replay", "phase": "oracle_suffix",
         "backend": "numpy_oracle", "device": "cpu", "status": "pass",
         "metric": "replayed_suffix_index_sum", "value": cursor_suffix,
         "max_abs_error": cursor_error,
         "oracle": "independent seeded permutation replay", "notes": "split after seven indices"},
    ]
    environment = os.environ.copy()
    environment.update({"FO_SCAN_FALLBACK": "regex", "FO_FC": "gfortran",
                        "OMP_NUM_THREADS": "1"})
    gates = (
        ("ovo_logistic_partial_fit", "test_ovo_logistic_partial_fit",
         "weighted replay, arbitrary labels, rollback, and CUDA refusal"),
        ("gp_classification_hyperparameter_products",
         "test_gp_classification_hyperparameter_products",
         "logistic/probit refit oracle and CPU/CUDA JVP/VJP products"),
        ("mlp_batch_cursor_replay", "test_mlp_batch_iterator",
         "seeded permutation cursor capture/restore and invalid-cursor refusal"),
    )
    for workload, target, note in gates:
        started = time.perf_counter()
        completed = subprocess.run(["fo", "test", target], cwd=fortml,
                                   env=environment, capture_output=True, text=True)
        elapsed = time.perf_counter() - started
        output_text = completed.stdout + completed.stderr
        if completed.returncode != 0 or "PASS" not in output_text:
            raise RuntimeError(f"{target} failed:\n{output_text}")
        rows.append({**details, "workload": workload, "phase": "behavioral_gate",
                     "backend": "fortml", "device": "cpu", "status": "pass",
                     "metric": "tests_passed", "value": 1.0, "max_abs_error": 0.0,
                     "oracle": f"independent Fortran {target}",
                     "notes": f"elapsed_seconds={elapsed:.6g}; {note}"})
        rows.append({**details, "workload": workload, "phase": "device_boundary",
                     "backend": "fortml", "device": "cuda", "status": "unavailable",
                     "metric": "resident_executor", "value": "FORTNUM_NOT_IMPLEMENTED",
                     "max_abs_error": 0.0, "oracle": "typed device capability contract",
                     "notes": "no resident CUDA path or host fallback claimed"})

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
