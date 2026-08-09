#!/usr/bin/env python3
"""Benchmark native ordered-logit/probit value and derivative products.

The independent oracle uses NumPy CDF/PDF formulas and central-differences its
gradient only for the HVP check.  The release application is compared against
that oracle before timing rows are retained; CUDA is recorded as an explicit
unsupported capability rather than a host fallback.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import re
import subprocess
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_classes", "repetitions", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "fortml_revision", "benchmark_revision", "compiler", "flags", "notes",
)
ETA = np.array([-1.2, -0.15, 0.35, 1.1, 0.2], dtype=np.float64)
ETA_DOT = np.array([0.17, -0.11, 0.08, 0.13, -0.09], dtype=np.float64)
THRESHOLDS = np.array([-0.45, 0.8], dtype=np.float64)
THRESHOLDS_DOT = np.array([0.06, -0.04], dtype=np.float64)
LABELS = np.array([1, 2, 3, 2, 1], dtype=np.int64)
VALUE_BAR = 1.7


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_paths = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_paths:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def cdf_pdf(z: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    if kind == "logistic":
        cdf = np.where(z >= 0.0, 1.0 / (1.0 + np.exp(-z)),
                       np.exp(z) / (1.0 + np.exp(z)))
        return cdf, cdf * (1.0 - cdf)
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return cdf, pdf


def value_gradient(eta: np.ndarray, thresholds: np.ndarray, kind: str) -> tuple[float, np.ndarray]:
    gradient_eta = np.zeros(eta.size, dtype=np.float64)
    gradient_thresholds = np.zeros(thresholds.size, dtype=np.float64)
    total = 0.0
    for i, label in enumerate(LABELS):
        upper = 1.0
        upper_pdf = 0.0
        if label <= thresholds.size:
            upper_cdf, upper_pdf_array = cdf_pdf(
                np.array([thresholds[label - 1] - eta[i]]), kind)
            upper = float(upper_cdf[0])
            upper_pdf = float(upper_pdf_array[0])
        lower = 0.0
        lower_pdf = 0.0
        if label > 1:
            lower_cdf, lower_pdf_array = cdf_pdf(
                np.array([thresholds[label - 2] - eta[i]]), kind)
            lower = float(lower_cdf[0])
            lower_pdf = float(lower_pdf_array[0])
        probability = upper - lower
        total += np.log(probability)
        gradient_eta[i] = (lower_pdf - upper_pdf) / probability
        if label <= thresholds.size:
            gradient_thresholds[label - 1] += upper_pdf / probability
        if label > 1:
            gradient_thresholds[label - 2] += -lower_pdf / probability
    return float(total), np.concatenate((gradient_eta, gradient_thresholds))


def oracle(kind: str) -> dict[str, float]:
    value, gradient = value_gradient(ETA, THRESHOLDS, kind)
    _, gradient_plus = value_gradient(ETA + 2.0e-6 * ETA_DOT,
                                      THRESHOLDS + 2.0e-6 * THRESHOLDS_DOT, kind)
    _, gradient_minus = value_gradient(ETA - 2.0e-6 * ETA_DOT,
                                       THRESHOLDS - 2.0e-6 * THRESHOLDS_DOT, kind)
    value_plus, _ = value_gradient(ETA + 2.0e-6 * ETA_DOT,
                                   THRESHOLDS + 2.0e-6 * THRESHOLDS_DOT, kind)
    value_minus, _ = value_gradient(ETA - 2.0e-6 * ETA_DOT,
                                    THRESHOLDS - 2.0e-6 * THRESHOLDS_DOT, kind)
    value_dot = float(np.dot(gradient, np.concatenate((ETA_DOT, THRESHOLDS_DOT))))
    hvp = VALUE_BAR * (gradient_plus - gradient_minus) / (4.0e-6)
    return {
        "value": value,
        "value_dot": value_dot,
        "eta_bar_norm": VALUE_BAR * float(np.linalg.norm(gradient)),
        "hvp_norm": float(np.linalg.norm(hvp)),
        "jvp_fd": (value_plus - value_minus) / (4.0e-6),
    }


def row(details: dict[str, object], **values: object) -> dict[str, object]:
    result: dict[str, object] = {field: "" for field in FIELDS}
    result.update(details)
    result.update({"workload": "gp_ordinal_likelihood", "device": "cpu",
                   "n_samples": ETA.size, "n_classes": THRESHOLDS.size + 1,
                   "repetitions": 100000})
    result.update(values)
    return result


def parse_app(stdout: str, details: dict[str, object], rows: list[dict[str, object]],
              expected: dict[str, dict[str, float]]) -> None:
    pattern = re.compile(
        r"^ordinal_likelihood_(value|jvp|vjp|hvp),(logistic|probit),seconds,"
        r"\s*([0-9Ee+.-]+),(?:value|value_dot|eta_bar_norm|hvp_norm),\s*"
        r"([0-9Ee+.-]+)$")
    observed: dict[tuple[str, str], tuple[float, float]] = {}
    devices: dict[str, str] = {}
    refusal_code: int | None = None
    for line in stdout.splitlines():
        match = pattern.match(line.strip())
        if match:
            metric, kind, seconds, value = match.groups()
            observed[(kind, metric)] = (float(seconds), float(value))
        device_match = re.match(r"^ordinal_likelihood_device,(cpu|cuda),supported,(T|F)$",
                                line.strip())
        if device_match:
            devices[device_match.group(1)] = device_match.group(2)
        refusal_match = re.match(r"^ordinal_likelihood_device,cuda,refused,(\d+)$",
                                 line.strip())
        if refusal_match:
            refusal_code = int(refusal_match.group(1))
    for kind in ("logistic", "probit"):
        for metric in ("value", "jvp", "vjp", "hvp"):
            if (kind, metric) not in observed:
                raise RuntimeError(f"release app omitted {kind} {metric} row\n{stdout}")
            seconds, observed_value = observed[(kind, metric)]
            metric_name = {"value": "value", "jvp": "value_dot",
                           "vjp": "eta_bar_norm", "hvp": "hvp_norm"}[metric]
            error = abs(observed_value - expected[kind][metric_name])
            tolerance = 3.0e-11 if metric in ("value", "jvp") else 3.0e-9
            if error > tolerance:
                raise RuntimeError(f"{kind} {metric} checksum mismatch: {error:.3e}")
            rows.append(row(details, phase="release_app", status="pass",
                            seconds_per_operation=seconds, metric=f"{kind}_{metric}",
                            value=observed_value, max_abs_error=error,
                            oracle="independent NumPy ordered likelihood products"))
    if devices != {"cpu": "T", "cuda": "F"} or refusal_code != 3:
        raise RuntimeError(
            f"unexpected ordinal device capabilities: {devices}, code={refusal_code}")
    rows.append(row(details, phase="device_boundary", backend="fortml", device="cuda",
                    status="refused", metric="likelihood_kernel", value="nan",
                    max_abs_error=0.0, oracle="FORTNUM_NOT_IMPLEMENTED",
                    notes="CUDA resident ordinal likelihood reduction is not linked"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gp_ordinal_likelihood.csv"))
    parser.add_argument("--target", default="fortml_bench_gp_ordinal_likelihood")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    expected = {kind: oracle(kind) for kind in ("logistic", "probit")}
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(
            root, (output, root / "results/GP_ORDINAL_LIKELIHOOD.md")),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    rows: list[dict[str, object]] = []
    for kind, values in expected.items():
        for metric in ("value", "value_dot", "eta_bar_norm", "hvp_norm"):
            rows.append(row(details, phase="independent_oracle", backend="numpy",
                            status="pass", metric=f"{kind}_{metric}", value=values[metric],
                            max_abs_error=0.0,
                            oracle="independent NumPy ordered likelihood formulas"))
    if args.skip_fortml:
        rows.append(row(details, phase="behavioral_gate", backend="fortml", status="skipped",
                        metric="tests_passed", value="nan",
                        oracle="test_gp_ordinal_likelihood", notes="--skip-fortml"))
    else:
        environment = os.environ.copy()
        environment.update({"FO_SCAN_FALLBACK": "regex", "OMP_NUM_THREADS": "1"})
        subprocess.run(["fo", "test", "test_gp_ordinal_likelihood"], cwd=fortml,
                       env=environment, check=True)
        rows.append(row(details, phase="behavioral_gate", backend="fortml", status="pass",
                        metric="tests_passed", value=1.0, max_abs_error=0.0,
                        oracle="independent Fortran value/JVP/VJP/HVP oracle",
                        notes="both ordered-logit and ordered-probit paths"))
        subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                       env=environment, check=True, capture_output=True, text=True)
        completed = subprocess.run(["fo", "exec", "--no-build", args.target], cwd=fortml,
                                   env=environment, check=True,
                                   capture_output=True, text=True)
        parse_app(completed.stdout, details, rows, expected)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
