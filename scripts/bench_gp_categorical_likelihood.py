#!/usr/bin/env python3
"""Correctness-gated benchmark for categorical variational-GP temperature products.

The oracle independently evaluates the variance-corrected softmax and its
log-temperature derivative from latent means/variances emitted by the release
probe.  This isolates the likelihood and reduction algebra from the Fortran
latent-state implementation.  CUDA is a typed refusal row, never a host
fallback timing.
"""

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
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_classes", "n_features", "n_parameters", "seconds_per_operation",
    "metric", "value", "max_abs_error", "oracle", "python_version",
    "numpy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
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


def parse_probe(output: str) -> dict[str, list[list[str]]]:
    records: dict[str, list[list[str]]] = {}
    for line in output.splitlines():
        if not line.startswith("gp_categorical_likelihood_"):
            continue
        fields = next(csv.reader([line]))
        records.setdefault(fields[0], []).append(fields[1:])
    return records


def oracle(means: np.ndarray, variances: np.ndarray, log_scale: float,
           labels: np.ndarray, probabilities_bar: np.ndarray) -> tuple[
               np.ndarray, np.ndarray, float, float, float]:
    correction = np.sqrt(1.0 + np.pi * variances / 8.0)
    logits = np.exp(log_scale) * means / correction
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    logits_dot = logits
    mean_tangent = np.sum(probabilities * logits_dot, axis=1, keepdims=True)
    probability_jvp = probabilities * (logits_dot - mean_tangent)
    probability_bar_mean = np.sum(probabilities * probabilities_bar, axis=1, keepdims=True)
    logits_bar = probabilities * (probabilities_bar - probability_bar_mean)
    probability_bar_mean_dot = np.sum(probability_jvp * probabilities_bar, axis=1, keepdims=True)
    logits_bar_dot = probability_jvp * (probabilities_bar - probability_bar_mean) - \
        probabilities * probability_bar_mean_dot
    probability_hvp = float(np.sum(logits_bar_dot * logits + logits_bar * logits_dot))
    class_indices = np.searchsorted(np.array([10, 20, 30]), labels)
    elbo_gradient = float(np.sum(logits[np.arange(labels.size), class_indices] -
        np.sum(probabilities * logits, axis=1)))
    mean_logit_dot = np.sum(probability_jvp * logits, axis=1) + \
        np.sum(probabilities * logits_dot, axis=1)
    elbo_hvp = float(np.sum(logits_dot[np.arange(labels.size), class_indices] - mean_logit_dot))
    return probabilities, probability_jvp, elbo_gradient, elbo_hvp, probability_hvp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gp_categorical_likelihood.csv"))
    parser.add_argument("--probe", type=Path,
                        help="optional prebuilt FortML release probe")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml, output = args.fortml.resolve(), args.output.resolve()
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)), "compiler": "gfortran",
        "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(details)
        row.update({"workload": "gp_categorical_likelihood", "backend": "fortml",
                    "device": "cpu", "n_samples": 6, "n_classes": 3,
                    "n_features": 1, "n_parameters": 1})
        row.update(values)
        rows.append(row)

    started = time.perf_counter()
    if args.skip_fortml:
        status, records, notes = "skipped", {}, "--skip-fortml"
    else:
        environment = os.environ.copy()
        environment["FO_SCAN_FALLBACK"] = "regex"
        command = ([str(args.probe.resolve())] if args.probe is not None else
                   ["fo", "exec", "--no-build", "fortml_bench_gp_categorical_likelihood"])
        completed = subprocess.run(command, cwd=fortml, env=environment,
                                   check=True, capture_output=True, text=True)
        status, records, notes = "pass", parse_probe(completed.stdout), "release probe"
    elapsed = time.perf_counter() - started
    if status == "pass":
        scale = float(records["gp_categorical_likelihood_scale"][0][0])
        log_scale = float(np.log(scale))
        means = np.zeros((6, 3), dtype=np.float64)
        variances = np.zeros((6, 3), dtype=np.float64)
        observed = np.zeros_like(means)
        observed_jvp = np.zeros_like(means)
        observed_logs = np.zeros_like(means)
        observed_log_parameter_jvp = np.zeros_like(means)
        observed_log_input_jvp = np.zeros_like(means)
        observed_parameter_probability = np.zeros_like(means)
        observed_parameter_probability_jvp = np.zeros_like(means)
        observed_input_probability = np.zeros_like(means)
        observed_input_probability_jvp = np.zeros_like(means)
        for fields in records["gp_categorical_likelihood_latent"]:
            i, j = int(fields[0]) - 1, int(fields[1]) - 1
            means[i, j], variances[i, j] = float(fields[2]), float(fields[3])
        for fields in records["gp_categorical_likelihood_probability"]:
            i, j = int(fields[0]) - 1, int(fields[1]) - 1
            observed[i, j], observed_jvp[i, j] = float(fields[2]), float(fields[3])
        for fields in records["gp_categorical_likelihood_log_probability"]:
            i, j = int(fields[0]) - 1, int(fields[1]) - 1
            observed_logs[i, j] = float(fields[2])
            observed_log_parameter_jvp[i, j] = float(fields[3])
            observed_log_input_jvp[i, j] = float(fields[4])
        for fields in records["gp_categorical_likelihood_parameter_probability"]:
            i, j = int(fields[0]) - 1, int(fields[1]) - 1
            observed_parameter_probability[i, j] = float(fields[2])
            observed_parameter_probability_jvp[i, j] = float(fields[3])
        for fields in records["gp_categorical_likelihood_input_probability"]:
            i, j = int(fields[0]) - 1, int(fields[1]) - 1
            observed_input_probability[i, j] = float(fields[2])
            observed_input_probability_jvp[i, j] = float(fields[3])
        labels = np.array([30, 10, 20, 10, 30, 20], dtype=np.int64)
        probabilities_bar = np.array([
            [0.17, -0.05, 0.02], [-0.09, 0.11, 0.06], [0.04, -0.08, 0.09],
            [0.12, 0.07, -0.04], [-0.06, 0.03, 0.05], [0.08, -0.02, 0.01],
        ], dtype=np.float64)
        expected, expected_jvp, expected_gradient, expected_elbo_hvp, expected_probability_hvp = oracle(
            means, variances, log_scale, labels, probabilities_bar)
        expected_logs = np.log(expected)
        probability_error = float(np.max(np.abs(observed - expected)))
        jvp_error = float(np.max(np.abs(observed_jvp - expected_jvp)))
        log_error = float(np.max(np.abs(observed_logs - expected_logs)))
        log_parameter_identity_error = float(np.max(np.abs(
            observed_log_parameter_jvp - observed_parameter_probability_jvp /
            observed_parameter_probability)))
        log_input_identity_error = float(np.max(np.abs(
            observed_log_input_jvp - observed_input_probability_jvp /
            observed_input_probability)))
        gradient_observed = float(records["gp_categorical_likelihood_gradient"][0][0])
        gradient_error = abs(gradient_observed - expected_gradient)
        tangent = float(records["gp_categorical_likelihood_jvp"][0][0])
        tangent_error = abs(tangent - expected_gradient)
        observed_elbo_hvp = float(records["gp_categorical_likelihood_elbo_hvp"][0][0])
        elbo_hvp_error = abs(observed_elbo_hvp - expected_elbo_hvp)
        observed_probability_hvp = float(records["gp_categorical_likelihood_probability_hvp"][0][0])
        probability_hvp_error = abs(observed_probability_hvp - expected_probability_hvp)
        error = max(probability_error, jvp_error, log_error,
                    log_parameter_identity_error, log_input_identity_error,
                    gradient_error, tangent_error, elbo_hvp_error, probability_hvp_error)
        if error > 3.0e-9:
            raise RuntimeError(f"categorical likelihood oracle mismatch: {error:.3e}")
        iterations = int(records["gp_categorical_likelihood_iterations"][0][0])
        fit_seconds = float(records["gp_categorical_likelihood_fit_seconds"][0][0])
        if tangent_error > 3.0e-9:
            raise RuntimeError(f"categorical ELBO JVP mismatch: {tangent_error:.3e}")
    else:
        probability_error = jvp_error = log_error = error = float("nan")
        log_parameter_identity_error = log_input_identity_error = float("nan")
        gradient_error = float("nan")
        tangent_error = elbo_hvp_error = probability_hvp_error = float("nan")
        iterations = 0
        fit_seconds = tangent = observed_elbo_hvp = observed_probability_hvp = float("nan")
    add(phase="independent_oracle", backend="numpy_oracle", status="pass",
        metric="softmax_temperature_formula", value="nan", max_abs_error=0.0,
        oracle="independent NumPy variance-corrected softmax and log-temperature JVP",
        notes="latent means/variances are read from the probe; likelihood algebra is independent")
    add(phase="fit_likelihood", status=status, seconds_per_operation=fit_seconds,
        metric="fit_seconds", value=fit_seconds, max_abs_error=gradient_error,
        oracle="FortOpt likelihood-only fit versus independent ELBO gradient",
        notes=f"iterations={iterations}; {notes}")
    add(phase="probability_products", status=status, seconds_per_operation=elapsed,
        metric="probability_and_jvp_max_abs", value=error, max_abs_error=error,
        oracle="NumPy variance-corrected softmax and exact log-temperature JVP",
        notes=f"probability_error={probability_error:.3e}; jvp_error={jvp_error:.3e}; "
        f"log_error={log_error:.3e}; log_parameter_identity={log_parameter_identity_error:.3e}; "
        f"log_input_identity={log_input_identity_error:.3e}; "
        f"elbo_jvp_error={tangent_error:.3e}; probability_hvp_error={probability_hvp_error:.3e}; {notes}")
    add(phase="log_probability_products", status=status,
        metric="log_probability_max_abs", value=log_error,
        max_abs_error=log_error,
        oracle="NumPy log of independently reconstructed coupled categorical probabilities",
        notes=f"parameter_identity_error={log_parameter_identity_error:.3e}; "
        f"input_identity_error={log_input_identity_error:.3e}; {notes}")
    add(phase="probability_products", status=status, metric="probability_hvp", value=observed_probability_hvp,
        max_abs_error=probability_hvp_error,
        oracle="NumPy fixed-cotangent probability VJP directional derivative",
        notes=notes)
    add(phase="elbo_products", status=status, metric="elbo_jvp", value=tangent,
        max_abs_error=gradient_error, oracle="NumPy categorical ELBO log-temperature derivative",
        notes=notes)
    add(phase="elbo_products", status=status, metric="elbo_hvp", value=observed_elbo_hvp,
        max_abs_error=elbo_hvp_error,
        oracle="NumPy fixed-state categorical ELBO directional Hessian",
        notes=notes)
    cuda_code = 3
    cuda_probability_hvp_code = 3
    cuda_elbo_hvp_code = 3
    if status == "pass":
        cuda_code = int(records["gp_categorical_likelihood_cuda_jvp"][0][0])
        if cuda_code != 3:
            raise RuntimeError(f"unexpected CUDA status code {cuda_code}")
        cuda_probability_hvp_code = int(records["gp_categorical_likelihood_cuda_probability_hvp"][0][0])
        cuda_elbo_hvp_code = int(records["gp_categorical_likelihood_cuda_elbo_hvp"][0][0])
        cuda_log_code = int(records["gp_categorical_likelihood_cuda_log_probability"][0][0])
        cuda_log_parameter_code = int(records["gp_categorical_likelihood_cuda_log_parameter_vjp"][0][0])
        cuda_log_input_code = int(records["gp_categorical_likelihood_cuda_log_input_vjp"][0][0])
        if (cuda_probability_hvp_code != 3 or cuda_elbo_hvp_code != 3 or
                cuda_log_code != 3 or cuda_log_parameter_code != 3 or cuda_log_input_code != 3):
            raise RuntimeError("unexpected CUDA categorical HVP status")
    else:
        cuda_log_code = cuda_log_parameter_code = cuda_log_input_code = 3
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_categorical_likelihood_graph", value="nan", max_abs_error="nan",
        oracle="typed FORTNUM_NOT_IMPLEMENTED refusal",
        notes=f"JVP status_code={cuda_code}; probability HVP status_code={cuda_probability_hvp_code}; "
        f"ELBO HVP status_code={cuda_elbo_hvp_code}; log probability status_code={cuda_log_code}; "
        f"log parameter VJP status_code={cuda_log_parameter_code}; "
        f"log input VJP status_code={cuda_log_input_code}; no host fallback")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
