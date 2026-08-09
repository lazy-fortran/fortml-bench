#!/usr/bin/env python3
"""Independent exact-GP evidence-gradient/HVP benchmark for ordinal GPs.

The NumPy oracle assembles the latent rank-target covariance from scratch and
uses a dense Cholesky solve.  It central-differences the independently coded
likelihood gradient for the directional HVP, then gates the Fortran release
application and its typed CUDA refusal before retaining timing rows.
"""

from __future__ import annotations

import argparse
import csv
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
X = np.linspace(-1.7, 1.7, 18, dtype=np.float64)
LABELS = np.where(X < -0.55, -4, np.where(X < 0.55, 7, 19)).astype(np.int64)
TARGETS = np.searchsorted(np.unique(LABELS), LABELS).astype(np.float64) + 1.0
THETA = np.log(np.array([1.35, 0.79, 0.05], dtype=np.float64))
DIRECTION = np.array([0.07, -0.04, 0.03], dtype=np.float64)
JITTER = 1.0e-8


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    """Return HEAD and mark unrelated working-tree edits as dirty."""
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


def value_gradient(theta: np.ndarray) -> tuple[float, np.ndarray]:
    """Independent latent rank-target RBF evidence and log-coordinate gradient."""
    variance, lengthscale, noise = np.exp(theta)
    distance = X[:, None] - X[None, :]
    squared = distance * distance
    signal = variance * np.exp(-0.5 * squared / (lengthscale * lengthscale))
    covariance = signal + (noise + JITTER) * np.eye(X.size)
    factor = np.linalg.cholesky(covariance)
    alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, TARGETS))
    inverse = np.linalg.solve(factor.T, np.linalg.solve(factor, np.eye(X.size)))
    logdet = 2.0 * np.sum(np.log(np.diag(factor)))
    value = (-0.5 * TARGETS @ alpha - 0.5 * logdet -
             0.5 * X.size * np.log(2.0 * np.pi))
    blocks = (
        signal,
        signal * squared / (lengthscale * lengthscale),
        noise * np.eye(X.size),
    )
    matrix_bar = 0.5 * (np.outer(alpha, alpha) - inverse)
    gradient = np.array([np.sum(matrix_bar * block) for block in blocks])
    return float(value), gradient


def oracle() -> tuple[float, float, float, float]:
    """Check the analytic gradient and independently finite-difference HVP."""
    value, gradient = value_gradient(THETA)
    step = 2.0e-5
    value_plus, gradient_plus = value_gradient(THETA + step * DIRECTION)
    value_minus, gradient_minus = value_gradient(THETA - step * DIRECTION)
    scalar_fd = (value_plus - value_minus) / (2.0 * step)
    gradient_fd = np.empty_like(gradient)
    for index in range(gradient.size):
        probe = np.zeros_like(THETA)
        probe[index] = step
        plus, _ = value_gradient(THETA + probe)
        minus, _ = value_gradient(THETA - probe)
        gradient_fd[index] = (plus - minus) / (2.0 * step)
    hvp = (gradient_plus - gradient_minus) / (2.0 * step)
    gradient_error = float(np.max(np.abs(gradient - gradient_fd)))
    directional_error = float(abs(np.dot(gradient, DIRECTION) - scalar_fd))
    if gradient_error > 2.0e-8 or directional_error > 2.0e-8:
        raise RuntimeError(
            f"ordinal evidence oracle failed: gradient={gradient_error:.3e}, "
            f"directional={directional_error:.3e}")
    return value, float(np.linalg.norm(gradient)), float(np.linalg.norm(hvp)), gradient_error


def row(details: dict[str, object], **values: object) -> dict[str, object]:
    result: dict[str, object] = {field: "" for field in FIELDS}
    result.update(details)
    result.update({"workload": "gp_ordinal_hyperparameters", "device": "cpu",
                   "n_samples": X.size, "n_classes": 3, "repetitions": 32})
    result.update(values)
    return result


def parse_app(stdout: str, details: dict[str, object], rows: list[dict[str, object]],
              oracle_gradient_norm: float, oracle_hvp_norm: float) -> None:
    gradient_pattern = re.compile(
        r"^gp_ordinal_hyperparameter_gradient,cpu,seconds,\s*([0-9Ee+.-]+),"
        r"norm,\s*([0-9Ee+.-]+)$")
    hvp_pattern = re.compile(
        r"^gp_ordinal_hyperparameter_hvp,cpu,seconds,\s*([0-9Ee+.-]+),"
        r"norm,\s*([0-9Ee+.-]+)$")
    training_pattern = re.compile(
        r"^gp_ordinal_hyperparameter_training,seconds,\s*([0-9Ee+.-]+),"
        r"iterations,(\d+),evaluations,(\d+),nll,\s*([0-9Ee+.-]+),"
        r"gradient_norm,\s*([0-9Ee+.-]+)$")
    refusal_hvp = re.compile(r"^gp_ordinal_hyperparameter_hvp,device,cuda,refused,(\d+)$")
    refusal_train = re.compile(
        r"^gp_ordinal_hyperparameter_training,device,cuda,refused,(\d+)$")
    gradient_match = hvp_match = training_match = None
    refusal_codes: list[int] = []
    for line in stdout.splitlines():
        line = line.strip()
        gradient_match = gradient_pattern.match(line) or gradient_match
        hvp_match = hvp_pattern.match(line) or hvp_match
        training_match = training_pattern.match(line) or training_match
        match = refusal_hvp.match(line)
        if match:
            refusal_codes.append(int(match.group(1)))
        match = refusal_train.match(line)
        if match:
            refusal_codes.append(int(match.group(1)))
    if gradient_match is None or hvp_match is None or training_match is None:
        raise RuntimeError(f"release app did not emit all ordinal rows:\n{stdout}")
    gradient_seconds, observed_gradient_norm = map(float, gradient_match.groups())
    hvp_seconds, observed_hvp_norm = map(float, hvp_match.groups())
    training_seconds, iterations, evaluations, nll, final_norm = training_match.groups()
    gradient_error = abs(observed_gradient_norm - oracle_gradient_norm)
    hvp_error = abs(observed_hvp_norm - oracle_hvp_norm)
    if gradient_error > 2.0e-7 or hvp_error > 2.0e-7:
        raise RuntimeError(
            f"ordinal release checksum mismatch: gradient={gradient_error:.3e}, "
            f"hvp={hvp_error:.3e}")
    rows.append(row(details, phase="release_app", status="pass",
                    seconds_per_operation=gradient_seconds, metric="gradient_norm",
                    value=observed_gradient_norm, max_abs_error=gradient_error,
                    oracle="independent dense NumPy exact-GP evidence gradient"))
    rows.append(row(details, phase="release_app", status="pass",
                    seconds_per_operation=hvp_seconds, metric="hvp_norm",
                    value=observed_hvp_norm, max_abs_error=hvp_error,
                    oracle="independent dense NumPy central-difference HVP"))
    rows.append(row(details, phase="release_app", status="pass",
                    seconds_per_operation=float(training_seconds), metric="final_gradient_norm",
                    value=float(final_norm), max_abs_error=0.0,
                    oracle="FortOpt L-BFGS-B convergence gate",
                    notes=f"iterations={iterations}; evaluations={evaluations}; nll={nll}"))
    if refusal_codes != [3, 3]:
        raise RuntimeError(f"unexpected ordinal CUDA refusal codes: {refusal_codes}")
    rows.append(row(details, phase="device_boundary", backend="fortml", device="cuda",
                    status="refused", metric="ordinal_evidence_graph", value="nan",
                    max_abs_error=0.0, oracle="FORTNUM_NOT_IMPLEMENTED",
                    notes="gradient/HVP and optimizer are control-plane CPU paths"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gp_ordinal_hyperparameters.csv"))
    parser.add_argument("--target", default="fortml_bench_gp_ordinal_hyperparameters")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    value, gradient_norm, hvp_norm, gradient_error = oracle()
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"), "flags": "-O3",
    }
    rows: list[dict[str, object]] = [row(
        details, phase="independent_oracle", backend="numpy", status="pass",
        metric="gradient_norm", value=gradient_norm, max_abs_error=gradient_error,
        oracle="independent dense NumPy Cholesky evidence gradient",
        notes=f"lml={value:.16e}; hvp_norm={hvp_norm:.16e}")]
    if args.skip_fortml:
        rows.append(row(details, phase="behavioral_gate", backend="fortml", status="skipped",
                        metric="tests_passed", value="nan",
                        oracle="test_gp_ordinal_classification_hyperparameters",
                        notes="--skip-fortml"))
    else:
        environment = os.environ.copy()
        environment.update({"FO_SCAN_FALLBACK": "regex", "OMP_NUM_THREADS": "1"})
        subprocess.run(["fo", "test", "test_gp_ordinal_classification_hyperparameters"],
                       cwd=fortml, env=environment, check=True)
        rows.append(row(details, phase="behavioral_gate", backend="fortml", status="pass",
                        metric="tests_passed", value=1.0, max_abs_error=0.0,
                        oracle="independent Fortran FD/HVP/optimizer oracle",
                        notes="evidence gradient/HVP, transactional and CUDA boundaries"))
        subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                       env=environment, check=True, capture_output=True, text=True)
        completed = subprocess.run(["fo", "exec", "--no-build", args.target], cwd=fortml,
                                   env=environment, check=True,
                                   capture_output=True, text=True)
        parse_app(completed.stdout, details, rows, gradient_norm, hvp_norm)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
