#!/usr/bin/env python3
"""Independent weighted cut-point oracle for ordinal GP calibration."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import re
import subprocess
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import ndtr


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_classes", "repetitions", "seconds_per_operation", "metric", "value",
    "max_abs_error", "oracle", "python_version", "numpy_version",
    "scipy_version", "fortml_revision", "benchmark_revision", "compiler",
    "flags", "notes",
)
N = 24
X = np.linspace(-1.8, 1.8, N, dtype=np.float64)
LABELS = np.where(X < -0.62, -4, np.where(X < 0.47, 7, 19)).astype(np.int64)
RANKS = np.searchsorted(np.unique(LABELS), LABELS).astype(np.int64)
WEIGHTS = np.array([0.75 + 0.09 * ((5 * index) % 9)
                    for index in range(1, N + 1)], dtype=np.float64)
INITIAL_CUTS = np.array([1.18, 2.78], dtype=np.float64)
DIRECTION = np.array([0.16, -0.11], dtype=np.float64)
MINIMUM_GAP = 1.0e-6


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    """Return HEAD and mark working-tree changes outside generated outputs."""
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


def latent_mean() -> np.ndarray:
    """Assemble the independent dense RBF posterior mean at training rows."""
    distance = X[:, None] - X[None, :]
    signal = 1.3 * np.exp(-0.5 * distance * distance / (0.81 * 0.81))
    covariance = signal + (0.08 + 1.0e-8) * np.eye(N)
    targets = RANKS.astype(np.float64) + 1.0
    alpha = np.linalg.solve(covariance, targets)
    return signal @ alpha


ETA = latent_mean()


def value_gradient(cuts: np.ndarray) -> tuple[float, np.ndarray]:
    """Independent weighted ordered-probit NLL and threshold gradient."""
    upper = np.ones(N, dtype=np.float64)
    lower = np.zeros(N, dtype=np.float64)
    upper_mask = RANKS < cuts.size
    lower_mask = RANKS > 0
    upper[upper_mask] = ndtr(cuts[RANKS[upper_mask]] - ETA[upper_mask])
    lower[lower_mask] = ndtr(cuts[RANKS[lower_mask] - 1] - ETA[lower_mask])
    probability = upper - lower
    if np.any(probability <= 0.0):
        raise RuntimeError("independent ordered-probit probability underflow")
    mass = float(np.sum(WEIGHTS))
    value = float(-np.sum(WEIGHTS * np.log(probability)) / mass)
    gradient = np.zeros(cuts.size, dtype=np.float64)
    normalizer = np.sqrt(2.0 * np.pi)
    for row, rank in enumerate(RANKS):
        scale = WEIGHTS[row] / (mass * probability[row])
        if rank < cuts.size:
            z_upper = cuts[rank] - ETA[row]
            gradient[rank] -= scale * np.exp(-0.5 * z_upper * z_upper) / normalizer
        if rank > 0:
            z_lower = cuts[rank - 1] - ETA[row]
            gradient[rank - 1] += scale * np.exp(-0.5 * z_lower * z_lower) / normalizer
    return value, gradient


def decode(parameters: np.ndarray) -> np.ndarray:
    """Apply the same mathematical cut-point transform from first principles."""
    cuts = np.empty_like(parameters)
    cuts[0] = parameters[0]
    for index in range(1, parameters.size):
        cuts[index] = cuts[index - 1] + MINIMUM_GAP + np.exp(parameters[index])
    return cuts


def transformed(parameters: np.ndarray) -> tuple[float, np.ndarray]:
    """Independent transformed objective used only by SciPy."""
    cuts = decode(parameters)
    value, cut_gradient = value_gradient(cuts)
    gradient = np.empty_like(parameters)
    gradient[0] = np.sum(cut_gradient)
    for index in range(1, parameters.size):
        gradient[index] = np.exp(parameters[index]) * np.sum(cut_gradient[index:])
    return value, gradient


def oracle() -> dict[str, object]:
    """Gate derivatives and solve the bounded calibration independently."""
    value, gradient = value_gradient(INITIAL_CUTS)
    step = 2.0e-6
    coordinate_fd = np.empty_like(gradient)
    for index in range(gradient.size):
        probe = np.zeros_like(gradient)
        probe[index] = step
        plus, _ = value_gradient(INITIAL_CUTS + probe)
        minus, _ = value_gradient(INITIAL_CUTS - probe)
        coordinate_fd[index] = (plus - minus) / (2.0 * step)
    _, gradient_plus = value_gradient(INITIAL_CUTS + step * DIRECTION)
    _, gradient_minus = value_gradient(INITIAL_CUTS - step * DIRECTION)
    product = (gradient_plus - gradient_minus) / (2.0 * step)
    gradient_error = float(np.max(np.abs(gradient - coordinate_fd)))
    if gradient_error > 5.0e-8:
        raise RuntimeError(f"cut-point gradient oracle failed: {gradient_error:.3e}")
    initial_parameters = np.array([
        INITIAL_CUTS[0],
        np.log(INITIAL_CUTS[1] - INITIAL_CUTS[0] - MINIMUM_GAP),
    ])
    solved = minimize(
        fun=lambda parameters: transformed(parameters),
        x0=initial_parameters,
        method="L-BFGS-B",
        jac=True,
        bounds=((-1.0, 4.0), (-4.0, 2.0)),
        options={"maxiter": 160, "ftol": 1.0e-15, "gtol": 2.0e-9,
                 "maxls": 50},
    )
    if not solved.success:
        raise RuntimeError(f"independent SciPy optimizer failed: {solved.message}")
    final_cuts = decode(np.asarray(solved.x, dtype=np.float64))
    final_value, final_gradient = value_gradient(final_cuts)
    return {
        "initial_value": value,
        "gradient_norm": float(np.linalg.norm(gradient)),
        "hvp_norm": float(np.linalg.norm(product)),
        "gradient_error": gradient_error,
        "final_value": final_value,
        "final_gradient_norm": float(np.linalg.norm(final_gradient)),
        "final_cuts": final_cuts,
    }


def result_row(details: dict[str, object], **values: object) -> dict[str, object]:
    row: dict[str, object] = {field: "" for field in FIELDS}
    row.update(details)
    row.update({"workload": "gp_ordinal_cutpoints", "device": "cpu",
                "n_samples": N, "n_classes": 3, "repetitions": 64})
    row.update(values)
    return row


def parse_app(stdout: str, details: dict[str, object], rows: list[dict[str, object]],
              expected: dict[str, object]) -> None:
    """Validate release checksums and retain timing and boundary rows."""
    gradient_pattern = re.compile(
        r"^gp_ordinal_cutpoint_gradient,cpu,seconds,\s*([0-9Ee+.-]+),"
        r"nll,\s*([0-9Ee+.-]+),norm,\s*([0-9Ee+.-]+)$")
    hvp_pattern = re.compile(
        r"^gp_ordinal_cutpoint_hvp,cpu,seconds,\s*([0-9Ee+.-]+),"
        r"norm,\s*([0-9Ee+.-]+)$")
    training_pattern = re.compile(
        r"^gp_ordinal_cutpoint_training,cpu,seconds,\s*([0-9Ee+.-]+),"
        r"iterations,(\d+),evaluations,(\d+),initial_nll,\s*([0-9Ee+.-]+),"
        r"final_nll,\s*([0-9Ee+.-]+),gradient_norm,\s*([0-9Ee+.-]+),"
        r"cut1,\s*([0-9Ee+.-]+),cut2,\s*([0-9Ee+.-]+)$")
    refusal = re.compile(
        r"^gp_ordinal_cutpoint_(?:gradient|training),device,cuda,refused,(\d+)$")
    gradient_match = hvp_match = training_match = None
    refusal_codes: list[int] = []
    for line in stdout.splitlines():
        line = line.strip()
        gradient_match = gradient_pattern.match(line) or gradient_match
        hvp_match = hvp_pattern.match(line) or hvp_match
        training_match = training_pattern.match(line) or training_match
        match = refusal.match(line)
        if match:
            refusal_codes.append(int(match.group(1)))
    if gradient_match is None or hvp_match is None or training_match is None:
        raise RuntimeError(f"release app omitted cut-point rows:\n{stdout}")
    gradient_seconds, observed_value, observed_gradient = map(
        float, gradient_match.groups())
    hvp_seconds, observed_hvp = map(float, hvp_match.groups())
    (training_seconds, iterations, evaluations, initial_value, final_value,
     final_gradient, cut1, cut2) = training_match.groups()
    final_cuts = np.array([float(cut1), float(cut2)])
    errors = {
        "value": abs(observed_value - float(expected["initial_value"])),
        "gradient": abs(observed_gradient - float(expected["gradient_norm"])),
        "hvp": abs(observed_hvp - float(expected["hvp_norm"])),
        "final_value": abs(float(final_value) - float(expected["final_value"])),
        "cuts": float(np.max(np.abs(final_cuts - expected["final_cuts"]))),
    }
    if errors["value"] > 2.0e-12 or errors["gradient"] > 2.0e-11:
        raise RuntimeError(f"release gradient checksum mismatch: {errors}")
    if errors["hvp"] > 3.0e-9 or errors["final_value"] > 2.0e-10:
        raise RuntimeError(f"release HVP or optimum mismatch: {errors}")
    if errors["cuts"] > 4.0e-6:
        raise RuntimeError(f"release cut-point mismatch: {errors['cuts']:.3e}")
    rows.append(result_row(
        details, phase="release_app", backend="fortml", status="pass",
        seconds_per_operation=gradient_seconds, metric="gradient_norm",
        value=observed_gradient, max_abs_error=max(errors["value"], errors["gradient"]),
        oracle="independent NumPy/SciPy ordered-probit gradient",
        notes=f"initial_nll={observed_value:.16e}"))
    rows.append(result_row(
        details, phase="release_app", backend="fortml", status="pass",
        seconds_per_operation=hvp_seconds, metric="hvp_norm", value=observed_hvp,
        max_abs_error=errors["hvp"],
        oracle="independent central difference of NumPy/SciPy gradient"))
    rows.append(result_row(
        details, phase="release_app", backend="fortml", status="pass",
        seconds_per_operation=float(training_seconds), metric="final_nll",
        value=float(final_value), max_abs_error=max(errors["final_value"], errors["cuts"]),
        oracle="independent bounded SciPy L-BFGS-B",
        notes=(f"iterations={iterations}; evaluations={evaluations}; "
               f"gradient_norm={final_gradient}; cuts={cut1},{cut2}; "
               f"initial_nll={initial_value}")))
    if refusal_codes != [3, 3]:
        raise RuntimeError(f"unexpected CUDA refusal codes: {refusal_codes}")
    rows.append(result_row(
        details, phase="device_boundary", backend="fortml", device="cuda",
        status="refused", metric="cutpoint_graph", value="nan", max_abs_error=0.0,
        oracle="FORTNUM_NOT_IMPLEMENTED",
        notes="objective and optimizer require resident latent and likelihood state"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gp_ordinal_cutpoints.csv"))
    parser.add_argument("--target", default="fortml_bench_gp_ordinal_cutpoints")
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    expected = oracle()
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": os.environ.get("FO_FC", "gfortran"),
        "flags": "-O3",
    }
    rows: list[dict[str, object]] = [result_row(
        details, phase="independent_oracle", backend="numpy_scipy", status="pass",
        metric="gradient_fd_error", value=expected["gradient_error"],
        max_abs_error=expected["gradient_error"],
        oracle="independent dense RBF posterior and ordered-probit reduction",
        notes=(f"initial_nll={expected['initial_value']:.16e}; "
               f"hvp_norm={expected['hvp_norm']:.16e}; "
               f"final_nll={expected['final_value']:.16e}; "
               f"final_cuts={expected['final_cuts']}"))]
    if args.skip_fortml:
        rows.append(result_row(
            details, phase="behavioral_gate", backend="fortml", status="skipped",
            metric="tests_passed", value="nan",
            oracle="test_gp_ordinal_cutpoint_training", notes="--skip-fortml"))
    else:
        environment = os.environ.copy()
        environment.update({"FO_SCAN_FALLBACK": "regex", "OMP_NUM_THREADS": "1"})
        subprocess.run(["fo", "test", "test_gp_ordinal_cutpoint_training"],
                       cwd=fortml, env=environment, check=True)
        rows.append(result_row(
            details, phase="behavioral_gate", backend="fortml", status="pass",
            metric="tests_passed", value=1.0, max_abs_error=0.0,
            oracle="independent Fortran value/FD/adjoint/rollback oracle",
            notes="products, transformed bounds, convergence, transaction, CUDA"))
        subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml,
                       env=environment, check=True, capture_output=True, text=True)
        completed = subprocess.run(
            ["fo", "exec", "--no-build", args.target], cwd=fortml,
            env=environment, check=True, capture_output=True, text=True)
        parse_app(completed.stdout, details, rows, expected)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
