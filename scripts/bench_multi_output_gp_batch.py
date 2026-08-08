#!/usr/bin/env python3
"""Independent oracle and contract gate for batched multi-output GP products.

The NumPy oracle assembles the output-major intrinsic-coregionalization
system directly, predicts every batch member, and differentiates query points
analytically.  It does not call FortML for expected values.  The public gate
then runs the independent Fortran test, while CUDA remains an explicit typed
refusal until resident multi-output state exists.
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "n_features", "n_parameters", "seconds_per_operation", "metric", "value",
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


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, batch, points, outputs = 5, 2, 3, 2
    x = (-0.8 + 0.35*np.arange(n, dtype=np.float64))[:, None]
    y = np.column_stack((np.sin(1.2*x[:, 0]), np.cos(0.9*x[:, 0]) - 0.2))
    query = np.empty((batch, points, 1), dtype=np.float64)
    direction = np.empty_like(query)
    for b in range(batch):
        query[b, :, 0] = -0.55 + 0.27*np.arange(points) + 0.18*b
        direction[b, :, 0] = 0.11 - 0.035*np.arange(1, points + 1) + 0.04*b
    mean_bar = np.array(
        [[[0.3, -0.2], [0.7, -0.4], [0.1, 0.6]],
         [[-0.5, 0.8], [-0.25, 0.35], [-0.15, 0.45]]],
        dtype=np.float64,
    )
    return x, y, query, direction, mean_bar


def oracle() -> tuple[float, float, float, int]:
    x, y, query, direction, mean_bar = fixture()
    variance, lengthscale, noise = 1.3, 0.7, 0.14
    weights = np.array([[0.75], [-0.4]], dtype=np.float64)
    independent = np.array([0.22, 0.31], dtype=np.float64)
    coreg = weights @ weights.T + np.diag(independent)
    distance = (x - x.T)**2
    kernel = variance*np.exp(-0.5*distance/lengthscale**2)
    joint = np.kron(coreg, kernel) + noise*np.eye(x.shape[0]*2)
    alpha = np.linalg.solve(joint, y.T.reshape(-1))

    def evaluate(points: np.ndarray, tangent: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        means = np.zeros((points.shape[0], points.shape[1], 2), dtype=np.float64)
        query_bar = np.zeros_like(points)
        tangent_mean = None
        if tangent is not None:
            tangent_mean = np.zeros_like(means)
        for batch in range(points.shape[0]):
            cross = variance*np.exp(
                -0.5*(points[batch, :, 0, None] - x[None, :, 0])**2/
                lengthscale**2
            )
            for output in range(2):
                for latent_output in range(2):
                    block = coreg[output, latent_output]*cross
                    alpha_block = alpha[latent_output*x.shape[0]:
                                       (latent_output + 1)*x.shape[0]]
                    means[batch, :, output] += block @ alpha_block
                    derivative = block*(-(points[batch, :, 0, None] - x[:, 0])/
                                        lengthscale**2)
                    derivative_values = derivative @ alpha_block
                    if tangent_mean is not None:
                        tangent_mean[batch, :, output] += (
                            derivative_values*tangent[batch, :, 0]
                        )
                    query_bar[batch, :, 0] += (
                        mean_bar[batch, :, output]*derivative_values
                    )
        return means, query_bar, tangent_mean

    mean, query_bar, tangent_mean = evaluate(query, direction)
    eps = 2.0e-6
    plus, _, _ = evaluate(query + eps*direction)
    minus, _, _ = evaluate(query - eps*direction)
    mean_dot = (plus - minus)/(2.0*eps)
    mean_bar_dot = float(np.sum(mean_bar*mean_dot))
    query_bar_dot = float(np.sum(query_bar*direction))
    adjoint_error = abs(mean_bar_dot - query_bar_dot)
    jvp_error = float(np.max(np.abs(tangent_mean - mean_dot)))
    if not np.isfinite(mean).all() or adjoint_error > 2.0e-9 or jvp_error > 2.0e-9:
        raise RuntimeError(f"batched multi-output oracle failed: {adjoint_error:.3e}")
    return jvp_error, jvp_error, adjoint_error, 7


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/multi_output_gp_batch.csv"))
    parser.add_argument("--skip-fortml", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml, output = args.fortml.resolve(), args.output.resolve()
    value, jvp_error, adjoint_error, n_parameters = oracle()
    details = {
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output,)),
        "compiler": "gfortran", "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(details)
        row.update({"workload": "multi_output_gp_batch", "backend": "fortml",
                    "device": "cpu", "n_samples": 5, "n_features": 1,
                    "n_parameters": n_parameters})
        row.update(values)
        rows.append(row)

    add(phase="independent_oracle", status="pass", metric="input_products_max_abs_error",
        value=value, max_abs_error=max(jvp_error, adjoint_error),
        oracle="independent NumPy output-major ICM solve and query derivative",
        notes="batched mean, central-difference JVP, and VJP scalar duality")
    started = time.perf_counter()
    if args.skip_fortml:
        status, notes = "skipped", "--skip-fortml"
    else:
        subprocess.run(["fo", "test", "test_multi_output_gp_batch"],
                       cwd=fortml, check=True)
        status, notes = "pass", "batch mean/JVP/VJP shape and device contract gate"
    elapsed = time.perf_counter() - started
    add(phase="public_contract_gate", status=status,
        seconds_per_operation=elapsed, metric="oracle_adjoint_error",
        value=adjoint_error, max_abs_error=adjoint_error,
        oracle="FortML test_multi_output_gp_batch", notes=notes)
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_multi_output_batch_graph", value="nan", max_abs_error="",
        oracle="typed FortML CUDA refusal",
        notes="coregionalized batch covariance/factorization is not resident")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
