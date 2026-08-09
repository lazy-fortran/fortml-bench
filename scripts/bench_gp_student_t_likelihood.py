#!/usr/bin/env python3
"""Benchmark fixed-latent Student-t likelihood products.

The oracle is an independent scalar density in Python. Central differences of
that density check the value, gradient, JVP, and directional HVP printed by the
FortML release probe. The objective row records the FortOpt context's decrease;
CUDA is a typed refusal rather than a host fallback.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "n_samples",
    "seconds_per_operation", "metric", "value", "max_abs_error", "oracle",
    "python_version", "numpy_version", "fortml_revision", "benchmark_revision",
    "compiler", "flags", "notes",
)


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


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    locations = np.linspace(-1.2, 1.2, 7, dtype=np.float64)
    observations = locations + 0.25 * np.sin(1.7 * locations)
    parameters = np.log(np.array([0.75, 4.3], dtype=np.float64))
    direction = np.array([0.35, -0.6], dtype=np.float64)
    return observations, locations, parameters, direction


def oracle_value(observations: np.ndarray, locations: np.ndarray,
                 parameters: np.ndarray) -> float:
    scale, nu = np.exp(parameters)
    residual = observations - locations
    q = (residual / (scale * np.sqrt(nu))) ** 2
    return float(np.sum(
        math.lgamma(0.5 * (nu + 1.0)) - math.lgamma(0.5 * nu)
        - 0.5 * math.log(nu * math.pi) - math.log(scale)
        - 0.5 * (nu + 1.0) * np.log1p(q)
    ))


def oracle_gradient(observations: np.ndarray, locations: np.ndarray,
                    parameters: np.ndarray, step: float) -> np.ndarray:
    gradient = np.empty(2, dtype=np.float64)
    for index in range(2):
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (
            oracle_value(observations, locations, plus)
            - oracle_value(observations, locations, minus)
        ) / (2.0 * step)
    return gradient


def parse_probe(stdout: str) -> dict[str, float | int]:
    records: dict[str, float | int] = {}
    for line in stdout.splitlines():
        if not line.startswith("student_t_observation_"):
            continue
        key, raw = line.strip().split(",", maxsplit=1)
        records[key] = int(raw) if key.endswith("converged") or key.endswith("iterations") else float(raw)
    required = {
        "student_t_observation_value", "student_t_observation_gradient_scale",
        "student_t_observation_gradient_nu", "student_t_observation_jvp",
        "student_t_observation_vjp_scale", "student_t_observation_vjp_nu",
        "student_t_observation_hvp_scale", "student_t_observation_hvp_nu",
        "student_t_observation_initial_objective",
        "student_t_observation_optimized_objective",
        "student_t_observation_optimizer_iterations",
        "student_t_observation_optimizer_converged",
    }
    missing = required - records.keys()
    if missing:
        raise RuntimeError(f"Student-t likelihood probe omitted {sorted(missing)}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gp_student_t_likelihood.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/GP_STUDENT_T_LIKELIHOOD.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    report = args.report.resolve()
    environment = os.environ.copy()
    environment["FO_SCAN_FALLBACK"] = "regex"
    subprocess.run(["fo", "test", "test_student_t_likelihood"], cwd=fortml,
                   env=environment, check=True)
    started = time.perf_counter()
    probe = subprocess.run(
        ["fo", "exec", "fortml_bench_student_t_likelihood_products"],
        cwd=fortml, env=environment, check=True, text=True, capture_output=True,
    )
    elapsed = time.perf_counter() - started
    record = parse_probe(probe.stdout)
    observations, locations, parameters, direction = fixture()
    value_oracle = oracle_value(observations, locations, parameters)
    fd_step = 2.0e-4
    gradient_oracle = oracle_gradient(observations, locations, parameters, fd_step)
    plus = parameters + fd_step * direction
    minus = parameters - fd_step * direction
    hvp_oracle = (
        oracle_gradient(observations, locations, plus, fd_step)
        - oracle_gradient(observations, locations, minus, fd_step)
    ) / (2.0 * fd_step)
    value_error = abs(float(record["student_t_observation_value"]) - value_oracle)
    gradient_error = max(
        abs(float(record["student_t_observation_gradient_scale"]) - gradient_oracle[0]),
        abs(float(record["student_t_observation_gradient_nu"]) - gradient_oracle[1]),
    )
    jvp_oracle = float(np.dot(gradient_oracle, direction))
    jvp_error = abs(float(record["student_t_observation_jvp"]) - jvp_oracle)
    vjp_error = max(
        abs(float(record["student_t_observation_vjp_scale"]) - gradient_oracle[0]),
        abs(float(record["student_t_observation_vjp_nu"]) - gradient_oracle[1]),
    )
    hvp_error = max(
        abs(float(record["student_t_observation_hvp_scale"]) - hvp_oracle[0]),
        abs(float(record["student_t_observation_hvp_nu"]) - hvp_oracle[1]),
    )
    maximum_error = max(value_error, gradient_error, jvp_error, vjp_error, hvp_error)
    if value_error > 3.0e-12 or gradient_error > 3.0e-7 or jvp_error > 4.0e-7 or \
            vjp_error > 3.0e-7 or hvp_error > 3.0e-5:
        raise RuntimeError(
            "Student-t likelihood oracle failed: "
            f"value={value_error:.3e}, gradient={gradient_error:.3e}, "
            f"jvp={jvp_error:.3e}, vjp={vjp_error:.3e}, hvp={hvp_error:.3e}"
        )
    initial_objective = float(record["student_t_observation_initial_objective"])
    optimized_objective = float(record["student_t_observation_optimized_objective"])
    if not optimized_objective < initial_objective:
        raise RuntimeError("FortOpt Student-t objective did not decrease")
    details = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output, report)),
        "compiler": "gfortran", "flags": "-O3",
    }
    rows: list[dict[str, object]] = []

    def add(**values: object) -> None:
        row = {field: "" for field in FIELDS}
        row.update(details)
        row.update({"workload": "gp_student_t_likelihood", "backend": "fortml",
                    "device": "cpu", "n_samples": observations.size})
        row.update(values)
        rows.append(row)

    add(phase="fixed_latent_products", status="pass", seconds_per_operation=elapsed,
        metric="value_gradient_jvp_vjp_hvp_max_abs", value=maximum_error,
        max_abs_error=maximum_error,
        oracle="independent NumPy Student-t density central differences",
        notes=(f"value={value_error:.3e}; gradient={gradient_error:.3e}; "
               f"jvp={jvp_error:.3e}; vjp={vjp_error:.3e}; hvp={hvp_error:.3e}"))
    add(phase="fortopt_context", status="pass", metric="objective_decrease",
        value=initial_objective - optimized_objective, max_abs_error="nan",
        oracle="FortOpt L-BFGS-B callback value/gradient",
        notes=(f"initial={initial_objective:.16e}; optimized={optimized_objective:.16e}; "
               f"iterations={int(record['student_t_observation_optimizer_iterations'])}; "
               f"converged={int(record['student_t_observation_optimizer_converged'])}"))
    add(phase="device_contract", device="cuda", status="unavailable",
        metric="resident_student_t_special_functions", value="nan", max_abs_error="nan",
        oracle="typed FORTNUM_NOT_IMPLEMENTED refusal",
        notes="fixed-latent products are CPU-only; no host fallback")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# Fixed-latent Student-t GP likelihood products\n\n"
        "`bench_gp_student_t_likelihood.py` checks the normalized Student-t "
        "observation likelihood over stable `[log(scale), log(nu)]` coordinates. "
        "The independent NumPy scalar density uses central differences for the "
        "gradient and directional HVP, then checks the JVP/VJP products and the "
        "FortOpt objective callback.\n\n"
        "Run:\n\n"
        "```bash\n"
        "python -B scripts/bench_gp_student_t_likelihood.py "
        "--fortml ../fortml --output results/gp_student_t_likelihood.csv "
        "--report results/GP_STUDENT_T_LIKELIHOOD.md\n"
        "```\n\n"
        f"The recorded maximum product error is `{maximum_error:.3e}` for "
        f"{observations.size} fixed latent rows. FortOpt decreased the negative "
        f"log likelihood from `{initial_objective:.16e}` to "
        f"`{optimized_objective:.16e}`. The release source revision is "
        f"`{details['fortml_revision']}` and the benchmark revision is "
        f"`{details['benchmark_revision']}`. CUDA is an explicit typed refusal "
        "until resident latent batches and special functions are linked; no "
        "GPU timing or hidden host fallback is claimed.\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
