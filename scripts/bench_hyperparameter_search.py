#!/usr/bin/env python3
"""Benchmark FortML's grid and FortOpt L-BFGS-B search orchestration.

The callback is a three-parameter quadratic with a known minimum. NumPy
reconstructs the Cartesian grid and analytic optimum before any timing is
retained. The generic search layer has no resident CUDA objective state, so a
typed CUDA-unavailable row is recorded explicitly.
"""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
from pathlib import Path

import numpy as np


FIELDS = (
    "workload", "phase", "backend", "device", "status", "dimensions",
    "evaluations", "seconds_per_operation", "metric", "value", "max_abs_error",
    "oracle", "python_version", "numpy_version", "fortml_revision",
    "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    value = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True
    )
    ignored_names = set()
    for path in ignored:
        try:
            ignored_names.add(path.resolve().relative_to(repository.resolve()).as_posix())
        except ValueError:
            continue
    dirty = any(line[3:].strip() not in ignored_names for line in status.splitlines())
    return value + ("+dirty" if dirty else "")


def objective(x: np.ndarray) -> float:
    target = np.array((1.0, -0.5, 0.25), dtype=np.float64)
    return float(np.sum((x - target) ** 2))


def oracle_grid() -> tuple[int, np.ndarray, float]:
    values = np.linspace(-2.0, 2.0, 5)
    candidates = np.array(np.meshgrid(values, values, values, indexing="ij"))
    candidates = candidates.reshape(3, -1).T
    values_out = np.array([objective(candidate) for candidate in candidates])
    index = int(np.argmin(values_out))
    return candidates.shape[0], candidates[index], float(values_out[index])


def run_app(fortml: Path, target: str) -> list[str]:
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, check=True)
    completed = subprocess.run(
        ["fo", "exec", "--no-build", target], cwd=fortml,
        check=True, capture_output=True, text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/hyperparameter_search.csv"))
    parser.add_argument("--target", default="fortml_bench_hyperparameter_search")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    lines = run_app(fortml, args.target)
    grid_count, grid_parameters, grid_value = oracle_grid()
    optimum = np.array((1.0, -0.5, 0.25), dtype=np.float64)
    if grid_count != 125 or not np.isfinite(grid_value):
        raise RuntimeError("independent grid oracle is invalid")
    rows: list[dict[str, str]] = []
    fortml_rev = revision(fortml, (fortml / "verification" / "fortml-gfortran.txt",))
    bench_rev = revision(root, tuple(root / "results" / name for name in (
        "xgboost_poisson.csv", "hyperparameter_search.csv")))

    def row(**kwargs: object) -> dict[str, str]:
        output = {field: "" for field in FIELDS}
        output.update({
            "workload": "hyperparameter_search",
            "backend": "fortml",
            "device": "cpu",
            "status": "pass",
            "dimensions": "3",
            "compiler": "gfortran",
            "flags": "-O3",
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "fortml_revision": fortml_rev,
            "benchmark_revision": bench_rev,
            "oracle": "numpy_quadratic_grid_and_optimum",
        })
        output.update({key: str(value) for key, value in kwargs.items()})
        return output

    seen = set()
    for line in lines:
        fields = line.split(",")
        if fields[0] not in {"grid", "lbfgsb"}:
            continue
        evaluations = int(fields[1])
        best_value = float(fields[2])
        seconds = float(fields[3])
        phase = fields[0]
        seen.add(phase)
        if phase == "grid":
            error = abs(best_value - grid_value)
            if evaluations != grid_count or error > 1.0e-13:
                raise RuntimeError(
                    f"grid oracle mismatch: evaluations={evaluations}, error={error}"
                )
            notes = f"grid_parameters={grid_parameters.tolist()}"
        else:
            error = abs(best_value)
            if error > 1.0e-10:
                raise RuntimeError(f"L-BFGS-B quadratic oracle mismatch: {error}")
            notes = f"analytic_optimum={optimum.tolist()}"
        rows.append(row(phase=phase, evaluations=evaluations,
                        seconds_per_operation=seconds, metric="best_value",
                        value=best_value, max_abs_error=error, notes=notes))
    if seen != {"grid", "lbfgsb"}:
        raise RuntimeError(f"release app rows missing: {seen}")
    rows.append(row(phase="search", device="cuda", status="unavailable",
                    evaluations=0, seconds_per_operation=0.0, metric="best_value",
                    value="nan", max_abs_error="nan",
                    oracle="typed_device_contract",
                    notes="no resident CUDA objective/search state; FORTNUM_NOT_IMPLEMENTED"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
