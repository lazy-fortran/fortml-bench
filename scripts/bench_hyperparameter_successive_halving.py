#!/usr/bin/env python3
"""Correctness-gated multi-fidelity HPO benchmark.

The independent oracle is a three-parameter quadratic with an analytic
resource penalty. It checks the deterministic 64-candidate rung schedule and
the fixed-resource FortOpt L-BFGS-B optimum. CUDA is recorded as a typed
refusal because the search state is host-owned.
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
                        default=Path("results/hyperparameter_successive_halving.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/HYPERPARAMETER_SUCCESSIVE_HALVING.md"))
    parser.add_argument("--target", default="fortml_bench_hyperparameter_successive_halving")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    lines = run_app(fortml, args.target)
    target = np.array((0.75, -0.25, 0.40), dtype=np.float64)
    resource = 16
    optimum = 0.25 / resource
    rows: list[dict[str, str]] = []
    fortml_rev = revision(fortml, (fortml / "verification" / "fortml-gfortran.txt",))
    bench_rev = revision(root, (args.output, args.report))

    def row(**kwargs: object) -> dict[str, str]:
        output = {field: "" for field in FIELDS}
        output.update({
            "workload": "hyperparameter_successive_halving",
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
            "oracle": "numpy_resource_quadratic_and_lbfgsb_optimum",
        })
        output.update({key: str(value) for key, value in kwargs.items()})
        return output

    seen: set[str] = set()
    for line in lines:
        fields = line.split(",")
        phase = fields[0]
        seen.add(phase)
        if phase == "successive_halving":
            candidates, rungs, evaluations = map(int, fields[1:4])
            value, seconds = map(float, fields[4:6])
            expected_evaluations = 64 + 32 + 16 + 8 + 4
            error = max(abs(candidates - 64), abs(rungs - 5),
                        abs(evaluations - expected_evaluations))
            if candidates != 64 or rungs != 5 or evaluations != expected_evaluations:
                raise RuntimeError(
                    f"rung accounting mismatch: candidates={candidates}, "
                    f"rungs={rungs}, evaluations={evaluations}"
                )
            if not np.isfinite(value) or value > 1.0:
                raise RuntimeError(f"successive-halving oracle mismatch: value={value}")
            rows.append(row(phase=phase, evaluations=evaluations,
                            seconds_per_operation=seconds, metric="best_value",
                            value=value, max_abs_error=error,
                            notes="candidates=64; resources=1,2,4,8,16; factor=2; seed=20260809"))
        elif phase == "lbfgsb_resource":
            evaluations = int(fields[1])
            value, seconds = map(float, fields[2:4])
            error = abs(value - optimum)
            if evaluations < 1 or not np.isfinite(value) or error > 2.0e-9:
                raise RuntimeError(
                    f"resource L-BFGS-B oracle mismatch: evaluations={evaluations}, "
                    f"value={value}, error={error}"
                )
            rows.append(row(phase=phase, evaluations=evaluations,
                            seconds_per_operation=seconds, metric="best_value",
                            value=value, max_abs_error=error,
                            notes=f"target={target.tolist()}; resource={resource}"))
    if seen != {"successive_halving", "lbfgsb_resource"}:
        raise RuntimeError(f"release app rows missing: {seen}")
    rows.append(row(phase="search", device="cuda", status="unavailable",
                    evaluations=0, seconds_per_operation=0.0, metric="best_value",
                    value="nan", max_abs_error="nan", oracle="typed_device_contract",
                    notes="FORTNUM_NOT_IMPLEMENTED; no resident CUDA search state"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = """# Successive-halving hyperparameter search

This lane checks a deterministic multi-fidelity search over a three-parameter
quadratic objective. The independent NumPy oracle expects 64 candidates,
resource rungs 1, 2, 4, 8, and 16, and 124 total objective evaluations after
factor-two pruning. Every callback returns a value and analytic parameter
gradient. The surviving vector is refined at resource 16 through FortOpt
L-BFGS-B, whose value must equal the analytic resource penalty `0.25 / 16`.

The CUDA row records `FORTNUM_NOT_IMPLEMENTED` because the generic search state
is CPU-owned. No host fallback is counted as resident GPU execution.

Reproduce:

```bash
.venv/bin/python -B scripts/bench_hyperparameter_successive_halving.py \\
  --fortml ../fortml --output results/hyperparameter_successive_halving.csv \\
  --report results/HYPERPARAMETER_SUCCESSIVE_HALVING.md
```

Raw data: [`hyperparameter_successive_halving.csv`](hyperparameter_successive_halving.csv).
"""
    args.report.write_text(report)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
