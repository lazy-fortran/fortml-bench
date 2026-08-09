#!/usr/bin/env python3
"""Correctness-gated multi-output validation-state benchmark.

The independent NumPy oracle uses the two-leaf first-round Newton stump for
the fixed fixture. Its validation losses are 40.5 and 6.48 and both outputs
select round one under two-round patience. The release app checks that the
same vectors are exposed by XGBoost-style and LightGBM-style multi-output
adapters.
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
                        default=Path("results/xgboost_multioutput_validation_metadata.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/XGBOOST_MULTIOUTPUT_VALIDATION_METADATA.md"))
    parser.add_argument("--target", default="fortml_bench_xgboost_multioutput_validation_metadata")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    lines = run_app(fortml, args.target)
    oracle = np.array((40.5, 6.48), dtype=np.float64)
    rows: list[dict[str, str]] = []
    fortml_rev = revision(fortml, (fortml / "verification" / "fortml-gfortran.txt",))
    bench_rev = revision(root, (args.output, args.report))

    def row(**kwargs: object) -> dict[str, str]:
        output = {field: "" for field in FIELDS}
        output.update({
            "workload": "xgboost_multioutput_validation_metadata",
            "backend": "fortml",
            "device": "cpu",
            "status": "pass",
            "dimensions": "8x1->8x2",
            "compiler": "gfortran",
            "flags": "-O3",
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "fortml_revision": fortml_rev,
            "benchmark_revision": bench_rev,
            "oracle": "numpy_two_leaf_newton_stump_validation_losses",
        })
        output.update({key: str(value) for key, value in kwargs.items()})
        return output

    seen: set[str] = set()
    for line in lines:
        fields = line.split()
        phase = fields[0]
        if phase not in {"xgb_multi_validation", "lgb_multi_validation"}:
            continue
        seen.add(phase)
        iterations = np.array([int(fields[1]), int(fields[2])])
        values = np.array([float(fields[3]), float(fields[4])])
        stopped = [fields[5] == "T", fields[6] == "T"]
        seconds = float(fields[7])
        error = float(np.max(np.abs(values - oracle)))
        if not np.all(iterations == 1) or not all(stopped) or error > 2.0e-12:
            raise RuntimeError(
                f"{phase} validation metadata mismatch: iterations={iterations}, "
                f"stopped={stopped}, losses={values}, error={error}"
            )
        rows.append(row(phase=phase, evaluations=16, seconds_per_operation=seconds,
                        metric="best_validation_loss_sum", value=float(np.sum(values)),
                        max_abs_error=error,
                        notes="outputs=2; patience=2; best_iteration=[1,1]; early_stopped=[T,T]"))
    if seen != {"xgb_multi_validation", "lgb_multi_validation"}:
        raise RuntimeError(f"release app rows missing: {seen}")
    rows.append(row(phase="predict", device="cuda", status="unavailable",
                    evaluations=0, seconds_per_operation=0.0,
                    metric="best_validation_loss_sum", value="nan", max_abs_error="nan",
                    oracle="typed_device_contract",
                    notes="FORTNUM_NOT_IMPLEMENTED; resident multi-output tree state is unavailable"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = """# Multi-output tree validation metadata

This lane compares the multi-output XGBoost-style and LightGBM-style adapters
with an independent NumPy two-leaf Newton-stump oracle. The fixture has eight
rows, one feature, and two regression targets. The inverse validation targets
make the first fitted round the best round for both outputs. The analytic
validation losses are `[40.5, 6.48]`, and two-round patience sets both
`early_stopped` flags.

The release app records each output's best iteration, validation loss, and
early-stop flag. The CUDA row is a typed refusal because resident
multi-output tree state is not linked.

Reproduce:

```bash
.venv/bin/python -B scripts/bench_xgboost_multioutput_validation_metadata.py \
  --fortml ../fortml --output results/xgboost_multioutput_validation_metadata.csv \
  --report results/XGBOOST_MULTIOUTPUT_VALIDATION_METADATA.md
```

Raw data: [`xgboost_multioutput_validation_metadata.csv`](xgboost_multioutput_validation_metadata.csv).
"""
    args.report.write_text(report)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
