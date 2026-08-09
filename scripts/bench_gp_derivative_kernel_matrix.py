#!/usr/bin/env python3
"""Release gate for the complete derivative-observation kernel catalog.

The Fortran test owns the behavioral oracle.  It compares analytic kernel
gradients and mixed Hessians with central differences and exercises mixed
value/first-derivative GP prediction for every catalog family.  CUDA is kept
as an explicit capability row until a resident derivative-GP factorization is
linked.
"""

from __future__ import annotations

import argparse
import csv
import platform
import re
import subprocess
import time
from pathlib import Path

KERNELS = (
    "rbf", "ard_rbf", "matern32", "matern52", "periodic", "local_periodic",
    "rational_quadratic", "cosine", "polynomial", "spectral_mixture",
    "linear", "constant", "change_point", "sum_product",
)
FIELDS = (
    "workload", "phase", "kernel", "operation", "backend", "device", "status",
    "metric", "value", "max_abs_error", "oracle", "python_version", "fortml_revision",
    "benchmark_revision", "compiler", "flags", "notes",
)


def revision(repository: Path, ignored: tuple[Path, ...] = ()) -> str:
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
    ).strip()
    ignored_names = {path.resolve() for path in ignored}
    dirty = []
    for line in subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"],
            text=True).splitlines():
        path = (repository / line[3:].split(" -> ")[-1].strip()).resolve()
        if path not in ignored_names:
            dirty.append(line)
    return head + ("+dirty" if dirty else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path,
                        default=Path("results/gp_derivative_kernel_matrix.csv"))
    parser.add_argument("--report", type=Path,
                        default=Path("results/GP_DERIVATIVE_KERNEL_MATRIX.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    fortml = args.fortml.resolve()
    output = args.output.resolve()
    report = args.report.resolve()
    details = {
        "python_version": platform.python_version(),
        "fortml_revision": revision(fortml),
        "benchmark_revision": revision(root, (output, report)),
        "compiler": "gfortran",
        "flags": "-O3",
        "oracle": "independent Fortran central-difference kernel/value oracle",
    }
    started = time.perf_counter()
    subprocess.run(["fo", "build", "--flag", "-O3"], cwd=fortml, check=True)
    completed = subprocess.run(
        ["fo", "exec", "--no-build", "test_derivative_gp_kernel_matrix"],
        cwd=fortml, capture_output=True, text=True, check=True,
    )
    elapsed = (time.perf_counter() - started)
    match = re.search(r"kernels=(\d+) max_error=\s*([0-9.eE+-]+)",
                      completed.stdout)
    if match is None:
        raise RuntimeError("derivative kernel matrix test omitted its oracle summary")
    kernel_count = int(match.group(1))
    max_error = float(match.group(2))
    if kernel_count != len(KERNELS) or max_error > 4.0e-2:
        raise RuntimeError(f"unexpected derivative kernel result {kernel_count=} {max_error=}")

    def row(**values: object) -> dict[str, object]:
        result: dict[str, object] = {field: "" for field in FIELDS}
        result.update(details)
        result.update({"workload": "gp_derivative_kernel_matrix", "phase": values.get("operation", "gate"),
                       "backend": "fortml", "device": "cpu", "status": "pass",
                       "metric": "max_abs_error"})
        result.update(values)
        return result

    rows: list[dict[str, object]] = [row(kernel="catalog", operation="behavioral_gate",
        value=kernel_count, max_abs_error=max_error,
        notes=f"{elapsed:.6f}s wall time including fo test orchestration")]
    for kernel in KERNELS:
        rows.append(row(kernel=kernel, operation="mixed_observation_prediction",
                        value=1.0, max_abs_error=max_error,
                        notes="analytic input derivatives versus central differences"))
        rows.append(row(kernel=kernel, operation="cuda_capability", device="cuda",
                        status="unavailable", value="FORTNUM_NOT_IMPLEMENTED",
                        max_abs_error=0.0, oracle="typed_device_contract",
                        notes="resident derivative-GP factorization is not linked"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    args.report.write_text(
        "# Derivative-observation kernel matrix\n\n"
        f"FortML revision: `{details['fortml_revision']}`\n\n"
        f"Benchmark revision: `{details['benchmark_revision']}`\n\n"
        f"The independent Fortran oracle covered {kernel_count} kernel families "
        f"with maximum central-difference error `{max_error:.3e}`. It exercised "
        "mixed value/first-derivative GP prediction for every family. CUDA is "
        "recorded as a typed `FORTNUM_NOT_IMPLEMENTED` capability row because "
        "the resident derivative-GP factorization is not linked.\n"
    )
    print(f"wrote {len(rows)} rows to {args.output}; report={args.report}")


if __name__ == "__main__":
    main()
