"""Sweep every scalable-GP method of the Liu et al. review.

Reference: H. Liu, Y.-S. Ong, X. Shen and J. Cai, "When Gaussian Process Meets
Big Data: A Review of Scalable GPs", IEEE Transactions on Neural Networks and
Learning Systems 31(11):4405-4423, 2020, doi:10.1109/TNNLS.2019.2957109.

The review has no numeric result table to reproduce. What it does state is a
set of qualitative behaviours (its Figs. 4 and 5) and a set of complexity
orders (its Fig. 2 and Sections III to V). The behaviours are checked inside
fortml's own test suite on the paper's fixture; this script measures the
orders, plus wall time, peak resident memory and predictive accuracy, so the
claimed complexity can be compared against a measured one.

Every row is produced by the pinned fortml benchmark binary, which reports its
own peak resident set size from the kernel, so the memory figure covers all of
the process rather than the part Python can see.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

METHODS = (
    "full",
    "sod",
    "sor",
    "dtc",
    "fitc",
    "pitc",
    "vfe",
    "ski",
    "nle",
    "poe",
    "gpoe",
    "bcm",
    "rbcm",
    "grbcm",
    "moe",
    "keops",
)

#: What the review claims each method costs, for the measured-versus-claimed
#: table. `n` training points, `m` inducing or subset size, `M` experts,
#: `m0 = n/M` points per expert, `d` input dimensions.
CLAIMED_ORDER = {
    "full": "O(n^3) train, O(n^2) predict variance (Sec. II)",
    "sod": "O(m^3) train (Sec. III-A)",
    "sor": "O(n m^2) train, O(m) predict mean (Sec. III-C1)",
    "dtc": "O(n m^2) train, O(m^2) predict variance (Sec. III-C1)",
    "fitc": "O(n m^2) train (Sec. III-C1)",
    "pitc": "O(n m^2 + M b^3) train, b the block size (Sec. III-C1)",
    "vfe": "O(n m^2) train (Sec. III-C2)",
    "ski": "O(n + m log m) per product (Sec. III-C3, eq. 20)",
    "nle": "O(M m0^3) = O(n m0^2) train (Sec. IV-A)",
    "poe": "O(n m0^2) train, O(M m0^2) predict (Sec. IV-C)",
    "gpoe": "O(n m0^2) train, O(M m0^2) predict (Sec. IV-C)",
    "bcm": "O(n m0^2) train, O(M m0^2) predict (Sec. IV-C)",
    "rbcm": "O(n m0^2) train, O(M m0^2) predict (Sec. IV-C)",
    "grbcm": "O(n m0^2) train plus one global expert (Sec. IV-C, eq. 29)",
    "moe": "O(n m0^2) train, O(M m0^2) predict (Sec. IV-B)",
    "keops": "O(s n^2) train for s CG iterations, matrix free (Sec. III-C3)",
}

FIELDS = (
    "method",
    "n",
    "m",
    "n_experts",
    "d",
    "train_seconds",
    "predict_seconds",
    "peak_kib",
    "smse",
    "mnlpd",
)


@dataclass(frozen=True)
class Point:
    method: str
    n: int
    m: int
    experts: int
    d: int


def run_point(fortml: Path, point: Point, repetitions: int) -> dict[str, str]:
    """Run one configuration and parse the single CSV line it prints."""
    command = [
        "fo",
        "exec",
        "--no-build",
        "fortml_bench_scalable_gp",
        point.method,
        str(point.n),
        str(point.m),
        str(point.experts),
        str(point.d),
        str(repetitions),
    ]
    completed = subprocess.run(
        command, cwd=fortml, capture_output=True, text=True, check=True
    )
    line = completed.stdout.strip().splitlines()[-1]
    values = [item.strip() for item in line.split(",")]
    if len(values) != len(FIELDS):
        raise SystemExit(f"unexpected benchmark output: {line!r}")
    return dict(zip(FIELDS, values))


def build(fortml: Path) -> None:
    subprocess.run(["fo", "build"], cwd=fortml, check=True, capture_output=True)


def revisions(fortml: Path) -> dict[str, str]:
    def head(repository: Path) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()

    return {
        "fortml": head(fortml),
        "fortnum": head(fortml.parent / "fortnum"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--sweep",
        choices=("samples", "inducing", "experts", "dimension", "single"),
        default="samples",
    )
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--values", type=int, nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    arguments = parser.parse_args()

    fortml = arguments.fortml.resolve()
    if arguments.repetitions < 1:
        raise SystemExit("repetitions must be positive")
    unknown = set(arguments.methods) - set(METHODS)
    if unknown:
        raise SystemExit(f"unknown methods: {sorted(unknown)}")

    defaults = {
        "samples": [256, 512, 1024, 2048],
        "inducing": [16, 32, 64, 128, 256],
        "experts": [2, 4, 8, 16, 32],
        "dimension": [1, 2, 4, 8],
        "single": [0],
    }
    values = arguments.values or defaults[arguments.sweep]

    build(fortml)
    heads = revisions(fortml)
    print(f"fortml {heads['fortml'][:8]}  fortnum {heads['fortnum'][:8]}",
          file=sys.stderr)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("sweep", "swept_value", *FIELDS, "fortml_commit")
        )
        writer.writeheader()
        for value in values:
            for method in arguments.methods:
                point = Point(
                    method=method,
                    n=value if arguments.sweep == "samples" else arguments.n,
                    m=value if arguments.sweep == "inducing" else arguments.m,
                    experts=(
                        value if arguments.sweep == "experts" else arguments.experts
                    ),
                    d=value if arguments.sweep == "dimension" else arguments.d,
                )
                # SKI's grid path here is one dimensional; skip rather than
                # record a refusal as if it were a measurement.
                if method == "ski" and point.d != 1:
                    continue
                try:
                    row = run_point(fortml, point, arguments.repetitions)
                except subprocess.CalledProcessError as error:
                    print(
                        f"skip {method} at {arguments.sweep}={value}: "
                        f"{error.stderr.strip().splitlines()[-1:]}",
                        file=sys.stderr,
                    )
                    continue
                row["sweep"] = arguments.sweep
                row["swept_value"] = str(value)
                row["fortml_commit"] = heads["fortml"]
                writer.writerow(row)
                handle.flush()
                print(
                    f"{method:6s} {arguments.sweep}={value:<6d} "
                    f"train={row['train_seconds']} peak={row['peak_kib']} KiB "
                    f"smse={row['smse']}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
