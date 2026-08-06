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
import hashlib
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

LOCAL_METHODS = (
    "nle",
    "poe",
    "gpoe",
    "bcm",
    "rbcm",
    "grbcm",
    "moe",
)

METHODS = (
    "full",
    "sod",
    "sor",
    "dtc",
    "fitc",
    "pitc",
    "vfe",
    "ski",
    *(name for method in LOCAL_METHODS for name in (method, f"{method}_clustered")),
    "keops",
    "keops_gpu",
    "keops_matvec",
)

#: What the review claims each method costs, for the measured-versus-claimed
#: table. `n` training points, `m` inducing or subset size, `M` experts,
#: `m0 = n/M` points per expert, `d` input dimensions.
CLAIMED_ORDER = {
    "keops_gpu": "O(s n^2) train, resident OpenACC products",
    "keops_matvec": "O(n^2) for one matrix-free product",
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
for local_method in LOCAL_METHODS:
    CLAIMED_ORDER[f"{local_method}_clustered"] = (
        CLAIMED_ORDER[local_method] + "; plus Lloyd partition O(I n M d)"
    )

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
DEFAULT_FLAGS = "-O3 -funroll-loops"
REVIEW_TOY_METHODS = {
    "full",
    "sor",
    "dtc",
    "fitc",
    "vfe",
    "poe",
    "gpoe",
    "moe",
}


@dataclass(frozen=True)
class Point:
    method: str
    n: int
    m: int
    experts: int
    d: int


def ski_grid_shape(budget: int, dimensions: int) -> tuple[int, int]:
    """Return equal per-axis extent and used points under a total budget."""
    if budget < 1 or dimensions < 1:
        return 0, 0
    extent = max(int(round(budget ** (1.0 / dimensions))), 1)
    while (extent + 1) ** dimensions <= budget:
        extent += 1
    while extent**dimensions > budget:
        extent -= 1
    return extent, extent**dimensions


def partition_metadata(method: str) -> str:
    if method == "grbcm":
        return "seeded_communication+contiguous_remainder"
    if method == "grbcm_clustered":
        return "seeded_communication+deterministic_lloyd_remainder"
    if method.endswith("_clustered"):
        return "deterministic_lloyd"
    if method in LOCAL_METHODS:
        return "contiguous_input_order"
    return "not_applicable"


def refused_row(point: Point) -> dict[str, str]:
    return {
        "method": point.method,
        "n": str(point.n),
        "m": str(point.m),
        "n_experts": str(point.experts),
        "d": str(point.d),
        "train_seconds": "NaN",
        "predict_seconds": "NaN",
        "peak_kib": "NaN",
        "smse": "NaN",
        "mnlpd": "NaN",
    }


def is_mean_only(point: Point) -> bool:
    if point.method in {"ski", "keops_matvec"}:
        return True
    return point.method in {"keops", "keops_gpu"} and point.n > 2048


def classify_row(point: Point, row: dict[str, str]) -> str:
    required_names = ("train_seconds", "predict_seconds", "peak_kib", "smse")
    try:
        required = [float(row[name]) for name in required_names]
        mnlpd = float(row["mnlpd"])
    except (KeyError, ValueError):
        return "failed_nonfinite"
    if all(not math.isfinite(value) for value in (*required, mnlpd)):
        return "refused_by_driver"
    if not all(math.isfinite(value) for value in required):
        return "failed_nonfinite"
    if not math.isfinite(mnlpd) and not is_mean_only(point):
        return "failed_nonfinite"
    return "pass"


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


def build(fortml: Path, flags: str) -> float:
    """Release flags. The default fo profile is -O0 -fcheck=all, which would
    make every timing here a debug-build timing."""
    started = time.perf_counter()
    subprocess.run(
        ["fo", "build", "--flag", flags],
        cwd=fortml,
        check=True,
        capture_output=True,
    )
    return time.perf_counter() - started


def method_oracles(method: str) -> tuple[str, ...]:
    """Return the focused behavioral tests supporting one benchmark lane."""
    base = method.removesuffix("_clustered")
    if base in LOCAL_METHODS:
        targets = ["test_local_experts"]
    elif method == "full":
        targets = ["test_gaussian_process"]
    elif method in {"sod", "ski"}:
        targets = ["test_ski_gp"]
    elif method in {"sor", "dtc", "fitc", "pitc"}:
        targets = ["test_sparse_prior_gp"]
    elif method == "vfe":
        targets = ["test_sparse_gp"]
    elif method.startswith("keops"):
        targets = ["test_kernel_operator"]
    else:
        raise ValueError(f"no correctness oracle mapped for {method}")
    if method in REVIEW_TOY_METHODS:
        targets.append("test_review_toy")
    return tuple(targets)


def verify_method_oracles(fortml: Path, methods: list[str]) -> None:
    targets = list(
        dict.fromkeys(
            target for method in methods for target in method_oracles(method)
        )
    )
    for target in targets:
        subprocess.run(
            ["fo", "test", target],
            cwd=fortml,
            check=True,
            capture_output=True,
        )


def oracle_metadata(method: str) -> str:
    return ";".join(method_oracles(method))


def revisions(fortml: Path) -> dict[str, str]:
    def revision(repository: Path) -> str:
        commit = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(repository), "status", "--porcelain"], text=True
        ).strip()
        return commit + ("+dirty" if dirty else "")

    return {
        "fortml": revision(fortml),
        "fortnum": revision(fortml.parent / "fortnum"),
        "benchmark": revision(Path(__file__).resolve().parents[1]),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_diff_sha256(repository: Path) -> str:
    difference = subprocess.check_output(
        ["git", "-C", str(repository), "diff", "--binary", "HEAD"]
    )
    return hashlib.sha256(difference).hexdigest()


def compiler_version(compiler: str) -> str:
    output = subprocess.check_output(
        [compiler, "--version"], text=True, stderr=subprocess.STDOUT
    )
    return next(line.strip() for line in output.splitlines() if line.strip())


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fortml", type=Path, default=Path("../fortml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cpu-affinity", type=int, nargs="+", default=None)
    parser.add_argument("--compiler", default=os.environ.get("FC", "gfortran"))
    parser.add_argument("--flags", default=DEFAULT_FLAGS)
    parser.add_argument(
        "--sweep",
        choices=("samples", "inducing", "experts", "dimension", "single"),
        default="samples",
    )
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--expert-size", type=int, default=None)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--values", type=int, nargs="+", default=None)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    arguments = parser.parse_args()

    fortml = arguments.fortml.resolve()
    if arguments.repetitions < 1:
        raise SystemExit("repetitions must be positive")
    if arguments.threads < 1:
        raise SystemExit("threads must be positive")
    unknown = set(arguments.methods) - set(METHODS)
    if unknown:
        raise SystemExit(f"unknown methods: {sorted(unknown)}")
    allowed_cpus = sorted(os.sched_getaffinity(0))
    affinity = (
        sorted(dict.fromkeys(arguments.cpu_affinity))
        if arguments.cpu_affinity is not None
        else allowed_cpus[: arguments.threads]
    )
    unavailable = set(affinity) - set(allowed_cpus)
    if unavailable:
        raise SystemExit(f"CPU affinity is outside the allowed mask: {sorted(unavailable)}")
    if len(affinity) < arguments.threads:
        raise SystemExit(
            f"{arguments.threads} threads require at least that many affinity CPUs"
        )
    os.sched_setaffinity(0, affinity)
    affinity = sorted(os.sched_getaffinity(0))
    thread_count = str(arguments.threads)
    os.environ.update(
        {
            "FO_FC": arguments.compiler,
            "OMP_NUM_THREADS": thread_count,
            "OPENBLAS_NUM_THREADS": thread_count,
            "MKL_NUM_THREADS": thread_count,
        }
    )

    defaults = {
        "samples": [256, 512, 1024, 2048],
        "inducing": [16, 32, 64, 128, 256],
        "experts": [2, 4, 8, 16, 32],
        "dimension": [1, 2, 4, 8],
        "single": [0],
    }
    values = arguments.values or defaults[arguments.sweep]

    verify_method_oracles(fortml, arguments.methods)
    build_seconds = build(fortml, arguments.flags)
    heads = revisions(fortml)
    provenance = {
        "repetitions": str(arguments.repetitions),
        "threads": thread_count,
        "cpu_affinity": ";".join(str(cpu) for cpu in affinity),
        "cpu_model": cpu_model(),
        "os": platform.platform(),
        "compiler": arguments.compiler,
        "compiler_version": compiler_version(arguments.compiler),
        "flags": arguments.flags,
        "build_seconds": str(build_seconds),
        "fortml_commit": heads["fortml"],
        "fortnum_commit": heads["fortnum"],
        "benchmark_commit": heads["benchmark"],
        "fortml_diff_sha256": tracked_diff_sha256(fortml),
        "driver_sha256": sha256(Path(__file__).resolve()),
        "app_sha256": sha256(
            fortml / "app" / "fortml_bench_scalable_gp.f90"
        ),
        "local_experts_sha256": sha256(
            fortml / "src" / "gp" / "fortml_local_experts.f90"
        ),
        "ski_source_sha256": sha256(
            fortml / "src" / "gp" / "fortml_ski_gp.f90"
        ),
    }
    print(f"fortml {heads['fortml'][:8]}  fortnum {heads['fortnum'][:8]}",
          file=sys.stderr)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sweep",
                "swept_value",
                *FIELDS,
                "status",
                "claimed_order",
                "method_order",
                "partition",
                "ski_grid_budget",
                "ski_axis_extent",
                "ski_grid_points",
                "notes",
                "correctness_oracle",
                "repetitions",
                "threads",
                "cpu_affinity",
                "cpu_model",
                "os",
                "compiler",
                "compiler_version",
                "flags",
                "build_seconds",
                "fortml_commit",
                "fortnum_commit",
                "benchmark_commit",
                "fortml_diff_sha256",
                "driver_sha256",
                "app_sha256",
                "local_experts_sha256",
                "ski_source_sha256",
            ),
        )
        writer.writeheader()
        for value in values:
            for method in arguments.methods:
                samples = value if arguments.sweep == "samples" else arguments.n
                experts = value if arguments.sweep == "experts" else arguments.experts
                # Holding the per-expert size fixed is what makes the local
                # methods scalable: at fixed M the per-expert Cholesky grows
                # with n, so both time and memory grow quadratically.
                if arguments.expert_size:
                    experts = max(samples // arguments.expert_size, 1)
                point = Point(
                    method=method,
                    n=samples,
                    m=value if arguments.sweep == "inducing" else arguments.m,
                    experts=experts,
                    d=value if arguments.sweep == "dimension" else arguments.d,
                )
                axis_extent, grid_points = ski_grid_shape(point.m, point.d)
                notes = ""
                if method == "ski" and axis_extent < 2:
                    row = refused_row(point)
                    status = "refused_grid_budget"
                    notes = "total grid budget cannot provide two points per dimension"
                    print(
                        f"refuse ski at d={point.d}: total grid budget "
                        f"{point.m} cannot provide two points per dimension",
                        file=sys.stderr,
                    )
                else:
                    try:
                        row = run_point(fortml, point, arguments.repetitions)
                    except subprocess.CalledProcessError as error:
                        row = refused_row(point)
                        status = "failed"
                        notes = " ".join(error.stderr.strip().splitlines()[-1:])
                        print(
                            f"failed {method} at {arguments.sweep}={value}: "
                            f"{error.stderr.strip().splitlines()[-1:]}",
                            file=sys.stderr,
                        )
                    else:
                        status = classify_row(point, row)
                        if status == "refused_by_driver":
                            notes = "FortML application refused an infeasible request"
                        elif status == "failed_nonfinite":
                            notes = "FortML application returned incomplete metrics"
                        elif not math.isfinite(float(row["mnlpd"])):
                            notes = "mean-only lane; predictive density is undefined"
                row["sweep"] = arguments.sweep
                row["swept_value"] = str(value)
                row["status"] = status
                row["claimed_order"] = CLAIMED_ORDER[method]
                row["method_order"] = str(METHODS.index(method))
                row["partition"] = partition_metadata(method)
                row["ski_grid_budget"] = str(point.m) if method == "ski" else ""
                row["ski_axis_extent"] = (
                    str(axis_extent) if method == "ski" else ""
                )
                row["ski_grid_points"] = (
                    str(grid_points) if method == "ski" else ""
                )
                row["notes"] = notes
                row["correctness_oracle"] = oracle_metadata(method)
                row.update(provenance)
                writer.writerow(row)
                handle.flush()
                print(
                    f"{method:18s} {arguments.sweep}={value:<6d} "
                    f"status={status} train={row['train_seconds']} "
                    f"peak={row['peak_kib']} KiB smse={row['smse']}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
