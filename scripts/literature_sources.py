"""The reference material FortML's models are built against.

Generic machine-learning and Gaussian-process literature. Anything
Bayesian-optimization specific belongs in `fortbo-bench` instead — the split
follows the code: FortML owns models and posteriors, FortBO owns acquisition
and policy.

Read, not ported. Every entry says what FortML needs from it and which file
consumes it, so an implementation traces back to the definition it was written
from rather than to somebody's recollection of it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Paper:
    key: str
    arxiv_id: str
    title: str
    needed_for: str
    consumed_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class Repository:
    key: str
    url: str
    ref: str
    licence: str
    needed_for: str
    sparse_paths: tuple[str, ...] = ()
    consumed_by: tuple[str, ...] = ()


PAPERS: tuple[Paper, ...] = (
    Paper(
        key="student-t-process",
        arxiv_id="1402.4306",
        title="Student-t Processes as Alternatives to Gaussian Processes",
        needed_for=(
            "The Student-t process: its predictive marginals, how the degrees "
            "of freedom update on conditioning, and — the part most easily got "
            "wrong from memory — which of its properties actually differ from a "
            "GP's rather than merely looking as though they should."
        ),
        consumed_by=("src/gp/fortml_student_t_process.f90",),
    ),
    Paper(
        key="svgp",
        arxiv_id="1309.6835",
        title="Gaussian Processes for Big Data",
        needed_for=(
            "The stochastic variational sparse GP, its ELBO, and the sense in "
            "which its predictive variance is not the exact posterior's."
        ),
        consumed_by=("src/gp/fortml_sparse_gp.f90",),
    ),
    Paper(
        key="ski",
        arxiv_id="1503.01057",
        title=(
            "Kernel Interpolation for Scalable Structured Gaussian Processes"
        ),
        needed_for=(
            "The structured kernel interpolation approximation and the "
            "conditions under which its error is controlled."
        ),
        consumed_by=("src/gp/fortml_ski_gp.f90",),
    ),
    Paper(
        key="spectral-mixture",
        arxiv_id="1302.4245",
        title="Gaussian Process Kernels for Pattern Discovery and Extrapolation",
        needed_for=(
            "The spectral mixture kernel's closed form and its parameterization, "
            "which FortML's derivative-observation path also has to differentiate."
        ),
        consumed_by=("src/gp/fortml_kernels.f90",),
    ),
)


REPOSITORIES: tuple[Repository, ...] = (
    Repository(
        key="gpytorch",
        url="https://github.com/cornellius-gp/gpytorch",
        ref="main",
        licence="MIT",
        needed_for=(
            "Reference kernel and likelihood definitions, and the numerical "
            "safeguards a mature GP library applies where the closed forms are "
            "ill-conditioned — which is where an independently written "
            "implementation is most likely to differ silently."
        ),
        sparse_paths=(
            "gpytorch/kernels",
            "gpytorch/likelihoods",
            "gpytorch/distributions",
        ),
        consumed_by=("src/gp/fortml_kernels.f90",),
    ),
)
