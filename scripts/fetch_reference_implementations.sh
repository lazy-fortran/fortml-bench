#!/usr/bin/env bash
# Record and fetch the third-party implementations of the scalable-GP methods
# reviewed by Liu, Ong, Shen and Cai (IEEE TNNLS 31(11):4405-4423, 2020).
#
# Everything lands in the ignored .provenance/ tree: shallow clones, the recorded
# revision of each, and a checksum manifest. Nothing fetched here is linked into
# the MIT Fortran libraries or redistributed; the clones exist so a claim about
# what another package does can be checked against its source rather than against
# memory.
#
# The paper itself is not redistributable. Its DOI and bibliographic record are
# written to the manifest instead.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${root}/.provenance/reference_implementations"
manifest="${target}/MANIFEST.txt"
mkdir -p "${target}"

: >"${manifest}"
{
    echo "# Reference implementations of the scalable GPs reviewed in"
    echo "# H. Liu, Y.-S. Ong, X. Shen, J. Cai, \"When Gaussian Process Meets Big"
    echo "# Data: A Review of Scalable GPs\", IEEE TNNLS 31(11):4405-4423, 2020,"
    echo "# doi:10.1109/TNNLS.2019.2957109 (not redistributable; cited only)."
    echo "# Fetched: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
} >>"${manifest}"

# name|url|methods it implements, per the review's Table I
repositories=(
    "GPy|https://github.com/SheffieldML/GPy|VFE, SPEP, SKI, SVGP"
    "gpstuff|https://github.com/gpstuff-dev/gpstuff|SoR, DTC, FITC, VFE, SVGP, CS, PIC, FIC"
    "GPflow|https://github.com/GPflow/GPflow|FITC, VFE, SVGP"
    "gpytorch|https://github.com/cornellius-gp/gpytorch|SKI, deep kernel learning"
    "pymc|https://github.com/pymc-devs/pymc|DTC, FITC, VFE"
    "keops|https://github.com/getkeops/keops|matrix-free kernel reductions"
    "AugmentedGaussianProcesses.jl|https://github.com/theogf/AugmentedGaussianProcesses.jl|VFE, SVGP"
)

for entry in "${repositories[@]}"; do
    IFS='|' read -r name url methods <<<"${entry}"
    destination="${target}/${name}"
    if [ -d "${destination}/.git" ]; then
        git -C "${destination}" fetch --depth 1 origin HEAD >/dev/null 2>&1 || true
        git -C "${destination}" reset --hard FETCH_HEAD >/dev/null 2>&1 || true
    else
        rm -rf "${destination}"
        if ! git clone --depth 1 "${url}" "${destination}" >/dev/null 2>&1; then
            printf '%s\tCLONE FAILED\t%s\t%s\n' "${name}" "${url}" "${methods}" \
                >>"${manifest}"
            continue
        fi
    fi
    revision="$(git -C "${destination}" rev-parse HEAD)"
    printf '%s\t%s\t%s\t%s\n' "${name}" "${revision}" "${url}" "${methods}" \
        >>"${manifest}"
done

# The MATLAB and R packages of Table I are distributed as archives rather than
# as public git repositories. Record where they live; do not download binaries.
{
    echo
    echo "# Not cloned (archive distributions), recorded by location:"
    echo "GPML	http://www.gaussianprocess.org/gpml/code/matlab/doc/	FITC, VFE, SPEP, SKI"
    echo "pyGPs	https://github.com/PMBio/pygp	FITC"
    echo "laGP	http://bobby.gramacy.com/r_packages/laGP/	NeNe"
    echo "GPLP	http://www.jmlr.org/mloss/	NeNe, PoE, DDM, PIC"
} >>"${manifest}"

echo "wrote ${manifest}"
cat "${manifest}"
