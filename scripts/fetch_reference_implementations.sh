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
revision_file="${root}/reference_revisions.tsv"
mkdir -p "${target}"

if [ ! -f "${revision_file}" ]; then
    echo "missing pinned revision file: ${revision_file}" >&2
    exit 1
fi

: >"${manifest}"
{
    echo "# Reference implementations of the scalable GPs reviewed in"
    echo "# H. Liu, Y.-S. Ong, X. Shen, J. Cai, \"When Gaussian Process Meets Big"
    echo "# Data: A Review of Scalable GPs\", IEEE TNNLS 31(11):4405-4423, 2020,"
    echo "# doi:10.1109/TNNLS.2019.2957109 (not redistributable; cited only)."
    echo "# Fetched: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo -e "# name\trevision\tarchive_sha256\turl\tmethods"
    echo
} >>"${manifest}"

failures=0
while IFS=$'\t' read -r name revision expected_checksum url methods; do
    if [ -z "${name}" ] || [[ "${name}" == \#* ]]; then
        continue
    fi
    destination="${target}/${name}"
    if [ -d "${destination}/.git" ]; then
        if ! git -C "${destination}" fetch --depth 1 origin "${revision}" \
                >/dev/null 2>&1; then
            printf '%s\tFETCH FAILED\t%s\t%s\t%s\n' "${name}" "${revision}" \
                "${url}" "${methods}" >>"${manifest}"
            failures=1
            continue
        fi
    else
        if [ -e "${destination}" ]; then
            printf '%s\tTARGET IS NOT A GIT CLONE\t%s\t%s\t%s\n' "${name}" \
                "${revision}" "${url}" "${methods}" >>"${manifest}"
            failures=1
            continue
        fi
        git init --quiet "${destination}"
        git -C "${destination}" remote add origin "${url}"
        if ! git -C "${destination}" fetch --depth 1 origin "${revision}" \
                >/dev/null 2>&1; then
            printf '%s\tFETCH FAILED\t%s\t%s\t%s\n' "${name}" "${revision}" \
                "${url}" "${methods}" >>"${manifest}"
            failures=1
            continue
        fi
    fi
    git -C "${destination}" checkout --detach --force "${revision}" \
        >/dev/null 2>&1
    actual_revision="$(git -C "${destination}" rev-parse HEAD)"
    actual_checksum="$(git -C "${destination}" archive "${actual_revision}" | \
        sha256sum | awk '{print $1}')"
    if [ "${actual_revision}" != "${revision}" ] || \
            [ "${actual_checksum}" != "${expected_checksum}" ]; then
        printf '%s\tVERIFICATION FAILED\t%s\t%s\t%s\t%s\n' "${name}" \
            "${actual_revision}" "${actual_checksum}" "${url}" "${methods}" \
            >>"${manifest}"
        failures=1
        continue
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "${name}" "${actual_revision}" \
        "${actual_checksum}" "${url}" "${methods}" >>"${manifest}"
done <"${revision_file}"

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
exit "${failures}"
