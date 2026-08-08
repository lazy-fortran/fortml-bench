#!/usr/bin/env python3
"""Fetch the reference material listed in `sources.py` into `provenance/`.

Reproducible and resumable: an entry already present is not re-fetched unless
`--force` is given, so running this after adding one source costs one download
rather than all of them.

A manifest records what was fetched, when, and from where. Without it a claim
of the form "this matches the paper" is uncheckable six months later, because
nobody can tell which version of the paper was read.

Nothing fetched here is compiled, vendored, or ported. FortML implements against
the definitions, it does not port the references.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from literature_sources import PAPERS, REPOSITORIES, Paper, Repository  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROVENANCE = ROOT / "literature"
PAPER_DIR = PROVENANCE / "papers"
CODE_DIR = PROVENANCE / "codes"
MANIFEST = PROVENANCE / "MANIFEST.json"

#: arXiv asks for a descriptive agent and rate limiting. Both are courtesy that
#: costs nothing and getting blocked mid-fetch costs a re-run.
USER_AGENT = "fortml-bench literature fetcher (+https://github.com/lazy-fortran)"
TIMEOUT_SECONDS = 120


def log(message: str) -> None:
    print(message, flush=True)


def fetch_paper(paper: Paper, force: bool) -> dict:
    """Download one arXiv PDF, and extract text if a extractor is available."""
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    pdf = PAPER_DIR / f"{paper.key}-{paper.arxiv_id}.pdf"
    record: dict = {
        "kind": "paper",
        "key": paper.key,
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "needed_for": paper.needed_for,
        "consumed_by": list(paper.consumed_by),
        "path": str(pdf.relative_to(PROVENANCE)),
    }

    if pdf.exists() and not force:
        record["status"] = "cached"
        record["bytes"] = pdf.stat().st_size
        log(f"  {paper.key}: cached ({pdf.stat().st_size // 1024} KiB)")
        return record

    url = f"https://arxiv.org/pdf/{paper.arxiv_id}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as error:
        # A failed fetch is recorded rather than raised: one unreachable source
        # should not cost the whole run, and the manifest then says which.
        record["status"] = "failed"
        record["error"] = str(error)
        log(f"  {paper.key}: FAILED ({error})")
        return record

    if not payload.startswith(b"%PDF"):
        record["status"] = "failed"
        record["error"] = "response was not a PDF"
        log(f"  {paper.key}: FAILED (not a PDF)")
        return record

    pdf.write_bytes(payload)
    record["status"] = "fetched"
    record["bytes"] = len(payload)
    record["url"] = url
    log(f"  {paper.key}: fetched ({len(payload) // 1024} KiB)")

    text = extract_text(pdf)
    if text is not None:
        record["text"] = str(text.relative_to(PROVENANCE))
        # A wrong identifier still returns a perfectly valid PDF, so the only
        # way to know the right paper arrived is to look at it. Checking a few
        # distinctive title words against the opening page catches a
        # transcribed or guessed id, which is otherwise discovered much later
        # by someone reading the "wrong" paper and concluding the code is wrong.
        matched, opening = title_matches(paper.title, text)
        record["title_verified"] = matched
        if not matched:
            record["status"] = "mismatched"
            record["error"] = (
                "the downloaded document does not look like the stated title; "
                "opening text: " + opening
            )
            log(f"    TITLE MISMATCH: got {opening!r}")
    return record


def title_matches(expected: str, text: Path) -> tuple[bool, str]:
    """Whether the fetched document's opening page carries the expected title.

    Deliberately loose: PDFs break titles across lines and mangle ligatures, so
    an exact match would reject correct fetches. Requiring most of the
    distinctive words is enough to separate the right paper from a different
    one.
    """
    try:
        opening = text.read_text(errors="replace")[:4000]
    except OSError:
        return True, ""
    flattened = " ".join(opening.lower().split())
    words = [w for w in expected.lower().split() if len(w) > 3]
    if not words:
        return True, ""
    hits = sum(1 for word in words if word in flattened)
    preview = " ".join(opening.split())[:120]
    return hits >= max(1, (2 * len(words)) // 3), preview


def extract_text(pdf: Path) -> Path | None:
    """Extract a PDF to text so it can be searched and read without a viewer.

    Optional: if no extractor is installed the PDF is still usable, so a missing
    tool is reported and skipped rather than treated as a failure.
    """
    target = pdf.with_suffix(".txt")
    for command in (["pdftotext", "-layout", str(pdf), str(target)],):
        try:
            subprocess.run(command, check=True, capture_output=True)
            return target
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError as error:
            log(f"    text extraction failed: {error}")
            return None
    log("    no pdftotext available; PDF fetched without text extraction")
    return None


def fetch_repository(repository: Repository, force: bool) -> dict:
    """Shallow-clone one reference tree, sparsely where paths are declared."""
    CODE_DIR.mkdir(parents=True, exist_ok=True)
    target = CODE_DIR / repository.key
    record: dict = {
        "kind": "repository",
        "key": repository.key,
        "url": repository.url,
        "ref": repository.ref,
        "licence": repository.licence,
        "needed_for": repository.needed_for,
        "consumed_by": list(repository.consumed_by),
        "path": str(target.relative_to(PROVENANCE)),
    }

    if target.exists() and not force:
        record["status"] = "cached"
        record["commit"] = describe_commit(target)
        log(f"  {repository.key}: cached at {record['commit']}")
        return record

    if target.exists():
        subprocess.run(["rm", "-rf", str(target)], check=True)

    clone = [
        "git", "clone", "--depth", "1", "--branch", repository.ref,
        "--single-branch",
    ]
    if repository.sparse_paths:
        clone += ["--filter=blob:none", "--sparse"]
    clone += [repository.url, str(target)]

    try:
        subprocess.run(clone, check=True, capture_output=True, timeout=600)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        record["status"] = "failed"
        record["error"] = str(error)
        log(f"  {repository.key}: FAILED ({error})")
        return record

    if repository.sparse_paths:
        subprocess.run(
            ["git", "-C", str(target), "sparse-checkout", "set",
             *repository.sparse_paths],
            check=True, capture_output=True,
        )

    record["status"] = "fetched"
    # Pinned by commit, not by branch: a claim checked against "main" is
    # checked against nothing in particular.
    record["commit"] = describe_commit(target)
    log(f"  {repository.key}: fetched at {record['commit']}")
    return record


def describe_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    return result.stdout.strip() or "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="re-fetch entries that are already present",
    )
    parser.add_argument(
        "--papers-only", action="store_true",
        help="skip the source trees, which are much larger",
    )
    parser.add_argument(
        "--only", metavar="KEY", action="append", default=[],
        help="fetch just these keys; repeatable",
    )
    arguments = parser.parse_args()

    PROVENANCE.mkdir(parents=True, exist_ok=True)
    wanted = set(arguments.only)
    records = []

    log("papers:")
    for paper in PAPERS:
        if wanted and paper.key not in wanted:
            continue
        records.append(fetch_paper(paper, arguments.force))

    if not arguments.papers_only:
        log("source trees:")
        for repository in REPOSITORIES:
            if wanted and repository.key not in wanted:
                continue
            records.append(fetch_repository(repository, arguments.force))

    manifest = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Read, not ported. FortBO derives its closed forms through FortSym "
            "and writes its implementations against the definitions here."
        ),
        "entries": records,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    failed = [r for r in records if r.get("status") == "failed"]
    log(f"\n{len(records) - len(failed)} of {len(records)} available")
    if failed:
        log("failed: " + ", ".join(r["key"] for r in failed))
    # A partial fetch is still useful, so this is not an error exit unless
    # nothing at all arrived.
    return 0 if len(failed) < len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
