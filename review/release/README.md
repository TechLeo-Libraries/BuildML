# Release review

Release confidence is supported by reproducible checks of documented workflows
and environments, with validation coverage and remaining limitations recorded
explicitly.

This is the maintainer and external-reviewer evidence package. The user serves
as the external human reviewer. Automated results and agent reviews inform that
decision; they do not record approval on the reviewer's behalf.

## Candidate identity

Run `python scripts/release_snapshot.py` after source changes finish. The output
under `artifacts/release-review/<source_sha256>/` contains a source archive and
per-file hashes. The base Git commit is context; the source digest identifies
the actual candidate, including uncommitted fixes. Run
`python scripts/release_snapshot.py --verify <manifest.json>` to detect changes.
An altered candidate requires a new snapshot and affected checks to be rerun.

For CI, the acceptance workflow records the exact checked-out commit and wheel
hash. Every matrix cell installs that same wheel outside the checkout in a new
environment. The release workflow requires full CI, the installation matrix,
and protected human approval before publishing that tested artifact. See
[release gate configuration](RELEASE_GATES.md). Local source results do not
substitute for pending matrix cells.

## Human acceptance record

Record the following against the candidate manifest, preserving the original
review notes and the disposition of each finding:

- Reviewer name and review date.
- Candidate source digest, commit when available, and distribution hashes.
- Claims accepted, challenged, narrowed, or excluded; see CLAIMS.md.
- Statistical review conclusions; see STATISTICAL_REVIEW.md.
- Installation matrix results and any unverified optional backends.
- Findings by severity, evidence, fix, independent recheck, and final disposition.
- Final decision: approve, approve with explicit limitations, or request changes.

The current human decision is **pending**. Approval must come from the reviewer.

## Release acceptance criteria

1. No unresolved critical or high-severity finding.
2. Every material public claim has evidence or an explicit scope limitation.
3. No unexplained failure in the advertised supported installation configurations.
4. Skips and unavailable dependencies remain unverified, not passed.
5. All distribution checks refer to the files intended for publication.
6. Human acceptance and any remaining limitations are recorded for that candidate.
7. After publication, verify the actual PyPI metadata and Read the Docs build.

The matrix covers core workflows; it does not establish that every optional
industry backend works on every platform. Expand optional-backend evidence
before expanding corresponding compatibility claims. The repository's docstring
budget also retains existing documentation debt; a passing ratchet is not a
zero-findings assessment.

## Review order

Start with the claims register and statistical findings. Reproduce the cases
that materially support the proposed use of BuildML as evidence. Inspect the
actual report outputs and uncertainty assumptions, then review installation
results and the candidate's changed files. Record disagreements as findings
with a reproduction or a precise unsupported claim.
