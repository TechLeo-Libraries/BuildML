# Candidate evidence summary

This package supports the external human review. It does not record that review
as approved. Start with README.md, CLAIMS.md and STATISTICAL_REVIEW.md here.

## Current checks

The table below describes the earlier frozen candidate. The subsequent agreed
COORD01–03 repairs and independent recheck are documented in
[AGREED_REPAIRS.md](AGREED_REPAIRS.md); their new build and validation evidence
live under `artifacts/agreed-repairs/`. Do not apply the earlier candidate's
hashes or approval to the updated source.

| Check | Result | Evidence |
| --- | --- | --- |
| Claim references | 49 claim groups; references resolve | `scripts/check_release_claims.py` |
| Independent statistical review | Five findings with verified dispositions | `STATISTICAL_REVIEW.md` |
| Conformal, probabilistic and teaching regressions | 34 passed; 3 optional skips | `artifacts/probabilistic-acceptance.log` |
| EDA and drift-provenance regressions | 12 passed | `artifacts/release-review/eda-tests.log` |
| Fresh installed wheel, Windows / Python 3.12 | Six checks passed; no inherited site packages | `artifacts/release-review/fresh-install.json` |
| Fresh core dependency resolution | Install succeeded; pip check passed | `artifacts/release-review/fresh-install.log` |
| Wheel and source archive | Built; required contents and twine checks passed | `artifacts/release-review/dist/` |
| Teaching API contracts | 455 calls; zero errors | `artifacts/release-review/teaching-contracts.log` |
| Copy, docstring and typing conventions | Copy lint and docstring ratchet passed; Ruff passed | `artifacts/release-review/` |

The earlier full unit/integration run and its repaired wording failure are
documented separately in CLAIMS.md. Those results predate the newest conformal
and drift-provenance changes; the scoped reruns above cover those changes.

## Pending acceptance

- Run the 12-cell core installation matrix on the committed candidate. The
  workflow is implemented and publication depends on its success, but local
  workflow syntax validation is not remote execution evidence.
- Record the human methodology and claim-level decisions for the candidate.
- Expand optional-backend evidence for any optional capability central to the
  intended expert submission. A core installation pass does not cover extras.
- Confirm hosted PyPI and Read the Docs content after an approved publication.

No version bump, commit, push or publication is performed by this local review
package. The candidate archive identifies the working tree by content hash;
the base Git commit alone does not identify the uncommitted repairs.

## Reproducibility

The content-addressed candidate folder under `artifacts/release-review/` contains
the source manifest, source archive and an evidence binding. The binding records
distribution and evidence-file hashes so reviewers can distinguish an exact
tested artifact from an earlier build. Re-run source verification before using
the results for a later candidate. Preserve findings and resolved reproductions
alongside the final human decision.
