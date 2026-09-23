# Validation gap repairs

The follow-up review exercised previously skipped optional workflows and strict
documentation references, and ran the existing candidate on GitHub Actions.
It found and repaired additional defects rather than treating local passes as
cross-platform evidence.

## Repairs

- Modern MAPIE regression jackknife-plus now uses leave-one-out cross-conformal
  prediction instead of the incompatible bootstrap estimator. Real installed
  MAPIE and NGBoost checks cover fitting, prediction, evaluation and persistence.
- Ordinary tests and internal dispatch use supported Session facades. Explicit
  legacy-format tests validate the expected upstream deprecation messages.
  Unknown labels are rejected before pandas categorical construction; Torch
  adapters avoid exposing read-only NumPy buffers to mutable tensors.
- Fold preprocessing implements sklearn estimator tags and cloning. Online
  passive-aggressive aliases use the supported equivalent on newer sklearn.
- The wheel provenance checker supports both standardized PEP 610 hash formats,
  and the matrix updates pip inside each fresh environment before installation.
  The SHA-256 comparison remains mandatory; recorded provenance is retained for
  diagnosing installation metadata mismatches.
- Copy inventory excludes generated directories relative to the repository root,
  fixing the Linux test failure when the checkout is inside a temporary directory.
- Cliff-walking examples use the current Gymnasium environment ID, while tests
  select the registered ID for the installed version.
- PuLP uses current variable and CBC discovery APIs with older-version
  compatibility, and refuses to label an unproven solution as exact. Real solves
  passed with warnings treated as errors on PuLP 2.7.0, 3.3.2 and the 4.0.0a12
  prerelease; that prerelease probe is not a production-version support claim.
- Documentation enables strict reference checking, removes blanket missing-link
  suppression, and provides verified build-local source links. A rendered-link
  audit checks actual output files and fragment anchors.

## Reproducible verification

Local evidence is retained under `artifacts/validation-gaps/`, including original
failures and their corrections. The broad Windows/Python 3.12 test run completed
with 1,336 passes, 151 skips and no emitted warnings. Skips still describe that
environment; separate dependency-equipped runs provide additional coverage.
Real MAPIE 1.5.0 and NGBoost 0.5.11 tests passed ten checks with one expected
absence-test skip. Strict HTML documentation built successfully, and 12,368
rendered local links were checked without a missing file or fragment.

Full CI now includes strict documentation and real probabilistic-backend jobs,
plus focused optional dependency groups. Each added optional job requires its
dependencies to import, checks dependency consistency and retains versions and
test results. The Torch anomaly placeholder is replaced by a real fit/evaluation
test. Imports alone do not establish every optional method's correctness.

## Release status

The GitHub release environment is configured and its protections were checked
through the live API. PyPI Trusted Publisher registration and candidate-specific
human approval remain separate requirements. No version bump or package
publication is part of these repairs. Remote CI results must be taken from the
exact candidate commit; earlier failed runs remain historical evidence.
