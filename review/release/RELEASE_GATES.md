# Release gates after COORD03

The Release workflow now calls full CI and the existing 12-configuration installed
artifact matrix in the same workflow run. Both must succeed before publication.
The candidate wheel and source archive are built once by the acceptance workflow;
publication downloads those artifacts, checks their SHA256 hashes and source
commit, checks their package metadata agrees, and requires the tag to equal
`v` followed by the package version. Publication no longer silently skips files
that already exist on PyPI.

The `pypi-release` GitHub Environment records human acceptance for the specific
workflow run. A preflight job queries its existing configuration before any job
requests that environment. Missing environments, inaccessible API responses,
missing required reviewers, self-review permission, and administrator bypass all
fail the gate. The publishing job checks the configuration again after approval.
Its environment link points to the run containing CI and candidate evidence.

## Configuration required before publication

On 2026-09-22 the GitHub `pypi-release` environment was created and its live API
response passed `check_release_gate.validate_environment`: LeonardLeo is the
required reviewer, self-review is prevented, administrator bypass is disabled,
and custom deployment policies allow `v2.*` and `v3.*` tags. Evidence is retained
in `artifacts/validation-gaps/release-environment.json`. On 2026-09-23 the PyPI
Trusted Publisher was registered and verified for `TechLeo-Libraries/BuildML`,
`release.yml`, and environment `pypi-release`. Additional external remediation
review was waived by the user; GitHub deployment protections remain active.
Candidate-specific deployment approval remains pending.

Required configuration and operating procedure:

1. Create `pypi-release` in repository Settings, Environments. Set the human
   reviewer as a required reviewer, enable Prevent self-review, and disable
   administrator bypass. Configure deployment tag restrictions for release tags.
2. Register a PyPI Trusted Publisher for this repository, `release.yml`, and the
   exact environment name `pypi-release`. Publishing uses OIDC; a repository API
   token is no longer a fallback.
3. Ensure the workflow token can read environment settings. API permission or
   plan restrictions fail closed and must be resolved before publishing.
4. Have another authorized account initiate the release if the designated human
   reviewer will approve it. Prevent self-review prohibits approving one's own run.
5. Review the exact commit, candidate hashes, all CI jobs and matrix evidence in
   the run before approving its deployment. Earlier review acceptance does not
   approve a changed candidate. Repository administrators should restrict workflow
   edits and protect release tags; these controls do not prevent an administrator
   from deliberately changing repository policy itself.

Dry runs build and test without requesting publishing approval. They do not
establish that remote environment configuration or PyPI publisher setup works.
Non-publishing CI and installation-matrix runs have been executed; their results
must be checked against the exact candidate commit. No publication was performed.

## Verification

`tests/unit/test_release_gate.py` covers accepted/rejected environment settings,
commit and tag mismatches, altered archives and wheel/source version disagreement.
Local tests establish the validator behavior; successful remote CI and a new
candidate-specific human acceptance remain release requirements.

GitHub references: [environment REST API](https://docs.github.com/en/rest/deployments/environments),
[deployment protections](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments),
[reviewing deployments](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/review-deployments).
