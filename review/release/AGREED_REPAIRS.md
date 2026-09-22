# Agreed review repairs

The human and coordinator reviews agreed on three release findings. This
revision addresses those findings and preserves the original review records
under `artifacts/coordinator-review/` and the earlier frozen candidate under
`artifacts/release-review/`.

| Finding | Correction | Regression evidence |
| --- | --- | --- |
| COORD01: changed alpha relabeled stored intervals | Native conformal and modern MAPIE reject unsupported alpha changes before prediction or evaluation; posterior-standard-deviation intervals can recompute, and legacy MAPIE still receives alpha | `test_probabilistic_reporting_acceptance.py`, `test_mapie_alpha_contract.py` |
| COORD02: evaluation overstated holdout independence | Results name the selected population, mark training-inclusive scores as diagnostic, retain interval warnings/disclosures, and require original-data provenance to establish independence | `test_probabilistic_reporting_acceptance.py` |
| COORD03: publication lacked full-CI and human gates | Same-run full CI and installed-artifact matrix precede protected human approval; commit, archive hashes, versions and tag are checked; existing uploads are no longer silently skipped | `test_release_gate.py`, [configuration](RELEASE_GATES.md) |

The independent reviewer reproduced an additional modern MAPIE path during the
recheck. Its correction received a second review. Final independent validation
passed 43 tests across the reporting, MAPIE-contract and release-gate files,
with no remaining actionable finding within these three repairs. The report is
`artifacts/agreed-repairs/INDEPENDENT_RECHECK.md`. MAPIE contract tests use stub
estimators; they do not establish real optional-backend installation coverage.

The installed-wheel runner now also checks alpha rejection and diagnostic
population reporting for regression and classification. Build, installation,
full-suite and documentation logs are retained in `artifacts/agreed-repairs/`.
The content-addressed source snapshot and evidence binding identify the final
candidate after validation; the previous snapshot remains historical evidence.

Publication still requires remote environment and Trusted Publisher setup,
successful CI/matrix execution, and human acceptance of the changed candidate.
Local artifacts keep version 2.6.2 for validation only; they cannot replace an
already published version. Hosted PyPI and Read the Docs content must be checked
after an approved new release. No publication or external approval is implied.
