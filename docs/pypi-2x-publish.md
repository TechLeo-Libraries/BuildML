# Release maintenance

This page is for BuildML maintainers who build and publish releases.
For installation, see the [installation guide](installation.rst).
For supported APIs, see the [stability policy](stability.md).

## Release history

BuildML 2.6.3 is the release documented by this checkout. Releases
`2.4.0a3`, `2.4.0`, `2.5.0`, and `2.6.0` were yanked because their wheels
omitted `operation_index.json` and could not import. Version `2.6.1` corrected
the packaging; `2.6.2` includes dependency and backend compatibility fixes.
The legacy `1.0.9` line remains available by an explicit version pin.

Verify release status using [PyPI metadata](https://pypi.org/pypi/buildml/json).
The `a3` suffix identifies a pre-release, which pip does not normally select
when a compatible stable release satisfies the request.

## Prepare a release

1. Update `buildml/_version.py` and `pyproject.toml` to the same new version.
2. Move the relevant changelog entries from `Unreleased` to a dated release.
3. Run the checks in [CONTRIBUTING.md](https://github.com/TechLeo-Libraries/BuildML/blob/main/CONTRIBUTING.md),
   including teaching sync, documentation checks, tests, and a Sphinx build.
4. Build artifacts in a clean output directory for that version:

   ```bash
   python -m build --outdir artifacts/release-candidate
   python scripts/check_wheel_contents.py artifacts/release-candidate
   python -m twine check artifacts/release-candidate/*
   ```

5. Install the wheel into a clean environment outside the source checkout.
   Verify `import buildml` and run a representative Session example.
6. Inspect the wheel's README metadata and rendered documentation. Updating
   repository prose does not update artifacts previously uploaded to PyPI.

Use a new output directory for each candidate; the build command does not
remove older artifacts. Do not upload files from several versions together.

## Publish with the configured workflow

`.github/workflows/release.yml` builds on version-tag pushes, published GitHub
releases, and manual dispatch. Manual dispatch defaults to a build-only dry run:

```bash
gh workflow run release.yml --ref main -f dry_run=true
```

The workflow requires full CI and the 12-configuration artifact acceptance matrix
for the selected commit. Publication requires a matching version tag and human
approval through the protected `pypi-release` GitHub Environment. The downloaded
candidate hashes, source commit and package versions are checked before upload.
Existing PyPI distributions are not silently skipped or replaced.

Publishing uses OIDC Trusted Publishing. Register owner `TechLeo-Libraries`,
repository `BuildML`, workflow `release.yml`, and environment `pypi-release` in
PyPI. Configure required reviewers, prevent self-review and disable administrator
bypass in the GitHub Environment before attempting publication. The workflow
fails closed if it cannot confirm those protections. See the
[release gate setup](https://github.com/TechLeo-Libraries/BuildML/blob/main/review/release/RELEASE_GATES.md)
for the complete configuration and approval procedure.

## Verify publication

Check the new version's PyPI metadata, install it in a clean environment,
and run an import and Session smoke test. Verify the project description on
PyPI and the corresponding Read the Docs build. Update version statements
only after checking the published state.

An earlier upload can exist even if a later workflow attempt fails. Inspect
PyPI metadata and file hashes to distinguish a publishing failure from a
failed duplicate upload or authentication attempt.
