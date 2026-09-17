# Releasing

Releases publish from GitHub Actions with PyPI Trusted Publishing. No API token
exists on any machine or in repository secrets, so a release cannot go out under
the wrong PyPI account.

## One-time setup (PyPI owner: jman4162)

1. On pypi.org, logged in as `jman4162`: athletevalue → Manage → Publishing → add a
   GitHub publisher with
   - Owner: `jman4162`
   - Repository: `athletevalue`
   - Workflow: `release.yml`
   - Environment: `pypi`
2. On GitHub: Settings → Environments → create `pypi`. Optionally restrict it to
   tags matching `v*` and require a reviewer.

## Each release

1. Update `version` in `pyproject.toml`, `__version__` and `MODEL_VERSION` in
   `src/athletevalue/versions.py`, `version` in `CITATION.cff`, and `CHANGELOG.md`.
   Between releases, bump the `.devN` suffix of `MODEL_VERSION` whenever a change
   alters what a derived cache file contains: derived-file keys see the model
   version, settings and input digests, not code.
2. If upstream files were re-fetched for this release, run the live suite, check
   the documented numbers against that cache, and regenerate the pins with
   `uv run python scripts/pin_artifacts.py`.
3. Commit and push to `main`; wait for CI.
4. `git tag -a vX.Y.Z -m "athletevalue X.Y.Z"` and `git push origin vX.Y.Z`.
   The tag runs:
   - `regression.yml`: full-season fits against live data and every cited page;
   - `release.yml`: checks the tag matches the version, builds, rejects an email in
     the metadata, smoke-tests the wheel outside the checkout, and publishes.
5. Confirm the release on https://pypi.org/project/athletevalue/.

Yanking a bad release has no API; use the project's Manage → Releases page.
