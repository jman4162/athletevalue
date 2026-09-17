"""The version string is written in six files; a release must not leave one behind.

``release.yml`` checks the git tag against ``pyproject.toml``. Nothing checked the
other copies, so 0.4.0 shipped with a stale literal in the golden fixture. Each
assertion below names the file a reader has to edit.

A ``.devN`` suffix on ``MODEL_VERSION`` marks a tree between releases, where the
documented numbers still describe the last release. The checks that compare the
documents and the changelog therefore run only once the suffix is gone, which is
the state every tag is cut from.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

from athletevalue.versions import MODEL_VERSION, __version__

ROOT = Path(__file__).resolve().parents[2]


DEV_BUILD = ".dev" in MODEL_VERSION
released_only = pytest.mark.skipif(
    DEV_BUILD, reason=f"{MODEL_VERSION} is a development build; documents describe the last release"
)


def _pyproject_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        version: str = tomllib.load(handle)["project"]["version"]
    return version


def test_pyproject_matches_the_package():
    assert _pyproject_version() == __version__, "pyproject.toml and versions.py disagree"


def test_model_version_tracks_the_package_version():
    """``mbb-v<version>``, where a ``.devN`` build may already name the next release.

    Between releases the model version leads: bumping it is what invalidates derived
    caches when a modelling change lands, and that happens before the version bump.
    A released build must name exactly the version it ships as.
    """
    match = re.fullmatch(r"mbb-v(?P<base>\d+\.\d+\.\d+)(?P<dev>\.dev\d+)?", MODEL_VERSION)
    assert match is not None, f"MODEL_VERSION {MODEL_VERSION!r} is not mbb-vX.Y.Z[.devN]"
    base = tuple(int(part) for part in match.group("base").split("."))
    shipped = tuple(int(part) for part in __version__.split("."))
    if match.group("dev"):
        assert base >= shipped, f"{MODEL_VERSION} is behind version {__version__}"
    else:
        assert base == shipped, "MODEL_VERSION and __version__ disagree"


@released_only
def test_citation_file_matches():
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r"^version:\s*(\S+)\s*$", text, re.MULTILINE)
    assert match is not None, "CITATION.cff has no version field"
    assert match.group(1).strip("\"'") == __version__


@released_only
def test_zenodo_metadata_matches():
    payload = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    assert payload["version"] == __version__, ".zenodo.json carries a stale version"


@released_only
def test_changelog_leads_with_this_version():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(r"^## (\S+)", text, re.MULTILINE)
    assert match is not None, "CHANGELOG.md has no version heading"
    assert match.group(1) == __version__, "CHANGELOG.md does not lead with this version"


@released_only
@pytest.mark.parametrize("document", ["README.md", "METHODOLOGY.md"])
def test_documents_name_the_current_model_version(document: str):
    text = (ROOT / document).read_text(encoding="utf-8")
    named = set(re.findall(r"mbb-v\d+\.\d+\.\d+(?:\.dev\d+)?", text))
    assert named, f"{document} does not name a model version"
    assert named == {MODEL_VERSION}, f"{document} names {sorted(named)}, not {MODEL_VERSION}"
