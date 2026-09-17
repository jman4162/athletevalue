"""Every cited constant must point at text a reader can find on the cited page."""

from __future__ import annotations

import html
import re

import httpx
import pytest

from athletevalue.assumptions.models import AssumptionBasis
from athletevalue.assumptions.registry import AssumptionRegistry

CITED = [
    a
    for a in AssumptionRegistry.load()
    if a.basis in (AssumptionBasis.PUBLISHED_RULE, AssumptionBasis.PUBLISHED_THIRD_PARTY)
]


def test_every_cited_assumption_quotes_its_source():
    missing = [a.assumption_id for a in CITED if not (a.citation and a.citation.quote)]
    assert not missing, f"citations without a verbatim quote: {missing}"


def test_quotes_are_short_passages():
    long = [a.assumption_id for a in CITED if len(a.citation.quote or "") > 300]
    assert not long, long


@pytest.mark.network
@pytest.mark.parametrize("assumption", CITED, ids=lambda a: a.assumption_id)
def test_quote_appears_on_the_cited_page(assumption):
    """Fetches the cited URL (or its archived copy) and looks for the quote.

    Pages that refuse automated readers are skipped, not passed: a skip is visible in
    the run, a false pass is not.
    """
    citation = assumption.citation
    urls = [_fetchable(u) for u in (citation.url, citation.archived_url) if u]
    text = None
    for url in urls:
        try:
            response = httpx.get(
                url, follow_redirects=True, timeout=30, headers={"User-Agent": "Mozilla/5.0"}
            )
        except httpx.HTTPError:
            continue
        if response.status_code == 200:
            text = response.text
            break
    if text is None:
        pytest.skip(f"{assumption.assumption_id}: no cited URL returned 200 to an automated reader")
    needle = _normalize(citation.quote)
    haystack = _normalize(text)
    assert needle in haystack, f"{assumption.assumption_id}: quote not found on {citation.url}"


def _fetchable(url: str) -> str:
    """GitHub blob pages render tables as HTML; the raw file holds the quoted text."""
    prefix = "https://github.com/"
    if url.startswith(prefix) and "/blob/" in url:
        owner_repo, _, path = url[len(prefix) :].partition("/blob/")
        return f"https://raw.githubusercontent.com/{owner_repo}/{path}"
    return url


def _normalize(text: str) -> str:
    """Tags out, entities unescaped, markdown escapes and typographic punctuation folded."""
    # Only tag-shaped tokens are removed; a bare "<" in prose must not swallow text.
    text = html.unescape(re.sub(r"<[A-Za-z/!][^<>]{0,300}>", "", text))
    for old, new in (
        ("\\", ""),
        ("\u2019", "'"),
        ("\u2018", "'"),
        ("\u201c", '"'),
        ("\u201d", '"'),
        ("\u2013", "-"),
        ("\u2014", "-"),
    ):
        text = text.replace(old, new)
    return " ".join(text.split())
