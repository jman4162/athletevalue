from __future__ import annotations

import polars as pl
import pytest

from athletevalue.identity.names import normalize_name, similarity
from athletevalue.identity.resolve import AmbiguousPlayerError, PlayerNotFoundError, resolve_player


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Michigan St.", "Michigan State University"),
        ("St. John's (NY)", "Saint John's"),
        ("Fla. Atlantic", "Florida Atlantic University"),
    ],
)
def test_abbreviations_normalize_together(left, right):
    assert similarity(left, right) > 0.8


def test_normalize_drops_filler():
    assert normalize_name("University of the Pacific") == "pacific"


TABLE = pl.DataFrame(
    {
        "athlete_id": ["1", "2", "3"],
        "name": ["Jordan Smith", "Jordan Smith", "Kon Knueppel"],
        "team": ["Duke", "Iowa St.", "Duke"],
    }
)


def test_resolve_exact_fuzzy_and_ambiguous():
    assert resolve_player(TABLE, "kon knueppel")["athlete_id"] == "3"
    assert resolve_player(TABLE, "Kon Knuepel")["athlete_id"] == "3"
    with pytest.raises(AmbiguousPlayerError):
        resolve_player(TABLE, "Jordan Smith")
    assert resolve_player(TABLE, "Jordan Smith", team="Iowa State")["athlete_id"] == "2"
    with pytest.raises(PlayerNotFoundError):
        resolve_player(TABLE, "Nobody Atall")
