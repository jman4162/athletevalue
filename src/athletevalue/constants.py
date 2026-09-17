"""Structural and numerical constants.

These are facts about the sport or about floating point, not modelling choices.
Modelling choices live in the assumption registry with a citation; the
governance tests fail if one appears as a bare literal in a modelling package.
"""

from __future__ import annotations

PLAYERS_ON_COURT = 5
"""Players per side on the floor in basketball."""

PER_100 = 100.0
"""Efficiency ratings are expressed per 100 possessions."""

FIRST_NIL_SEASON = 2022
"""NCAA interim NIL policy took effect 2021-07-01, so 2021-22 is the first NIL season."""

FIRST_REVENUE_SHARE_SEASON = 2026
"""House v. NCAA revenue sharing began 2025-07-01, so 2025-26 is the first season."""

EVEN_WIN_PCT = 0.5
"""A .500 record: the centre of the win-percentage scale, used where a team has no
rated opponents to average over."""

EPS = 1e-12
"""Guard against division by zero in rate calculations."""

DECILES = 10
"""Bins in a decile table."""
