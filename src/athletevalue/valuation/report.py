"""Plain-text rendering of a valuation."""

from __future__ import annotations

from athletevalue.schemas.estimate import Estimate
from athletevalue.schemas.evidence import EvidenceStatus
from athletevalue.valuation.result import PlayerValuation

_LABEL_WIDTH = 22
_VALUE_WIDTH = 14


def money(value: float) -> str:
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    if magnitude >= 1e6:
        return f"{sign}${magnitude / 1e6:.2f}M"
    if magnitude >= 1e3:
        return f"{sign}${magnitude / 1e3:.0f}k"
    return f"{sign}${magnitude:.0f}"


def _format(estimate: Estimate, value: float) -> str:
    if estimate.unit == "USD":
        return money(value)
    if estimate.unit == "wins":
        return f"{value:.1f}"
    return f"{value:+.1f}"


def _line(label: str, estimate: Estimate | None, suffix: str = "") -> str:
    if estimate is None:
        return f"{label:<{_LABEL_WIDTH}}{'n/a':>{_VALUE_WIDTH}}"
    point = _format(estimate, estimate.value) + suffix
    interval = f"{_format(estimate, estimate.lower)} to {_format(estimate, estimate.upper)}"
    status = f"{estimate.status.glyph} {estimate.status.value}"
    return (
        f"{label:<{_LABEL_WIDTH}}{point:>{_VALUE_WIDTH}}   "
        f"{estimate.level:.0%}: {interval:<24}{status}"
    )


def render_summary(v: PlayerValuation) -> str:
    p = v.player
    season = f"{p.season - 1}-{str(p.season)[-2:]}"
    role = f" · {v.role}" if v.role else ""
    lines = [
        f"{p.name} · {p.team} ({v.conference}) · {season}{role}",
        "",
        _line("Athletic impact", v.athletic_impact, " /100"),
        f"{'  offense / defense':<{_LABEL_WIDTH}}{v.offense.value:>+7.1f} / {v.defense.value:+.1f}",
        _line("Wins above replacement", v.war),
        _sensitivity_line(v),
        _line("Program value", v.program_value),
    ]
    for name, component in v.program_value_components.items():
        lines.append(_line(f"  {name}", component))
    if v.program_value_two_season is not None:
        lines.append(_line("  with next season", v.program_value_two_season))
    lines.append(_line("Roster market value", v.roster_market_value))
    if v.allocated_market_value is not None:
        lines.append(_line("  allocation", v.allocated_market_value))
    if v.observed_price is not None:
        lines.append(_line("Disclosed pay", v.observed_price))
    lines.append(_line("Surplus", v.surplus))
    if v.quadrant:
        label = v.quadrant.replace("_", " ")
        lines.append(
            f"{'Among paid teammates':<{_LABEL_WIDTH}}{label:>{_VALUE_WIDTH}}"
            "   (relative to team medians)"
        )
    lines += ["", "Drivers"]
    lines += [f"{d.sign} {d.text}" for d in v.drivers]
    if v.warnings:
        lines += ["", "Warnings"]
        lines += [f"! {w}" for w in v.warnings]
    legend = " ".join(
        f"{s.glyph} {s.value}" for s in EvidenceStatus if s is not EvidenceStatus.UNRESOLVED
    )
    lines += [
        "",
        f"Price basis: {v.price_basis or 'none'}. Status: {legend}",
        f"As of {v.as_of.isoformat()} · games through {v.data_through.isoformat()} "
        f"· model {v.model_version}",
        DISCLAIMER,
    ]
    return "\n".join(lines)


DISCLAIMER = (
    "Estimates, not reports of pay. An allocated market value spreads a published "
    "conference-tier budget by role and rating; it says nothing about this player's contract."
)


def _sensitivity_line(v: PlayerValuation) -> str:
    others = [
        f"{name.replace('_', ' ')} {v.war_sensitivity[name]:.1f}"
        for name in v.war_sensitivity
        if name != v.replacement_definition
    ]
    return (
        f"{'  replacement':<{_LABEL_WIDTH}}{v.replacement_definition.replace('_', ' ')} "
        f"({v.replacement_level:+.1f}/100); under {' / '.join(others)}"
    )
