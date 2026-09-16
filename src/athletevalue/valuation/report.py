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
        _line("Program value", v.program_value),
    ]
    for name, component in v.program_value_components.items():
        lines.append(_line(f"  {name.replace('_', ' ')}", component))
    lines.append(_line("Roster market value", v.roster_market_value))
    if v.observed_price is not None:
        lines.append(_line("Disclosed pay", v.observed_price))
    lines.append(_line("Surplus", v.surplus))
    if v.quadrant:
        lines.append(
            f"{'Value vs. price':<{_LABEL_WIDTH}}{v.quadrant.replace('_', ' '):>{_VALUE_WIDTH}}"
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
    ]
    return "\n".join(lines)
