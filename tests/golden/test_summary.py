from __future__ import annotations

from datetime import date
from pathlib import Path

from athletevalue.schemas.estimate import Estimate
from athletevalue.schemas.evidence import EvidenceStatus
from athletevalue.schemas.identity import PlayerRef
from athletevalue.valuation.report import money
from athletevalue.valuation.result import Driver, PlayerValuation

GOLDEN = Path(__file__).with_name("summary.txt")


def _valuation() -> PlayerValuation:
    est = EvidenceStatus.ESTIMATED
    scen = EvidenceStatus.SCENARIO
    return PlayerValuation(
        player=PlayerRef(athlete_id="1", name="Test Player", team="State", season=2026),
        conference="Big Ten",
        role="starter",
        athletic_impact=Estimate.normal(
            6.0, 2.0, unit="points per 100 possessions", status=est, method="rapm_net"
        ),
        offense=Estimate.normal(
            4.0, 1.5, unit="points per 100 possessions", status=est, method="o"
        ),
        defense=Estimate.normal(
            2.0, 1.5, unit="points per 100 possessions", status=est, method="d"
        ),
        war=Estimate(
            value=3.2, lower=2.0, upper=4.5, unit="wins", status=est, method="war_simulated"
        ),
        war_linear=Estimate.normal(3.0, 0.8, unit="wins", status=est, method="war_analytic"),
        program_value=Estimate(
            value=450_000, lower=200_000, upper=800_000, unit="USD", status=scen, method="pv"
        ),
        program_value_components={
            "win_revenue": Estimate(
                value=420_000, lower=190_000, upper=760_000, unit="USD", status=est, method="w"
            ),
        },
        roster_market_value=Estimate(
            value=900_000, lower=600_000, upper=1_300_000, unit="USD", status=scen, method="a"
        ),
        observed_price=None,
        price_basis="allocation",
        surplus=Estimate(
            value=-450_000, lower=-900_000, upper=50_000, unit="USD", status=scen, method="s"
        ),
        quadrant="fair_star",
        drivers=[Driver(sign="+", text="on-court impact ranks 40 of 4000 rated players")],
        model_version="mbb-v0.1.0",
        as_of=date(2026, 9, 16),
        data_through=date(2026, 4, 6),
        sources=[],
        assumptions_used=["mbb.impact.ridge_lambda"],
        warnings=[],
    )


def test_summary_matches_golden_file():
    rendered = _valuation().summary()
    if not GOLDEN.exists():
        GOLDEN.write_text(rendered + "\n", encoding="utf-8")
    assert rendered + "\n" == GOLDEN.read_text(encoding="utf-8")


def test_json_round_trip():
    valuation = _valuation()
    assert PlayerValuation.model_validate_json(valuation.model_dump_json()) == valuation


def test_money_formatting():
    assert [money(v) for v in (950, 12_400, -725_000, 1_853_000)] == [
        "$950",
        "$12k",
        "-$725k",
        "$1.85M",
    ]
