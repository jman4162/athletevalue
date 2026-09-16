# athletevalue

Open models that estimate, separately, how much a college basketball player
improves his team, what that improvement is worth to his program, and what the
roster market would pay him.

Rating sites publish one "NIL value" per athlete from a model nobody outside can
inspect. That single number mixes on-court impact, school revenue-sharing budgets,
collective payments and endorsement income. `athletevalue` keeps those apart,
attaches an interval and an evidence status to each, and cites every constant it
uses. The comparison between value and price is the point:

> In 2025-26 the model prices Michigan's Yaxel Lendeborg at $1.85M and credits his
> wins with $1.11M of program revenue. It prices Gonzaga's Graham Ike at $354k and
> credits him with $542k. Both medians carry wide intervals.

Version 0.1 covers NCAA Division I men's basketball.

## Install

```bash
pip install athletevalue          # or: uv add athletevalue
```

Python 3.12 or later. The first valuation downloads about 400 MB of public data
into a local cache and parses 15 years of EADA spreadsheets, which takes several
minutes. Later runs read the cache and finish in seconds. `athletevalue cache-path`
prints the cache location.

## Quick start

```bash
athletevalue value "Yaxel Lendeborg" --season 2026
```

```text
Yaxel Lendeborg · Michigan (Big Ten) · 2025-26 · starter

Athletic impact           +15.9 /100   80%: +10.8 to +20.9          ▲ estimated
  offense / defense      +8.9 / +7.0
Wins above replacement           8.1   80%: 5.5 to 11.1             ▲ estimated
Program value                 $1.11M   80%: $567k to $1.90M         ○ scenario
  win revenue                 $1.03M   80%: $522k to $1.70M         ▲ estimated
  bid revenue                   $28k   80%: $2k to $242k            ▲ estimated
  tournament units               $6k   80%: $985 to $47k            ○ scenario
Roster market value           $1.85M   80%: $1.31M to $2.53M        ○ scenario
Surplus                       -$725k   80%: -$1.58M to $232k        ○ scenario
Value vs. price            fair star

Drivers
+ on-court impact ranks 1 of 4030 rated players
+ on the floor for 78% of team possessions
~ team went 37-3, net rating +37.4
+ program reports $21.02M basketball revenue (D1 median $3.04M)
+ Big Ten roster budget tier: power

Price basis: allocation. Status: ● reported ◆ derived ▲ estimated ○ scenario
As of 2026-09-16 · games through 2026-04-06 · model mbb-v0.1.0
```

```python
from athletevalue import api

valuation = api.value_player("Yaxel Lendeborg", season=2026)
print(valuation.summary())
valuation.war.value, valuation.war.lower, valuation.war.upper

table = api.value_team("Gonzaga", season=2026)  # polars DataFrame, one row per player
```

Other commands:

| Command | What it does |
| --- | --- |
| `athletevalue fit --season 2026 [--cv]` | Fit RAPM and print the top players |
| `athletevalue validate --season 2026` | Check the fit against published references |
| `athletevalue team "Duke" --season 2026` | Value vs. price for a roster |
| `athletevalue assumptions` | List every model constant, its basis and status |
| `athletevalue fetch --season 2026` | Download inputs without fitting |

Seasons are keyed by ending year: 2026 is the 2025-26 season.

## What each number means

| Output | Question | Method |
| --- | --- | --- |
| Athletic impact | Points per 100 possessions a player adds over an average D1 player | Possession-weighted ridge regression (RAPM) on every lineup |
| Wins above replacement | Wins the team gains versus a replacement-level player | Per-game win model over the team's actual schedule |
| Program value | Revenue those wins bring the school, this season and next | School fixed-effects model of EADA revenue, plus tournament bids |
| Roster market value | What the 2025-26 market would plausibly pay | Reported roster budgets split by role and impact |
| Surplus | Program value minus price | Difference of the two simulations |

Every estimate carries an 80% interval and a status:

- ● **reported**: stated by a primary source.
- ◆ **derived**: computed from reported data by a fixed procedure.
- ▲ **estimated**: depends on a statistical fit or a published third-party figure.
- ○ **scenario**: depends on an assumption a user should question, such as how a
  conference splits tournament money.

A result takes the status of its weakest input. Market value is always a scenario
in v0.1 because no public dataset records individual college basketball pay.

## Validation

Each season fit is checked against the SportsDataverse league-wide RAPM, which is
fit on the same possessions, and against Bart Torvik's team ratings, which are not.

| Gate | 2025 | 2026 | Threshold |
| --- | --- | --- | --- |
| Spearman vs. reference RAPM, players with 500+ possessions | 0.963 (2,817) | 0.975 (3,036) | ≥ 0.90 |
| Spearman of team net rating vs. Torvik AdjOE − AdjDE | 0.946 (321 teams) | 0.963 (324 teams) | ≥ 0.93 |
| Intercept, points per 100 | 104.3 | 106.4 | 95–112 |
| Home-court advantage per side, per 100 | 3.18 | 2.91 | 1–4 |
| Residual variance | 13,313 | 13,328 | 11,000–15,000 |

Grouped cross-validation on the 2026 season picks a penalty of λ = 1000, the value
the reference model uses. Run `athletevalue validate` to reproduce.

## Data and licensing

| Source | Used for | Terms |
| --- | --- | --- |
| [SportsDataverse releases](https://github.com/sportsdataverse/sportsdataverse-data/releases) | Possessions, schedules, rosters, reference RAPM | MIT-licensed repository |
| [EADA](https://ope.ed.gov/athletics/) | Men's basketball revenue by school, 2011-2025 | U.S. government work |
| [Bart Torvik](https://barttorvik.com/) team results | Validation only | Not bundled, not used to fit |
| Press and methodology pages | Roster budgets, tournament unit value, conventions | Cited facts, see `athletevalue assumptions` |

Code is MIT. Curated data in `src/athletevalue/data` (the team crosswalk and the
deal registry) is CC BY 4.0; see [LICENSE-DATA](LICENSE-DATA). Data downloaded at
runtime stays under its own terms and is never redistributed.

KenPom and the Knight-Newhouse College Athletics Database restrict redistribution.
The package has a reader for a user's own KenPom export, for local comparison, and
an import-linter contract that stops fitting code from importing it.

## Limitations

- **Revenue is what schools attribute to men's basketball in EADA.** Conference
  media money, donations credited to the athletic department and brand effects are
  not in it, so program value is a lower bound on a player's value to the university.
  61% of D1 rows report revenue equal to expense and are excluded from the fit.
- **Market value is an allocation, not a prediction.** It spreads reported average
  roster budgets across a roster using reported pay ratios by role. It is only
  available for 2025-26, the season those figures describe.
- **College ratings are noisy.** A starter's net rating has a posterior SD near 4
  points per 100. Intervals are wide on purpose.
- **Players are rated within a season.** Transfers are not linked across seasons.
- **Tournament flags before 2023 come from matching ESPN games by date and team
  name.** A few First Four games each season go unmatched.

[the methodology](methodology.md) gives the equations, estimates and caveats for each layer.

## Roadmap

- **v0.2**: box-score prior for RAPM, multi-season ratings with recency weights,
  CollegeBasketballData for in-season updates, optional private KenPom calibration.
- **v0.3**: a fitted roster-market model once FY2026 NCAA financial reports (due
  January 2027) and deal-registry rows give it labels to learn from.
- **Later**: football, starting with quarterbacks.

## Contributing

See [CONTRIBUTING](contributing.md). Sourced deals for the registry are the most
useful contribution; the rules are in
[src/athletevalue/data/deal_registry/README.md](src/athletevalue/data/deal_registry/README.md).

## Citation

See [CITATION.cff](CITATION.cff). Estimates are research outputs; read
[DISCLAIMER.md](DISCLAIMER.md) before using them for decisions.
