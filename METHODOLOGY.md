# Methodology

This document describes model `mbb-v0.3.1`. Every modelling constant named here is an
entry in the assumption registry (`athletevalue assumptions` lists them with their
basis, citation or rationale); a governance test fails on bare numbers in modelling
code. Figures quoted are from the 2025 and 2026 season fits on the data cached on
2026-09-17. Results reproduce to about 1e-15 relative, not bitwise, because the dense
Cholesky solve is multithreaded; ranks on near-ties can differ between runs.

## 1. Data

**Possessions.** SportsDataverse publishes one row per possession for every D1
men's game from 2011, parsed from stats.ncaa.org play-by-play. Each row names the
offense team, the points scored, a garbage-time flag and the stats.ncaa.org ids of
all ten players on the floor. For 2026 the file has 878,168 possessions; 840,040
remain after dropping garbage time and rows where a D1 team's lineup is incomplete.
The publishing repository is MIT-licensed; the producer repository that parses
stats.ncaa.org carries no license, and the NCAA has granted nothing. See
LICENSE-DATA.

**Non-D1 opponents.** Their lineups carry no player ids. Each non-D1 lineup enters
the model as one shared column, so those games still inform D1 players' ratings.

**Venues.** ESPN schedules mark neutral sites and NCAA tournament games
(`tournament_id` 22). From 2023 the possession file carries ESPN game ids. For
earlier seasons neutral-site and postseason ESPN games are matched to NCAA games
within one day using token similarity of both team names. Matching recovers 93-99%
of neutral-site games and 62-67 of 67 tournament games per season.

**Program finances.** EADA sport-level files give each institution's reported men's
basketball revenue (`REV_MEN`) and expense (`EXP_MEN`) by academic year. A curated
crosswalk maps 369 of 373 stats.ncaa.org team names to IPEDS unit ids. The service
academies are not in EADA, and New Haven joined D1 after the latest file.

## 2. Player impact (RAPM)

Possessions are grouped into rows that share a game, an offense team, both
lineups and a venue. For row *r* with offense lineup *O* and defense lineup *D*:

    y_r = 100 · pts_r / poss_r
        = μ + h·v_r + Σ_{i∈O} off_i − Σ_{j∈D} def_j + ε_r,   Var(ε_r) = σ² / poss_r

where *v_r* is +1 when the offense is at home, −1 away and 0 neutral. A positive
`def_j` means player *j* lowers opponent scoring, so a player's net rating is
`off + def`. Players with fewer than 100 average possessions share one pooled
offense and one pooled defense column.

Coefficients minimize

    Σ_r poss_r (y_r − x_r β)² + λ Σ_k p_k (β_k − b0_k)²

with *p_k* = 1 for player columns and 0 for the intercept, home court, pooled and
non-D1 columns. `b0` is a prior mean, described in section 2b. Weighting rows by
possession count makes `X'WX` identical to the possession-level Gram matrix, so λ
means the same thing as in a possession-level fit.

**Penalty without a prior.** λ = 1000, the value in the SportsDataverse model card.
Cross-validation that holds out whole games picks the same value for 2026 (held-out
weighted error 5,246 / 5,207 / 5,187 / 5,192 / 5,210 at λ = 100 / 300 / 1000 / 3000 /
10000).

**Uncertainty.** Posterior variances are `σ̂² · diag(A⁻¹)` with `A = X'WX + λP`,
from the Cholesky factor; `σ̂² = Σ poss_r r_r² / (n_rows − df)` with
`df = n_cols − λ Σ_penalized (A⁻¹)_kk`. Net-rating SEs include the offense-defense
covariance. Two caveats. First, the formula treats `b0` as known, and with the prior of
section 2b `b0` is built from the same season's team ratings; the intervals are
narrower than a full accounting would give. Second, the fit refuses to report a
result if `df` is not below the column count, which catches a dropped penalty but
not a wrong one. Without a prior a starter's net-rating SE is about 4 points per 100;
with it, about 2.6. Clustering by game does not widen the SEs (the design effect is
below 1 for the players checked).

**Team ratings.** A team's adjusted offense is the possession-weighted average of
the offensive coefficients it put on the floor; defense likewise. Their sum is the
team's adjusted net rating relative to an average D1 lineup.

**Checks.** See the validation table in the README. The reference-RAPM comparison is
a reimplementation check: both fits use the same possessions and penalty and shrink
toward zero, so it catches a design-matrix error, not a modelling one. The Torvik
comparison is external.

## 2b. Box-score prior

Shrinking every player toward zero treats a high-scoring starter and a walk-on the
same until possessions say otherwise. Since 0.2.0 the ridge instead shrinks toward
what a player's box score predicts, adjusted so each team's players add up to the
team's own rating.

**Box model.** For each player-season, 13 counting stats (points, field-goal,
three-point, free-throw, rim and mid-range attempts, assists, turnovers, offensive
and defensive rebounds, steals, blocks, fouls) become rates per 100 offensive
possessions after adding 100 possessions at the league-average rate. A
possession-weighted ridge regression on standardized rates predicts no-prior
offensive and defensive RAPM, separately. It is fitted on the four most recent
seasons **before** the target season, never on the target or any later season (for
2026: 2022-2025; for 2025: 2021-2024). The first season with data, 2011, has no
prior. In-sample R² is 0.33 for offense and 0.12 for defense; on the target season,
before the team adjustment, the model explains 33% and 13% of the no-prior ratings
of players with 500 or more possessions.

**Team adjustment.** Box rates ignore opponents, so a low-major star's rates
overstate him. As in Box Plus/Minus, each team's offensive and defensive
predictions are shifted by a constant so their possession-weighted sum equals the
team's rating from the no-prior fit. Without this step the 2026 team ratings match
Torvik at Spearman 0.925; with it, 0.971.

**Fit.** The adjusted predictions are the prior mean `b0`, with λ = 3000.
Cross-validation that holds out whole games, and rebuilds both the box totals and
the team adjustment from each fold's training games, gives:

| λ with prior | 1000 | 3000 | 10000 | 30000 |
| --- | --- | --- | --- | --- |
| 2026 held-out error | 5,182.4 | 5,177.6 | 5,178.1 | 5,178.7 |
| 2025 held-out error | 5,052.8 | 5,046.8 | 5,046.5 | 5,046.8 |

against 5,186.6 (2026) and 5,055.5 (2025) with no prior at λ = 1000. Two earlier
versions of this check leaked: 0.2.0 computed the team adjustment from the full
season inside cross-validation, and the fix in 0.2.0's validation gate still used
full-season box totals, so about a fifth of every prior feature was the held-out
target. The `--cv` path also used a full-season prior until 0.3.1, which made
held-out error fall without bound as λ grew. The numbers above and the
`prior_cv_error_ratio` gate use the leak-free procedure.

**What the box model contributes.** A prior with zero box information and the same
team adjustment reaches 5,182.1 (2026) and 5,050.6 (2025) at λ = 3000. The team
adjustment, which restores the scale of team strength that shrinkage toward zero
compresses, accounts for most of the gain and all of the improvement in agreement
with Torvik (slope of team ratings on Torvik AdjEM 0.81 without a prior, 0.96 with
either prior). The box model itself lowers held-out error by a further 0.1%. Its
clearest effect is on stability: on 1,735 players rated in both 2025 and 2026
(linked by the `person_id` column of the SportsDataverse reference release), the 2025
rating with the prior correlates 0.52 with the 2026 no-prior rating, against 0.40
without it. `scripts/returning_players.py` reproduces that check.

The no-prior fit is still computed and is what the reference-RAPM gate compares.
`athletevalue fit --no-prior` reproduces it.

## 3. Wins above replacement

A player's share of team possessions is `s = (off_poss + def_poss) / team lineup
possessions`. Replacing him with a replacement-level player lowers the team's net
rating by `Δ = (net − R) · s`.

**Replacement level.** Three definitions are computed and one is chosen by the
registry entry `mbb.wins.replacement_definition` (a scenario input):

| Definition | 2025 | 2026 | Basis |
| --- | --- | --- | --- |
| `nba_convention` | −2.0 | −2.0 | Box Plus/Minus constant, borrowed from the NBA |
| `pooled` | −9.3 | −12.1 | Fitted coefficient of players under the 100-possession threshold |
| `bench_median` (default) | −1.1 | −1.3 | Possession-weighted median of players ranked 9th-12th in playing time on their team |

The pooled players are the marginal D1 player in these data, but they are not who
replaces a rotation player when he sits; the 9th-12th men are. Every valuation
prints wins under all three, because the choice scales every dollar figure.

**Win model.** Win probability in each game is

    P(win) = Φ( (net_team − net_opp + 2h·v) · poss_game / 100 / σ_margin )

`σ_margin` is fitted each season as the root-mean-square gap between actual and
expected margins over games between two D1 teams: 10.9 points in 2026 and 11.2 in
2025. Non-D1 opponents are rated no worse than the worst D1 team. WAR is the sum
over the team's games of `P(win | net_team) − P(win | net_team − Δ)`, computed for
4,000 draws of the player's net rating from its posterior. The team's own rating is
held at its point estimate in those draws.

**Shared draws.** The same 4,000 rating draws feed WAR, program value and the
allocated price, so surplus is the difference of comonotone quantities. Until 0.3.1
the allocation drew ratings independently, which widened the surplus interval by
roughly an order of magnitude.

`war_linear` is the derivative of the win sum times Δ. It agrees with the
simulation for small impacts and understates it for a star on a dominant team.

## 4. Program value

**Revenue response.** On school-seasons 2012-2025, excluding 2020 (tournament
cancelled) and 2021 (attendance limits):

    log R_st = a_s + g_t + b0·W_st + b1·W_s,t−1 + d0·Bid_st + d1·Bid_s,t−1 + e_st

School effects absorb brand, market and conference; season effects absorb league
growth. Only rows where reported revenue differs from expense enter the fit: 61% of
D1 rows report the two as equal, which means the school allocates revenue to the
sport rather than measuring it. The fit uses 1,370 school-seasons from 167 schools
with at least four usable seasons (median revenue $6.4M, against $2.4M for all
D1 school-seasons). A bootstrap over schools gives 400 replicates.

| Effect (log revenue per unit) | Median | 80% interval |
| --- | --- | --- |
| One win, this season (`b0`) | 0.0041 | see bootstrap |
| One win, this and next season (`b0 + b1`) | 0.0063 | 0.0037 to 0.0089 |
| NCAA bid, this and next season (`d0 + d1`) | 0.053 | 0.015 to 0.091 |

**This is an association, not a causal estimate.** The identifying variation is a
school winning more or less than its own norm. Schools that raise basketball
spending win more and earn more in the same seasons: within the same sample, wins
regressed on log expense with the same fixed effects give about 6 wins per log
point, and log revenue and log expense move together with correlation 0.96. That
biases `b0` upward. Revenue attribution can also shift with winning. Against that,
EADA revenue omits conference media money, donations and brand effects, which biases
the figure downward. The net direction is unknown. "A win is worth about $130k to a
$21M program" is the size of the association, not of a marginal win's causal effect.

**Revenue base.** A win's revenue effect is the coefficient times the school's mean
reported revenue over its latest three seasons. Allocated rows are kept in that
base: they still give the level the program books against the sport. So for the
222 of 366 teams whose last three seasons are all allocated, the dollar figure is an
elasticity from revenue-tracking schools applied to an expense number.

**Annual and two-season.** The headline program value is this season's effect
(`b0`, `d0`) plus the school's share of tournament units. The two-season figure adds
next season's carry-over (`b1`, `d1`), which accrues whether or not the player is
still on the roster. Only the annual figure is set against a season of pay.

**Tournament bids.** A logistic model of bid on win percentage, power-conference
membership and their interaction, fitted on 2011-2026 except 2020 and 2021. It is
unregularized, has no strength-of-schedule term and includes the season being
valued; a fitted power team goes from a 47% to a 94% bid probability between .600
and .700, so the bid component is sensitive near the bubble.

**Tournament units.** A conference earns one unit per game a member plays, except
the championship game; the data give 1.94 units per bid, a field average that
overstates what a marginal bubble bid earns (about one game). A unit is worth about
$2M paid over six years, undiscounted. The school's share defaults to an equal
split among conference members, a user assumption. The bid-revenue term above is
estimated from EADA revenue, which for revenue-tracking schools may already include
distributed unit money, so the two components can double count.

## 5. Roster market value

No public dataset records what individual college basketball players are paid, so
this is an allocation of a published budget, not a prediction, and it is labelled a
scenario.

**Budget.** Opendorse's 2025-26 study, as reported by On3 in March 2026, gives average
men's basketball roster spending of $7-10M in the ACC, Big 12, Big East, Big Ten
and SEC; $2-5M for a "B tier" the report names as the American, Atlantic 10,
Mountain West, Pac-12 and Horizon League; and $1-2M for every other conference.
Budgets are lognormal with the cited range as the central 80%. That reads a range of conference averages as a distribution for one school, so
every school in a tier gets the same budget distribution and the allocation sums to
it in every draw. The House settlement caps direct revenue sharing at about $20.5M
across all sports in 2025-26; the budgets above also include collective money.

**Roles.** Players are ranked by possessions. The top five are starters, the next
three are rotation, the rest are bench, and anyone on the floor for under 5% of
possessions is unpaid. The same study reports rotation players earn 15-25% less than
starters and bench players 30-55% less; each draw samples a ratio from those ranges.

**Impact within a role.** Pay within a role is proportional to
`max(net − R, 0.5)^κ` with κ = 1, normalized to the role mean so the role ratios
still hold. The result is flat: a national player-of-the-year candidate gets about a
fifth of a power-tier budget and every paid player is within a factor of about three
of every other. Real rosters are more concentrated than that.

The allocation does not see recruiting rank, draft stock, transfer leverage or
social following, which drive real contracts.

## 5b. Fitted market model (dormant)

**Labels.** Revenue-share and collective deals for a men's basketball player-season,
annualized and summed, from a user CSV in the deal-registry schema. Deals are
matched to rated players by stats.ncaa.org id when given, otherwise by team and
normalized name. Unmatched deals are listed.

**Features.** Net, offensive and defensive rating; box-score prior; share of team
possessions; starter flag; linear WAR; team net rating, win percentage and NCAA bid;
power and B-tier flags; log program revenue with a missing flag.

**Model.** Ridge regression of log pay on standardized features. The penalty is the
grid value (0.1, 1, 10, 100) with the lowest leave-one-school-out mean absolute error
in log dollars. Schools are the folds because pay is set school by school.

**Intervals.** CV+ (Barber, Candès, Ramdas and Tibshirani, 2021) with schools as
folds and miscoverage `(1 − level)/2` in each tail, which the theorem guarantees
covers at least `level` under exchangeability. Point estimate, interval and the
draws surplus uses all come from the same CV+ set. Two limits: schools contribute
unequal numbers of labels, which the standard proof does not cover, and labels are
disclosed deals, so the guarantee applies to players like those in the registry,
not to every player priced.

**Use gate.** The fitted price replaces the allocation only with at least 40
labeled player-seasons from 10 schools and held-out log error below both a
leave-one-school-out tier median and the allocation's error on the labels it prices.
Disclosed pay for the player-season itself still takes precedence. The registry is
empty, so no valuation uses the model.

## 6. Surplus and position among teammates

Surplus is annual program value minus annual price, draw by draw on shared rating
draws. The price is disclosed pay when the registry has it, then the fitted model if
it passes its gate, then the allocation.

The team table places each paid player against the medians of program value and
price among his paid teammates: value above price, both above, value below price, or
both below. It is a within-team description, not a league ranking and not a verdict
on anyone, and unpaid players get no label.

## 7. Reproducing

```bash
uv sync --group dev
uv run pytest                  # offline suite, synthetic seasons
uv run pytest -m network       # full 2025 and 2026 fits, citation checks against live pages
athletevalue validate --season 2026
athletevalue fit --season 2026 --cv
uv run python scripts/returning_players.py 2025 2026
uv run python scripts/readme_example.py
```

Every download is recorded in `manifest.json` in the cache with its URL, SHA-256,
size, retrieval time and license tag. Derived files (baseline fits used to train the
prior, EADA extracts, season outcomes) are not yet recorded there and are keyed on
the model version only for the baseline fits; that is scheduled for 0.4.0, along with
pinned expected hashes for the upstream assets, which can be re-cut without notice.
