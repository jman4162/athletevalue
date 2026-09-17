# Changelog

## 0.1.0 (2026-09-16)

First release, men's basketball.

- RAPM from SportsDataverse possession data with exact posterior standard errors,
  pooled low-minute players, non-D1 opponents, neutral sites and grouped
  cross-validation.
- Validation gates against the SportsDataverse league-wide RAPM and Bart Torvik
  team ratings.
- Wins above replacement from a per-game win model over each team's schedule.
- Program value from a school fixed-effects model of EADA revenue, a tournament-bid
  model and conference tournament units.
- Roster market value for 2025-26 from reported roster budgets allocated by role
  and impact.
- Assumption registry with citations and evidence status on every estimate.
- CLI: `fetch`, `fit`, `validate`, `value`, `team`, `assumptions`, `cache-path`.
