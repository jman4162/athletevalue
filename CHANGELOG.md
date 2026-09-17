# Changelog

## 0.4.0 (2026-09-17)

Reproducibility
- Upstream files behind the documented numbers are pinned by SHA-256 in
  `athletevalue/data/pinned_artifacts.json`. A download or cached copy that differs
  from its pin raises `ArtifactMismatchError`; `allow_unpinned=True`
  (`--allow-unpinned`) accepts it. `scripts/pin_artifacts.py` regenerates the file.
- Derived files (EADA extracts, no-prior baselines, economics outcomes) are keyed on
  the model version and settings, and record the digest of every input. A changed
  input, release or setting makes them stale instead of silently reused.
- Downloads retry HTTP 429, 5xx and connection failures with backoff, honouring
  `Retry-After`. The EADA file list is refreshed after 30 days; `fetch --refresh`
  downloads a season again.
- `ATHLETEVALUE_DETERMINISTIC=1` runs BLAS on one thread, so repeated runs agree bit
  for bit on one machine.
- Seasons skipped because a source has not published them are reported as notes on
  the economics model.

Figures and documentation
- `athletevalue[viz]` draws six figures from the snapshot: team ratings against Torvik,
  cross-validation curves, starters' WAR under each replacement definition, rating
  precision by possessions, revenue per win, and a synthetic roster's value against
  price. `scripts/generate_readme_figures.py` writes byte-stable SVGs to `docs/_static`,
  and CI fails if they differ from the committed files.
- A documentation site at https://jman4162.github.io/athletevalue/ built by GitHub
  Actions: the README and METHODOLOGY included rather than copied, equations in
  LaTeX, generated Validation and Assumptions pages, and an API reference.
- A test pins every validation figure quoted in the README and METHODOLOGY to the
  snapshot. Rebuilding it restated the revenue section: one win, this and next
  season, 0.0065 (80%: 0.0038 to 0.0090; was 0.0063); an NCAA bid, 0.049 (0.014 to
  0.083; was 0.053). The gates are unchanged.
- `.zenodo.json` describes the software for a Zenodo DOI on tagged releases.

API and CLI
- `api.Session` holds a cache and a registry and reuses fitted seasons, economics
  models and market fits, keyed on the season settings, the registry digest and the
  model version. Module-level functions share a default session. `value_player` and
  `value_team` accept `model=` and `economics_model=`.
- Global CLI options `--offline`, `--cache-dir` and `--allow-unpinned`; `cache info`,
  `cache verify` and `cache clear`; `--no-economics` for `value` and `team`. `fetch`
  also downloads the seasons the box-score prior trains on, so later commands can run
  offline. Missing files, pin mismatches and refused downloads print one line instead
  of a traceback.
- `api.load_economics_frames` is no longer exported; import it from
  `athletevalue.sports.mbb.economics_loaders`.

Validation and identity
- Rating tables and `PlayerRef` carry a `person_id` from the SportsDataverse reference
  RAPM release, which follows a player across seasons.
- `validate --extended` adds gates above the ratings: the fitted margin spread, the
  correlation of summed player WAR with team wins, the returning-player check (the
  prior must raise the correlation with next season's no-prior ratings), the share of
  revenue bootstrap draws with a non-positive win effect, and bid-model calibration by
  decile. Each band is a registry entry with its rationale.
- `scripts/build_snapshot.py` writes `docs/_static/data/snapshot.json`: gates, CV
  curves, team ratings, rating precision, WAR by replacement definition, revenue
  draws, bid calibration and synthetic interval coverage, with no player names or ids.
- Team tables gain `adj_net_no_prior`, the team rating from the ratings shrunk toward
  zero.

Fixes
- The fitted market model's use gate compared the tier median with a held-out error
  at the penalty chosen on that same error. With pure-noise labels that error beat the
  tier median (0.810 vs 0.812). The gate now uses nested cross-validation, choosing
  the penalty without the held-out school (0.832 on the same labels). No valuation
  used the model, since the registry is empty.
- Starters, allocation roles and the bench used for the replacement level broke ties
  in playing time by row order, and rows were sorted by rating, whose last bits vary
  with BLAS threading. Ties now break by athlete id. The synthetic README example
  moves by a few thousand dollars because players draw from the generator in a new
  order.
- Neutral-site and NCAA tournament games before 2023, which carry no ESPN ids, are
  matched by date and names. Abbreviated box-score names ("UNI", "SFA", "FDU") and
  ESPN's "USC" did not match, so 1 to 7 tournament games a season were missed and
  5% to 8% of neutral-site and postseason games kept a home-court flag. Names are now also
  compared with their crosswalk institution names and a short curated alias list.
  Every tournament game in 2012-2026 is matched, and at least 99% of neutral-site and
  postseason games in every season. This changes bids and tournament wins in the
  economics inputs, and venue flags in pre-2023 seasons.
- The revenue bootstrap and the market model numbered schools with polars categorical
  codes. In recent polars those codes are shared across the process, so a second fit
  in one session could see non-contiguous codes and resample empty schools. Codes are
  now computed locally in order of first appearance, which gives the same numbering
  the published fits used.
- Cross-validation subtracts the held-out Gram in place and reuses one factor buffer,
  cutting peak memory; results are bit-identical.

Wording
- The report footer, README and DISCLAIMER state plainly that no contract data is
  used, replacing "estimates, not reports" phrasing; docs were checked with
  slopscore-lint 0.14.1 (all pages score low).

Publishing
- Releases publish from GitHub Actions through PyPI Trusted Publishing. The package
  metadata no longer carries an email address.

## 0.3.1 (2026-09-17)

Fixes and corrections from an adversarial review. 0.2.0 and 0.3.0 are yanked on PyPI.

Model
- The box-score prior trains only on seasons before the target season. Every season
  but the latest previously shrank toward a prior that had seen later seasons.
- `fit --cv` no longer leaks held-out games into the prior. The penalty is chosen with
  the box totals and team adjustment rebuilt from each fold's training games; the
  validation gate uses the same procedure, which the earlier gate did only for the
  adjustment. The penalty stays at 3000.
- WAR, program value and the allocated price read one shared draw of each player's
  rating, so the surplus interval no longer samples the same rating twice.
- Program value is annual: this season's win and bid effects plus tournament units.
  The two-season figure with next season's carry-over is reported separately and is
  not set against one season of pay.
- The replacement level is a scenario input with three definitions (NBA convention,
  pooled coefficient, bench median); the default is the bench median (about −1.2) and
  every summary prints wins under all three.
- The game-margin spread is fitted on D1-vs-D1 games; non-D1 opponents are rated no
  worse than the worst D1 team.
- Fitted-model price point, interval and surplus draws come from one CV+ set, with
  ranks that guarantee the stated level.

Claims and citations
- The Pythagorean-exponent citation pointed at a page that did not contain the
  figures; it now quotes Pomeroy's ratings page (archived copy linked).
- Two paraphrased citations are verbatim quotes; market sources credit On3 and name
  the Opendorse study; the units source credits Deseret News; the House cap is a
  registry entry.
- The box-prior evidence is restated against a zero-box control: most of the gain is
  the team adjustment. The returning-player check ships as a script.
- "Lower bound" language is replaced with the bias discussion; "cites every constant"
  with "states the basis for every constant".

Public posture
- The README example is a synthetic player; real names appear only in tests.
- The team table's labels are neutral positions against team medians, with the
  medians printed, and unpaid players get none. Every summary ends with a disclaimer.
- The deal registry is closed until the correction and removal process in the new
  PRIVACY.md is staffed; the unnamed-source tier is gone; rows carry a status.
- LICENSE-DATA and the README describe the real chain of title for SportsDataverse
  data; ESPN-derived and NCAA-derived assets are tagged as such in the cache manifest.
- DCO text vendored; `--last-season` on `fit`, `validate` and `value`; the citation
  suite fetches every cited page in the network run.

## 0.3.0 (2026-09-17)

- Fitted roster-market model: ridge regression of log annual pay on 14 player and
  program features, penalty chosen by leave-one-school-out cross-validation, CV+
  prediction intervals with schools as folds.
- Use gate: at least 40 labeled player-seasons from 10 schools, and held-out error
  below both a tier median and the allocation. Disclosed pay still outranks the model.
- `athletevalue market-fit [--labels deals.csv]` and `value --labels`; valuations keep
  the allocation as `allocated_market_value` when the model sets the price and warn
  when the model is not used.
- No fitted model ships: the deal registry is empty.
- README badges, description and hashtags. Model version `mbb-v0.3.0`.

## 0.2.0 (2026-09-17)

- Box-score prior for RAPM: a ridge model of per-100 box rates fitted on the four
  nearest other seasons, with a Box Plus/Minus style team adjustment, used as the
  shrinkage target (λ = 3000). Held-out error falls 0.3% in 2025 and 2026, next-season
  correlation for returning players rises from 0.40 to 0.52, and team ratings match
  Torvik's scale (slope 0.96 vs 0.81).
- New validation gate `prior_cv_error_ratio`, computed with the team adjustment refit
  on training games only so held-out games cannot leak into the prior.
- `athletevalue fit --prior/--no-prior`; rating tables gain a `prior_net` column and
  valuations a driver showing how far possessions moved a player from his prior.
- The reference-RAPM gate compares the no-prior fit, which matches the reference's
  estimand.
- Model version `mbb-v0.2.0`.

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
