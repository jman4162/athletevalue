# athletevalue: notes for coding agents

- Layout: `src/athletevalue/{schemas,assumptions,sources,uncertainty,identity,frames,impact,wins,economics,market,valuation,sports,api.py,cli}`.
  Layer order and forbidden imports are in `pyproject.toml` under `[tool.importlinter]`.
- Checks: `uv run ruff check . && uv run ruff format --check . && uv run mypy src/ && uv run lint-imports && uv run pytest`.
  Live-data tests: `uv run pytest -m network`.
- If `import athletevalue` fails under `uv run`, the editable `.pth` file has the macOS
  `hidden` flag: `chflags nohidden .venv/lib/python3.12/site-packages/*.pth`, or prefix
  commands with `PYTHONPATH=src` and use `uv run --no-sync`.
- Modelling constants go in `assumptions/data/*.toml`, never inline; governance tests fail on
  bare numbers in `impact`, `wins`, `economics`, `market`, `valuation` (except `valuation/report.py`).
- Season key is the ending year. Market budgets exist only for 2026.
- Do not fit shipped values on KenPom, Knight-Newhouse or raw CollegeBasketballData; Torvik is validation only.
- Private research notes live in `*.local.md` files, which git ignores.
