# Contributing

## Setup

```bash
git clone https://github.com/jman4162/athletevalue
cd athletevalue
uv sync --group dev
```

Checks CI runs on every push:

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src/
uv run lint-imports
uv run pytest --cov
```

`uv run pytest -m network` fits full seasons from live data. It takes a few minutes
after the first download.

macOS note: if `import athletevalue` fails inside the virtual environment, check
whether the `.pth` files in `.venv/lib/python3.*/site-packages` carry the `hidden`
flag (`ls -lO`). Some sync tools set it, and Python 3.12.13 and later skip hidden
`.pth` files. `chflags nohidden .venv/lib/python3.*/site-packages/*.pth` fixes it.
The test configuration adds `src` to the path, so pytest is unaffected.

## Rules the build enforces

1. **No bare constants in modelling code.** A number in `impact`, `wins`,
   `economics`, `market` or `valuation` must come from the assumption registry
   (`src/athletevalue/assumptions/data/*.toml`) with a citation, a rationale or a
   `user_input` basis. `tests/governance` fails otherwise.
2. **Every registry entry is used.** An entry no code reads is removed.
3. **Layers.** Modelling packages do no I/O, and no fitting code imports the
   reader for user-supplied restricted data. `lint-imports` checks both.
4. **Evidence status propagates.** A new estimate takes the weakest status of its
   inputs.

## Adding a deal to the registry

See `src/athletevalue/data/deal_registry/README.md`. In short: a public URL that
states the figure for that athlete, no rating-site valuations, no data from sources
that forbid redistribution, and a signed-off commit (`git commit -s`) licensing the
row under CC BY 4.0.

## Writing

Docs and docstrings state what a thing does and what was measured. Avoid adjectives
that are not measurements.
