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

`uv run pytest -m network` fits full seasons from live data and checks every cited
quote against its page. It takes a few minutes after the first download. Bart
Torvik's site refuses some cloud address ranges; the Torvik comparison is skipped
and reported when that happens.

macOS note: if `import athletevalue` fails inside the virtual environment, check
whether the `.pth` files in `.venv/lib/python3.*/site-packages` carry the `hidden`
flag (`ls -lO`). Some sync tools set it, and Python 3.12.13 and later skip hidden
`.pth` files. `chflags nohidden .venv/lib/python3.*/site-packages/*.pth` fixes it.
The test configuration adds `src` to the path, so pytest is unaffected.

## Rules the build enforces

1. **No bare constants in modelling code.** A number in `impact`, `wins`,
   `economics`, `market` or `valuation` must come from the assumption registry
   (`src/athletevalue/assumptions/data/*.toml`) with a citation, a rationale, or a
   `user_input` basis. `tests/governance` fails otherwise.
2. **Every cited constant quotes its source.** A `published_*` entry needs a verbatim
   `quote`; the network suite fetches the page and looks for it.
3. **Every registry entry is used.** An entry no code reads is removed.
4. **Layers.** Modelling packages do no I/O, and no fitting code imports the
   reader for user-supplied restricted data. `lint-imports` checks both.
5. **Evidence status propagates.** A new estimate takes the weakest status of its
   inputs.

## Sign-off

Every commit is signed off (`git commit -s`), which certifies the Developer
Certificate of Origin in `.github/DCO`: that you wrote the change or have the right
to submit it under this project's licenses (MIT for code, CC BY 4.0 for files under
`src/athletevalue/data`).

## Deal registry

The registry is closed to contributions until the correction and removal process in
`PRIVACY.md` is staffed. When it opens, a row also needs the certifications listed
there (accurate transcription from a public URL, a named source, athlete 18 or
older, no confidential terms). Those are about the person in the row; the DCO is
about your right to license the text.

## Writing

Docs and docstrings state what a thing does and what was measured. Avoid adjectives
that are not measurements. Do not name real athletes in documentation; the README
example is synthetic.
