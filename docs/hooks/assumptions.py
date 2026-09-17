"""MkDocs hook: render every assumption-registry entry as the Assumptions page."""

from __future__ import annotations

from itertools import groupby

from mkdocs.structure.files import File, Files

from athletevalue.assumptions.registry import AssumptionRegistry

GROUP_TITLES = {
    "economics": "Program economics",
    "market": "Roster market",
    "mbb.impact": "Player impact",
    "mbb.prior": "Box-score prior",
    "mbb.wins": "Wins",
}


def _escape(text: str) -> str:
    """Keep dollar amounts from being read as math."""
    return text.replace("$", "\\$")


def _group(assumption_id: str) -> str:
    parts = assumption_id.split(".")
    return ".".join(parts[:2]) if parts[0] == "mbb" else parts[0]


def _value(value: object) -> str:
    if isinstance(value, tuple):
        return ", ".join(_value(v) for v in value)
    if isinstance(value, float) and value.is_integer():
        return f"{value:,.0f}"
    return str(value)


def render(registry: AssumptionRegistry) -> str:
    items = sorted(registry, key=lambda a: a.assumption_id)
    lines = [
        "# Assumptions",
        "",
        "Every constant the models use, generated from the registry TOML files in",
        "`src/athletevalue/assumptions/data` when the site is built. `athletevalue",
        "assumptions` prints the same list, and a governance test fails on a bare number",
        "in modelling code. Status: ● reported, ◆ derived, ▲ estimated, ○ scenario.",
        "Override any entry with `--assumptions my.toml`.",
        "",
    ]
    for group, members in groupby(items, key=lambda a: _group(a.assumption_id)):
        lines += [f"## {GROUP_TITLES.get(group, group)}", ""]
        for item in members:
            lines += [f"### `{item.assumption_id}`", "", _escape(item.description.strip()), ""]
            value = "derived from data" if item.value is None else _value(item.value)
            lines += [
                f"**Value** {_escape(value)} ({_escape(item.unit)}) · **basis** "
                f"`{item.basis.value}` · **status** {item.status.glyph} {item.status.value}",
                "",
            ]
            citation = item.citation
            if citation is not None:
                title = _escape(citation.title)
                link = f"[{title}]({citation.url})" if citation.url else title
                archived = (
                    f" ([archived]({citation.archived_url}))" if citation.archived_url else ""
                )
                lines += [f"Source: {link}{archived}", ""]
                if citation.quote:
                    lines += [f"> {_escape(citation.quote)}", ""]
            if item.rationale:
                lines += [f"*Rationale.* {_escape(' '.join(item.rationale.split()))}", ""]
    return "\n".join(lines)


def on_files(files: Files, config: dict) -> Files:
    files.append(
        File.generated(config, "assumptions.md", content=render(AssumptionRegistry.load()))
    )
    return files
