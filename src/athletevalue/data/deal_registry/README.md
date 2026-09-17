# Deal registry

Publicly sourced compensation deals for college athletes, one row per deal. The
file ships empty and **is not accepting contributions yet**: see `PRIVACY.md` for
the correction and removal process that has to exist first, and for the rules that
will apply when it opens. The registry is licensed CC BY 4.0 (see LICENSE-DATA).

## Columns

| Column | Required | Meaning |
| --- | --- | --- |
| deal_id | yes | Stable identifier, e.g. `mbb-2026-0001` |
| athlete_name | yes | Name as it appears in stats.ncaa.org box scores |
| athlete_id | no | stats.ncaa.org player id, if known |
| season | yes | Ending year of the season the deal covers (2026 = 2025-26) |
| school | yes | School at the time of the deal |
| sport | yes | `mbb`, `wbb` or `cfb` |
| deal_date | no | ISO date the deal was signed or reported |
| cash_value | yes | USD over the whole duration |
| in_kind_value | no | USD value of goods and services |
| duration_months | yes | Length of the deal |
| counterparty | no | School, collective or company |
| deal_type | yes | `revenue_share`, `collective`, `endorsement`, `appearance`, `other` |
| deliverables | no | What the athlete provides |
| source_url | yes | Public page stating the figure |
| source_quality | yes | `contract_or_records_request` or `named_report` |
| status | yes | `active`, `disputed` or `withdrawn` |
| status_note | no | Why a row is disputed or withdrawn |
| notes | no | Anything a reader needs to interpret the row |

Rows with status `disputed` or `withdrawn` are kept as tombstones and never used.

## Rules

1. Every row needs a public `source_url` that states the dollar figure for this
   athlete, from a named source. Unnamed-source figures are not accepted.
2. Valuations are not deals. Do not add figures from On3 NIL Valuations, The NIL
   Standard, Opendorse Market Intel or similar rating products; they are model
   outputs.
3. Do not add data from sources whose terms forbid redistribution, including
   subscription databases, or records that appear to have been released in error.
4. The athlete must be 18 or older at the deal date.
5. Sign off your commit (`git commit -s`) and include the certifications in
   `PRIVACY.md`.
