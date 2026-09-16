# Deal registry

Publicly sourced compensation deals for college athletes, one row per deal. The
file ships empty. Contributions are welcome under the rules below; the registry
is licensed CC BY 4.0 (see LICENSE-DATA).

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
| source_quality | yes | `contract_or_records_request`, `named_report`, `anonymous_report` |
| notes | no | Anything a reader needs to interpret the row |

## Rules

1. Every row needs a public `source_url` that states the dollar figure for this athlete.
2. Valuations are not deals. Do not add figures from On3 NIL Valuations, The NIL
   Standard, Opendorse Market Intel or similar rating products; they are model
   outputs.
3. Do not add data from sources whose terms forbid redistribution, including
   subscription databases.
4. Sign off your commit (`git commit -s`) to certify you have the right to submit
   the row under CC BY 4.0.
