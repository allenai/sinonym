# Canonical name normalization evaluation

## Result

The final rules were evaluated on 1,000 manually reviewed, real non-Chinese
author names:

| Measure | Exact | Accuracy |
|---|---:|---:|
| Person outcome | 1,000 / 1,000 | 100% |
| Canonical display text | 1,000 / 1,000 | 100% |
| First/given component | 1,000 / 1,000 | 100% |
| Suffix component | 1,000 / 1,000 | 100% |
| Middle component | 982 / 1,000 | 98.2% |
| Last/surname component | 982 / 1,000 | 98.2% |
| All five fields together | 982 / 1,000 | 98.2% |

The 18 component-only differences are reported as semantic-boundary
differences, separately from display accuracy. Raw undelimited strings cannot
identify those middle/surname boundaries reliably. When callers provide
structured first/middle/last fields, those source roles are authoritative by
default and are repaired only after mechanical cleanup empties a boundary.

The originally frozen 200-name holdout scores 196/200 (98.0%) on semantic
components and 200/200 on canonical display. The four component differences
remain visible under the source-preserving policy rather than being closed by
authority-specific family-span rules.

## Data and manual review

The sampling pool combined:

- 978 selected names from the internal S2 scholarly-author Parquet. The pinned
  source file contains 41,551,414 rows and has SHA-256
  `E4050B4832DAAF2F0B8464F0B78E6CA0D35412897BDDA5AF148F7BACBEE9D363`.
- 22 selected names from the repository's ACL 2025 author list.

Candidate sampling was deterministic, source-balanced, deduplicated, and
stratified by visible name shape. The final 1,000 contain:

| Shape | Count |
|---|---:|
| Ordinary | 498 |
| Initials | 130 |
| Non-ASCII Latin | 94 |
| Hyphen | 69 |
| Apostrophe | 57 |
| Family particle | 42 |
| Comma form | 40 |
| Suffix | 39 |
| Four-plus tokens | 31 |

Every initial output was assigned manually from the raw string under the
written rubric. Parser, model, and normalizer outputs were not shown to or used
by that initial labeling step. Ambiguous, corrupt, Chinese, and non-person rows
were excluded from gold rather than guessed. Review produced 1,053 usable gold
rows; a salted hash selected 1,000, then a separate salted hash fixed an 800/200
development/holdout split. The remaining 53 rows were a development reserve.

After the final evaluation exposed 23 ambiguous middle/surname boundaries, the
user explicitly requested a web-source adjudication of every mismatch. Six
manual labels were corrected from structured institutional or bibliographic
authority records. The remaining 18 boundaries are retained as semantic
benchmark differences; they are not treated as sufficient evidence to override
structured source roles. Each changed review row records its external source
URLs, and the full adjudication ledger is retained in ignored scratch artifacts.

The selected JSONL has SHA-256
`B367254A7282DA3B30A37B478A679202E5C65D465541A3A0E772AA649D30B8DE`.
It contains 977 high-confidence and 23 medium-confidence reviews.

Raw S2-derived names and manual review files remain under ignored
`scratch/canonical_names/`; they are not committed because they contain
internal personal data with upstream redistribution constraints. The ACL
source's redistribution terms are also not documented locally.

## Evaluation protocol

Rules were iterated only against the 800-name development split and the
53-name reserve. The holdout was evaluated once after the tuning rules were
frozen. A later independent code review found specification-level edge cases
not represented in the gold packet; those fixes were developed only against
synthetic regression strings, without consulting holdout labels. The full
packet was then rerun as a regression check and the reported holdout score was
unchanged. Exact match requires the person outcome, canonical text, first,
middle, last, and suffix to all agree with the manual record.

Reproduction commands, when the ignored source/review artifacts are present:

```powershell
uv run python scratch\canonical_names\assemble_gold.py `
  --acl-packet scratch\canonical_names\packets\acl_pilot_60.jsonl `
  --s2-packet scratch\canonical_names\packets\s2_candidates_1300.jsonl `
  --reviews-dir scratch\canonical_names\reviews `
  --output scratch\canonical_names\gold\canonical_non_chinese_gold_1000.jsonl `
  --reserve-output scratch\canonical_names\gold\canonical_non_chinese_reserve.jsonl `
  --manifest scratch\canonical_names\gold\manifest.json `
  --size 1000 --holdout-size 200

uv run python scratch\canonical_names\evaluate_gold.py `
  --input scratch\canonical_names\gold\canonical_non_chinese_gold_1000.jsonl `
  --split all --limit 1000 `
  --report scratch\canonical_names\metrics\all_final.json `
  --mismatches scratch\canonical_names\metrics\all_final_mismatches.jsonl
```

## Boundary policy

An undelimited raw string still cannot identify every middle/surname boundary
from shape alone. The normalizer therefore keeps the conservative final-token
surname rule, family-particle rules, and initial-shape rules as its default.
It does not use person-specific or authority-specific family-span exceptions.
Structured input preserves the supplied first/middle/last roles unless
mechanical cleanup empties a required boundary. Semantic boundary evaluation is
reported separately rather than changing that storage contract.
