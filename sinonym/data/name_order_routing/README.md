# Name-order routing fixtures

These fixtures are the compact, reviewed evaluation sets for
`sinonym.pipeline.name_order_routing`. This maintainer document keeps the
evaluation and regeneration detail out of the user-facing README.

## Files

- `pp_vys_abstain_labels.jsonl` contains 1,000 examples from four 250-row
  review rounds. Labels are `pp`, `vys`, `either`, or `uncertain`.
- `pp_abstain_labels.jsonl` contains 750 examples from three 250-row review
  splits. Labels are `pp`, `abstain`, or `uncertain`.

Each row keeps its review identity and label together with the evidence needed
by the corresponding router. Paper IDs, source URLs, reviewer notes, and full
batch name lists remain in the scratch provenance artifacts rather than the
packaged fixtures.

## Current snapshot

Evidence was replayed from the production builders on 2026-08-09 after the
parser, normalization, and batch-policy changes in the current branch:

- PP/VYS: 650 of 1,000 rows changed in at least one evidence field. Nineteen
  labels whose former `either` outputs now diverged were restored from the
  original human gold (18 `vys`, one `pp`), and five newly identical outputs
  were changed to `either`. Final labels: 443 `pp`, 362 `vys`, 150 `either`,
  and 45 `uncertain`.
- PP-only: 155 of 750 rows changed in at least one evidence field. Review
  labels were unchanged: 164 `pp`, 444 `abstain`, and 142 `uncertain`.

The regression test records these current results:

- PP/VYS: 725 of 805 decisive rows emit the reviewed string. The breakdown is
  379/443 for `pp` labels and 346/362 for `vys` labels. All 150 `either` rows
  still have identical successful PP and VYS output.
- PP-only: the raw router returns a person route for 198 of 608 decisive rows;
  174 match the reviewed route (147 `pp`, 27 `abstain`) and 24 disagree. The
  other 410 rows return raw-router `not_person` because the current PP parse
  failed.

These are evaluation results, not accuracy claims for arbitrary author data.
The fixtures were sampled from specific PP/VYS disagreements and missing-venue
experiments.

## What the evidence means

- PP evidence comes from `analyze_name_batch()` over a paper's author list.
- VYS evidence comes from `analyze_name_batch()` over the corresponding
  venue/source/year pool.
- Rows are built by `build_pp_vys_abstain_rows()` or
  `build_pp_abstain_rows()`. Batch confidence is
  `format_pattern.decision_confidence`, the value used by the batch gate.
- PP-only `pp_result_token_count` counts tokens in the rendered result; a
  hyphenated name component is one token and a failed parse counts as one.
- `selected_surname_position` and `selected_surname_token_count` describe the
  selected surname span in the raw input.

For PP/VYS routing, `abstain` means to emit the preprocessed input-order parse,
using `input_order_candidate` to identify which parse preserves that order.
Label `either` is reserved for rows where successful PP and VYS parses emit the
same string. The test asserts both directions of this invariant: every `either`
row is equivalent, and no decisive row is equivalent.

For PP-only routing, `abstain` also emits a person parse in input order. Spaced
Han surname-first rows use the PP parse because the source spacing already
marks the surname boundary. If an input-order parse cannot be materialized,
the result is an explicit failure rather than a fallback to the rejected PP
reorder.

### `not_person` and terminal resolution

At the raw router layer, `not_person` means a required batch candidate was
unusable. In PP-only routing that is a failed PP parse; in PP/VYS routing it can
also mean that either candidate failed or produced an overlong garbage result.
Direct raw-router consumers treat it as terminal.

The TIMO `Predictor` deliberately has a broader contract: it treats raw-router
`not_person` as negative batch evidence while resolving the full record. A real
person can therefore still receive a reviewed or scalar assignment. Only a
reviewed non-person source pattern produces `suppress`; if no later person
result is usable, other cases preserve the source fields.

## Refreshing after parser changes

The committed unit test scores the router against committed evidence. It cannot
detect extraction drift by itself because the full source batches are not
packaged. After parser, normalization, lexicon, or batch-evidence changes, replay
the source batches before updating expected metrics.

Run a small preflight first:

```powershell
uv run --with pyarrow python scratch/routing_overhaul/fixtures/refresh_pp_abstain_textcount.py --limit 2 --output scratch/routing_overhaul/fixtures/preflight_pp_abstain.jsonl --metrics-output scratch/routing_overhaul/fixtures/preflight_pp_abstain_metrics.json
uv run python scratch/routing_overhaul/fixtures/refresh_pp_vys.py --limit 2 --output scratch/routing_overhaul/fixtures/preflight_pp_vys.jsonl --metrics-output scratch/routing_overhaul/fixtures/preflight_pp_vys_metrics.json
```

Then run the complete local replay:

```powershell
uv run --with pyarrow python scratch/routing_overhaul/fixtures/refresh_pp_abstain_textcount.py --limit 750 --output scratch/routing_overhaul/fixtures/pp_abstain_labels.jsonl --metrics-output scratch/routing_overhaul/fixtures/pp_abstain_metrics.json
uv run python scratch/routing_overhaul/fixtures/refresh_pp_vys.py --limit 1000 --output scratch/routing_overhaul/fixtures/pp_vys_abstain_labels.jsonl --metrics-output scratch/routing_overhaul/fixtures/pp_vys_metrics.json
```

On the 2026-08-09 refresh, PP-only took about three seconds and PP/VYS took
about five minutes. The source inputs are:

- PP-only: `scratch/missing_venue_pp_abstain/batches_pp.parquet` plus the three
  `labeling*/review_items_250.csv` files.
- PP/VYS: `scratch/routing_impact_check/batches_needed.json` and
  `targets.json`.

Before replacing a tracked fixture, verify row count and preserve `item_id`,
split/round, script, and raw name. Evidence regeneration must not invent new
human decisions. If an `either` row diverges, restore its reviewed route from
the original gold or send it for manual adjudication; if decisive outputs
become identical, change the label to `either` only after verifying both parses
succeed and emit exactly the same string.

Finally run:

```powershell
uv run pytest -q tests/test_name_order_routing_label_fixtures.py
```

The source batch artifacts are currently ignored under `scratch/`. If this
workflow must run in a clean checkout or CI, promote a minimized, compressed
source-batch artifact rather than weakening the extraction check.
