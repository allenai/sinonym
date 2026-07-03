# Name Order Routing Label Fixtures

Compact labeled fixtures for the external routing rules in
`sinonym.pipeline.name_order_routing`.

These are deliberately minimized from scratch experiment artifacts. They keep
only the manual label, split metadata, raw name, and columns required by the
tracked routers. They omit paper IDs, source URLs, notes, packet text, and
debug-only columns.

## Files

- `pp_vys_abstain_labels.jsonl`: 1,000 PP/VYS/abstain routing examples from
  four 250-row labeling rounds (sampled from PP-vs-VYS disagreements in the
  June `fixed_full` run). This fixture includes raw `name` because the
  promoted narrow name-prior guards require it, plus parser evidence from the
  PP and VYS runs. It does not store `old_prediction`, `old_reason`,
  `new_prediction`, or `new_reason`; the router derives those audit fields
  from evidence. Labels are `pp`, `vys`, or `uncertain`.
- `pp_abstain_labels.jsonl`: 750 PP/abstain routing examples from the
  corrected train, corrected holdout, and fresh holdout feature sets of the
  missing-venue experiment. Rows store raw `name` so the fixture is
  self-contained for future evidence refreshes (paper IDs stay out by
  design). Labels are `pp`, `abstain`, or `uncertain`.

## Evidence provenance (refreshed at v0.3.0 / PR #18)

Labels and identity columns are unchanged from the original manual labeling
rounds (June 2026). The evidence columns were regenerated at v0.3.0 (new
parser, PR #18, commit `bffbad1`) by re-running the original batches and
building rows with the canonical production builders, so the fixtures match
exactly what production evidence extraction emits:

- PP evidence: `analyze_name_batch()` over the paper's own author list.
- VYS evidence (pp-vys fixture only): `analyze_name_batch()` over names pooled
  by venue/source/exact-year (greedy-packed, cap 1000).
- Rows built with `build_pp_vys_abstain_rows` / `build_pp_abstain_rows`. In
  particular, `pp_batch_confidence` / `vys_batch_confidence` store
  `format_pattern.decision_confidence` (the score the batch gate actually
  thresholds), not the raw vote share `format_pattern.confidence` that the
  original June extraction stored.
- `pp_result_token_count` counts text tokens of the rendered result string
  (hyphenated given name = one token, so "Chang-Qing Zhang" = 2; failed parse
  = 1). This is the convention the pp-abstain rules were tuned on; the
  builder's `_parse_result_token_count` was fixed to match it (it previously
  counted parsed surname+given+middle tokens, splitting hyphenated given
  names and defeating the `pp_result_token_count == 2` accept rules).

### Why the asserted metrics differ from the June numbers

The June fixture froze evidence that no longer matches what the pipeline
produces, for three stacked reasons:

1. Batch-evidence fixes landed on main after the June extraction; 980/1000
   pp-vys rows already had at least one changed evidence field when re-run
   under the pre-PR #18 parser.
2. PR #18's parser changes the PP/VYS parses and batch statistics further.
3. The confidence convention changed from `format_pattern.confidence` to
   `decision_confidence` (production convention).

Under fresh v0.3.0 evidence the pre-pruning router scored 830/969 (85.7%)
label-level on the pp-vys fixture, versus 876/969 (90.4%) asserted against
the stale June evidence — the old number measured the router against inputs
production could no longer generate. The regression tests assert the fresh
numbers for the current (pruned) router: 877/969 at the emitted-string level
(see Semantics; 825/969 label-level) on pp-vys, and 542/550 on pp-abstain
(the frozen June fixture scored 540/550 under the pre-pruning rules; -2 of
the drift is real parser drift — `batch_total_count` changed on 55 rows;
`has_latin`, `selected_format`, and frequency fields on 7 han rows — and
rule pruning recovered +4).

## Semantics

For PP/VYS/abstain scoring, router output `abstain` means "emit the
preprocessed input-order parse." The regression test scores decisive rows at
the emitted-string level: a row is correct iff the string the chosen route
emits (`pp_result` for `pp`, `vys_result` for `vys`, and for `abstain` the
result on whichever side `input_order_candidate` says preserves input order)
equals the labeled route's string. When PP and VYS emit the identical string
the route is cosmetic and either route scores as correct. Router output
`not_person` is a terminal non-person decision and means no person parse
should be emitted.

## Refreshing these fixtures after parser changes

Evidence (not labels) should be regenerated whenever parser or batch-evidence
behavior changes:

- `pp_vys_abstain_labels.jsonl`: the original PP/VYS batch name lists and the
  per-row (batch, index) mapping live in
  `scratch/routing_impact_check/batches_needed.json` / `targets.json`;
  regeneration script: `scratch/routing_overhaul/fixtures/refresh_pp_vys.py`
  (~4 min, 992 PP batches + 922 VYS batches / ~691k names).
- `pp_abstain_labels.jsonl`: raw names live in the fixture itself; the paper
  ids, focal positions, and PP batches live in
  `scratch/missing_venue_pp_abstain/labeling*/review_items_250.csv` and
  `scratch/missing_venue_pp_abstain/batches_pp.parquet`; regeneration script:
  `scratch/routing_overhaul/fixtures/refresh_pp_abstain_textcount.py` (~1 min).

Both fixtures still depend on untracked `scratch/` artifacts for the batch
name lists. If those are ever at risk, promote the minimal inputs (batch name
list or batch id + index) into the fixture rows themselves so a refresh needs
nothing outside the repo.
