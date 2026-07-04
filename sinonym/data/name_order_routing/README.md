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
  from evidence. Labels are `pp`, `vys`, `either`, or `uncertain`.
- `pp_abstain_labels.jsonl`: 750 PP/abstain routing examples from the
  corrected train, corrected holdout, and fresh holdout feature sets of the
  missing-venue experiment. Rows store raw `name` so the fixture is
  self-contained for future evidence refreshes (paper IDs stay out by
  design). Labels are `pp`, `abstain`, or `uncertain`.

## Evidence provenance (re-refreshed after the Korean-dominant full-share trim, July 2026)

The evidence columns were last regenerated after five Korean-dominant
`CANTONESE_SURNAMES` keys (`jung`, `moon`, `im`, `kyeong`, `pak`) were demoted
from full share to the penalty share in `surname_romanizations.csv`: they now
score as-written like rare Chinese surnames (their discounted mass) instead of
inheriting their Mandarin target's full mass (zheng/wen/lin/jing/bai). Heavily
Chinese-used overlap spellings stay full share (`lee`, `lim`, `chang`/`jang`,
`choi`, `ho`, `han`, `yu`, `koo`, and the bicultural `shin`/`son`/`soo`/`suh`);
see the generator's `KOREAN_DOMINANT_FULL_SHARE_EXCLUSIONS` for the
per-spelling measured rationale. This refresh changed `selected_surname_frequency`
on 1/750 pp-abstain rows (plus `batch_total_count` on 2 more from batch
recomposition) and at least one evidence field on 331/1000 pp-vys rows; no
pp-abstain routing outcome moved.

The prior refresh switched surname-frequency evidence to the canonical
as-written romanization resolution (`surname_romanizations.csv` rows resolve to
as-written mass: full-share rows carry the mandarin target's mass, penalty rows
the discounted as-written mass, so e.g. `Fai` no longer inherits xu's full mass
through the fai->hui remap). Before that they were
regenerated under the PR #18 parser after the
ordered-pair bigram feature was removed (one-sided order suppression,
romanization-conditional surname discount, and the corpus-regenerated
positional table remain) by re-running the original batches and building rows
with the canonical production builders, so the fixtures match exactly what
production evidence extraction emits (the pp-abstain columns dropped from the
builder in PR review — `selected_over_alternate_ratio`, `compact_cjk`,
`jp_probability` — remain dropped from the fixture too):

- PP evidence: `analyze_name_batch()` over the paper's own author list.
- VYS evidence (pp-vys fixture only): `analyze_name_batch()` over names pooled
  by venue/source/exact-year (greedy-packed, cap 1000).
- Rows built with `build_pp_vys_abstain_rows` / `build_pp_abstain_rows`. In
  particular, `pp_batch_confidence` / `vys_batch_confidence` store
  `format_pattern.decision_confidence` (the score the batch gate actually
  thresholds), not the raw vote share `format_pattern.confidence` that the
  original June extraction stored.
- `pp_result_token_count` counts text tokens of the rendered result string
  (a hyphenated given name is one token no matter how many syllables it joins,
  so "Chang-Qing Zhang" = 2 and "Ka-Wai-Man Wong" = 2; failed parse = 1). This
  is the convention the pp-abstain rules were tuned on; the builder's
  `_parse_result_token_count` was fixed to match it (it previously counted
  parsed surname+given+middle tokens, splitting hyphenated given names and
  defeating the `pp_result_token_count == 2` accept rules).
- `pp_success` is a required pp-abstain column (both builders emit it). A failed
  PP parse (`pp_success == false`, encoded in this fixture as
  `pp_result_token_count == 1`, always with `selected_format == "mixed"`) routes
  to the terminal `not_person` decision, so the fixture exercises the same
  failed-parse path production takes rather than an abstain path it never reaches.

## Label provenance (relabeling round, July 2026)

Identity columns are unchanged from the original manual labeling rounds
(June 2026). The `decision`/`confidence` labels went through one reviewed
relabeling round (packets in `scratch/relabeling/`, all machine drafts
accepted by the repo owner) applied on top of the re-refreshed evidence:

- pp-vys, `either` labels (163 rows): every row whose current label was
  `pp`/`vys` but whose refreshed PP and VYS parses emit the identical string
  was relabeled `either` — the June route label pretends to test routing
  behavior it cannot observe, and `either` is future-proof (see Semantics).
  The set was derived from the refreshed evidence itself, not the June-era
  draft list: 127 of the 136 drafted rows still had identical strings, 9
  diverged under the current parser and kept their original label, and 36
  newly-identical rows joined. After the bigram-feature revert, 7 of those
  rows' PP/VYS strings diverged again (all 3-token names, the removed
  feature's footprint); having no pre-`either` route label, they were demoted
  to `uncertain` pending relabeling, leaving 156 `either` rows. The July 2026
  as-written romanization-resolver refresh (below) re-derived the set once
  more: 12 `vys` rows whose strings converged joined and 4 `either` rows whose
  strings diverged were demoted to `uncertain`, leaving 164 `either` rows. The
  Korean-dominant full-share trim refresh re-derived the set again: 2 `pp` and
  1 `vys` row whose strings converged joined and 3 `either` rows whose strings
  diverged were demoted to `uncertain`, still 164 `either` rows (net zero).
- pp-vys, suspicious labels (20 rows adjudicated): June label disagreed with
  both the current router and at least one strong-evidence signal; 10 were
  overturned, 10 kept as labeled.
- pp-abstain, `clean_bilingual_given_first` support (36 rows): bilingual
  aligned Latin+Han rows where the Han gloss verifies the PP parse; 34
  relabeled `uncertain` -> `pp`, 2 kept `uncertain` (PP mis-segments and both
  routes emit the same wrong-order string).
- pp-abstain, curated uncertain triage (24 rows): 18 -> `pp` (material
  surname-first reorders, verified Han/bilingual glosses, Cantonese embedded
  Han), 6 -> `abstain` (clearly Japanese names where a pinyin parse is
  inappropriate).

### Why the asserted metrics differ from the June numbers

The June fixture froze evidence that no longer matches what the pipeline
produces, for three stacked reasons:

1. Batch-evidence fixes landed on main after the June extraction; 980/1000
   pp-vys rows already had at least one changed evidence field when re-run
   under the pre-PR #18 parser.
2. PR #18's parser changes the PP/VYS parses and batch statistics further
   (three more parser changes landed between the `bffbad1` refresh and the
   PR #18 re-refresh, which was re-run once more after the bigram-feature
   revert).
3. The confidence convention changed from `format_pattern.confidence` to
   `decision_confidence` (production convention).

The regression tests assert the current numbers for the current router on
the re-refreshed, relabeled fixtures: pp-vys 719/791 decisive rows correct at
the emitted-string level plus 164/164 `either` rows (see Semantics), and
pp-abstain at the route level as follows. Of the 608 decisive (`pp`/`abstain`)
rows, 312 are failed PP parses (encoded `pp_result_token_count == 1`, always
`selected_format == "mixed"` with zero selected-surname frequency) that route to
the terminal `not_person` decision, matching production — 311 are
`abstain`-labeled, 1 is `pp`-labeled. Of the 296 rows that receive a person
route, 276 match their label (`abstain`/`abstain` 127, `pp`/`pp` 149); the 20
mismatches (6 `abstain`→`pp`, 14 `pp`→`abstain`) are mostly `pp` labels the tuned
rules deliberately leave on `abstain` (e.g. below-threshold bilingual glosses),
documenting router conservatism, not label errors.

## Semantics

For PP/VYS/abstain scoring, router output `abstain` means "emit the
preprocessed input-order parse." The regression test scores decisive rows at
the emitted-string level: a row is correct iff the string the chosen route
emits (`pp_result` for `pp`, `vys_result` for `vys`, and for `abstain` the
result on whichever side `input_order_candidate` says preserves input order)
equals the labeled route's string. Router output `not_person` is a terminal
non-person decision and means no person parse should be emitted.

Label `either` marks rows whose PP and VYS parses emit the identical string
under the current parser: the route is cosmetic, so any person route scores
as correct. The regression test asserts `pp_result == vys_result` still holds
for every `either` row — if a later parser change makes the strings diverge
again, the test fails loudly and the row goes back for relabeling instead of
silently trusting a label made under different parses.

Abstain rows always have `input_order_candidate` in {`pp`, `vys`} — the only
abstain-emitting rule requires a defined input-order side — so consumers may
treat abstain + `unknown` as a contract violation.

In the PP-only regime there is no second run to pick from: most `abstain`
rows are materialized with `input_order_parsed(result)` (the as-typed reading
where the trailing token is the surname and everything else keeps its
position). Spaced Han surname-first rows are the exception: the source space
already marks the surname boundary, so `spaced_cjk_zero_batch_surname_first`
abstains emit the PP parse instead of flipping to trailing-token surname. It
is an emitted person parse, not a defer signal, and it deliberately does NOT
re-parse the name standalone, since the single-name detector re-decides order.

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
  `scratch/routing_overhaul/fixtures/refresh_pp_abstain_textcount.py` (~1 min;
  its token-count override is now a no-op because the canonical builder counts
  text tokens itself, so plain `build_pp_abstain_rows` output is equivalent).

A refresh regenerates evidence columns only: identity and label columns must
be preserved byte-for-byte. After a refresh, re-derive the `either` set —
`pp`/`vys`-labeled rows whose refreshed strings became identical get `either`;
`either` rows whose strings diverged must go back for manual relabeling.

Both fixtures still depend on untracked `scratch/` artifacts for the batch
name lists. If those are ever at risk, promote the minimal inputs (batch name
list or batch id + index) into the fixture rows themselves so a refresh needs
nothing outside the repo.
