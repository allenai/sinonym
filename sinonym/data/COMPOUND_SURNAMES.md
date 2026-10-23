# `compound_surnames.csv.gz` — corpus-backed compound surnames

A list of multi-word Latin surname surfaces (`gonzalez lopez`, `van den berg`, `dias de deus`) that the S2 corpus
treats as one surname. Sinonym uses it in `services/compound_surnames.py` and the scalar-stage candidate
`_corpus_compound_extension` (`timo/_resolution.py`): when the parsed `middle + last` words end with a listed
surface that is longer than the parsed surname, the middle words of the surface move into the surname
(`middle = "Rivera"`, `last = "Maza"` → `last = "Rivera Maza"`). A source that already wrote the surface whole is
untouched, so both shapes key to the same block, which is the point: a person whose sources disagree about the
split no longer ends up under two surnames. The rule only ever extends a surname leftwards: it never cuts a longer
parsed surname (`Silva dos Santos`) back to a listed suffix (`dos Santos`), never changes the given name and
never undoes a reordering the parser made (`Luong / Van / Viet` → `Viet Van Luong` stays). The first arm that
allowed shrinking broke 101 ORCID people against 325 fixed on the 690k index; the no-shrink form is what ships.

Two tiers, column `tier`:

- `listed`: surfaces the corpus writes whole often enough *and* that distinct ORCID people carry in both shapes.
- `vocabulary`: the Hispanic / Lusophone double-surname convention generalized from the listed tier. A 2-word
  pair whose head word opens at least 5 listed pairs (`garcia`, `rodriguez`, `rivera`, … 72 words) and whose
  tail word closes at least 5 (`lopez`, `garcia`, … 63 words), seen at least 5 times, is a double surname even
  when the pair itself was split too often to be listed (`garcia garcia`: 6,819 mentions, whole only 16 % of the
  time; `garcia lopez`). Pairs with a Vietnamese `van X` / `thi X` shape or an East Asian word are never admitted
  in either tier. 690k arm, production key, on top of the W3 tree: listed tier alone 139 united / 12 split, with
  the vocabulary and the source-last guard **251 / 21** (387 / 27 vs sinonym 0.5.1); shared-block ORCID pairs down.
  Rejected variant: letting any plain surname word be the tail (36k pairs, covers `rivera maza`) added 268
  united but 105 split (2.6:1): when a pair is rare, the person's other sources usually abbreviate its head to an
  initial (`Lara / G. / Merino` vs `Lara / Garcia / Merino`), so joining splits them. A pair the corpus itself
  rarely writes whole is not worth joining; `Silvia / Rivera / Maza` stays a known limit of this approach.

Columns: `surface` (folded: lower-case, accents stripped, single spaces), `tier`, `mentions`, `whole_share`
(share of mentions that wrote the surface whole in `last`), `given_names` (distinct given names seen with it),
`people` (distinct recurring-ORCID people carrying it; listed tier only), `people_whole`, `people_split_today`
(people seen with both shapes, i.e. split across production block keys before this rule). No names of people and
no identifiers.

## How it was built (2026-10-22, October snapshot) and how to rebuild from a later dump

All steps are read-only and local except the first dump. Scripts live in
`/Users/atalyaa/Documents/repos/author-names/pp_vys_orcid_conc/scripts/` unless noted; the final step is in this repo.

1. **Source-name counts.** One Athena pass over a post-backfill `paper_authors` snapshot
   (`author-names/NAME_COUNTS.md`, `pp_vys_orcid_conc/scripts/v4/athena_name_counts.sh`): one row per distinct
   source triple, where the source name is **all** `original_*` fields when `original_last IS NOT NULL`, else the
   stored fields (never a per-field coalesce — it leaks sinonym's own middle into rewritten rows). Counts: mentions,
   distinct papers, distinct ORCIDs; no production `author_id`. 113.6M triples, 702.9M mentions (Oct 2026).
2. (The May 2026 build used a pre-sinonym dump, `pre_name_counts.py`; same schema.)
3. **Partition statistics.** `partition_prior.py "<out_dir>/*.parquet" partition_prior.parquet` → for each 2–4-word
   surname tail (`middle` words + `last` words), how often the corpus puts k words in `middle`; `share_whole`
   = k = 0. Also writes `partition_prior_tokens.parquet`: per word, mentions as a given name vs as the surname word.
4. **Candidate universe.** Mention-count thresholds (in `build_compound_surnames.py`): 2–3 words, ≥ 50 mentions,
   `share_whole` ≥ 0.25, ≥ 5 distinct given names, head words with a given-name share ≤ 0.3, every word ≥ 2
   letters, no suffix tail (`junior`, `filho`, …), no East Asian surname head (Vietnamese `Nguyen Van`, Chinese
   `Liu Yang` are not compounds), no Vietnamese `van X` / `thi X` pair (`Van Le`, `Van Tran`), no Chinese compound
   head (`Au Yeung`, `Ou Yang` stay with sinonym's curated list).
   Kill test before anything ships: `sim_compound_canon.py --asset <csv.gz> --no-shrink --half A|B` applies the
   shipped rule to an existing routed arm and scores people united / split under the production block key on
   the held-out ORCID half. Without `--no-shrink` the simulation also cuts longer surnames down to the surface
   and over-counts (2026-10-21: 446 / 17) — the shipped rule's own numbers are in step 7.
5. **Person-level gate.** `scripts/v4/compound_orcid_gate_v4.py` over dataset v4 (every ORCID mention with its source
   name), excluding the ORCIDs of the 690k evaluation index: per surface, distinct people with the whole form, the
   split form, and both. `HALF=A` builds it on one ORCID half (last ORCID character not in 1 3 5 7 9 X) so the other
   half can score the asset; `HALF=all` for the shipped build.
6. **Asset.** `uv run python scripts/build_compound_surnames.py --partition partition_prior.parquet --tokens
   partition_prior_tokens.parquet --orcid-gate compound_orcid_gate.parquet --output sinonym/data/compound_surnames.csv.gz`
   keeps the `listed` surfaces where both signals agree (≥ 3 people, ≥ 2 of them whole), derives the
   `vocabulary` head and tail words from the listed 2-word pairs (≥ 5 pairs each) and adds every unlisted 2-word
   pair of the partition statistics built from them (≥ 5 mentions, same head-word given-name gate), then prints
   the sha256. 2026-10-22 build: 4,564 listed + 2,534 vocabulary surfaces (72 KB).
7. **Acceptance.** Route the 690k index with the new build (`run_690k_*.sh`), score with `conc_s2andfix.py`
   (FIXED/BROKE ≥ 5:1 on half B, no shared-block growth), run the detector (`weird_bugs.py`) and a blind judge
   sample (`judge_changes.py`), and the test suite at its baseline.

Known limits: sources that drop a compound's head entirely (`Maza` alone) cannot be joined by any rule; the
mention counts inherit MAG-style conventions that split particles into `middle`, which is why the whole-share
threshold is low and the person-level gate is required; a few surfaces are one prolific person's name rather than
a compound (harmless for keys, visible in `given_names`).
