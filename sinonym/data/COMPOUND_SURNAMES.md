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

Tiers, column `tier`:

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
- `corpus`: every 2–3-word tail the corpus itself writes whole in `last` at least 5 × 0.15 times (≥ 5 mentions,
  `whole_share` ≥ 0.15, ≥ 2 given names), with the listed tier's head-word, East Asian, Vietnamese and suffix gates.
  On the 690k index 60 % of the "surname word dropped" splits were boundary placements whose surface was in the
  corpus but under the listed thresholds; this tier is the largest single gain (690k: +1,354 united / 120 split at
  floor 0.2; every ORCID: 16,053 / 1,587). The floor went from 0.2 to 0.15 on 2026-10-23: at corpus scale the
  0.15–0.2 band reunites 3 ORCID people per 1 it splits (2,303 fewer split people) and 60 % of the people it touches
  registered the two-word family name with ORCID; the 0.1–0.15 band (1.9:1, 52 %) stays out.

- `hyphen`: 2–3-word surfaces the corpus writes hyphenated in `last_name` (`Ferro-Luzzi`, `Cohen-Tanugi`) for
  given names with at least 2 distinct initials (`scripts/build_hyphen_compound_surnames.py`). The hyphen declares
  the words one surname; the initials bar keeps one person, whose sources share an initial, from admitting their own
  surname (lowering the corpus tier's thresholds instead admits rare surfaces mostly on the evaluated person's own
  mentions). Counting the hyphenated mentions as whole, the surface must still be written whole at least 0.15 of the
  time (the corpus tier's floor): `Kumar-Singh` is hyphenated by a few people but `X / Kumar / Singh` is written 59k
  times, and without this gate the S2AND v5 replay split 20 gold clusters (Kumar, Pal, Alves, Holm middles) for 11
  reunited. Same suffix, East Asian, Chinese compound, Vietnamese,
  one-letter and head-word given-name gates as the other tiers, and surfaces made only of East Asian syllables are
  left out (surname-first sources put hyphenated given names such as `Zhong-Hua` in `last_name`). For this tier
  `mentions` counts the hyphenated mentions, `whole_share` includes them as whole, and `given_names` is the distinct
  given-name initials. A hyphen is any of `-`, U+2010, U+2011, U+2012, U+2013 (en dash) and U+2212 (minus sign).
  2026-10-31 build from the October name counts (`name_counts_20261021_v2`, with the `oct2` partition and token
  priors): 112,671 surfaces not already in another tier. Rows whose `first_name` is an East Asian surname (two or
  more letters) are not evidence: a surname-first source puts the hyphenated given name in `last_name`
  (`Wang / Cai-hon`, `Sakamoto / Shin-ichi`).

Construct prefixes (`abdul`, `abd`, `abdel`, `abou`, `ait`, `md`, `ab`, `mat`, `bani`, `mohd`, `mohamad`, `roy`) are exempt from the head-word given-name gate like particles: `Abdul Rahman`, `Mohd Noor`,
`Md Nor`, `Ait Brahim`, `Roy Chowdhury` open a family name even though each word is also a frequent given name.
Chosen on the ORCID registry: of the 23k split people carrying such surfaces, 73–100 % per prefix registered the
two-word family name; `ul`, `do`, `ali`, `ahmad`, `maria` (24–68 %) stay out.

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
   the sha256, and adds the `corpus` tier. 2026-10-23 October build (floor 0.15, construct prefixes): 7,240 listed
   + 1,719 vocabulary + 167,475 corpus surfaces.
7. **Acceptance.** Route the 690k index with the new build (`run_690k_*.sh`), score with `conc_arms.py`
   (FIXED/BROKE ≥ 5:1 on half B, no shared-block growth), run the detector (`weird_bugs.py`) and a blind judge
   sample (`judge_changes.py`), and the test suite at its baseline. Then the corpus-scale run (every ORCID,
   `author-names/ORCID_V4.md`): `scripts/v4/broke_v4.py` attributes every united / split person to the rule that
   moved it, and `scripts/v4/registry_agreement.py` scores each arm's surnames against the ORCID registry's
   self-declared family names (the name ground truth; the corpus share is a source convention, not a truth).

Known limits: sources that drop a compound's head entirely (`Maza` alone) cannot be joined by any rule; the
mention counts inherit MAG-style conventions that split particles into `middle`, which is why the whole-share
threshold is low and the person-level gate is required; a few surfaces are one prolific person's name rather than
a compound (harmless for keys, visible in `given_names`).

Excluded heads (2026-10-31): `mohamed`, `mohammad` and `bin` open no compound surname in any tier
(`EXCLUDED_HEADS` in `services/compound_surnames.py`, also left out of the builders' prefixes and particles; 1,186
surfaces removed from the asset). Across every tier, ORCID people who write such a surface split registered the last
word alone more often than both words (`mohamed` 40% / 32%, `mohammad` 42% / 36%, `bin` 58% / 29%): the head is a
given name or a patronymic (`Ahmed / Mohamed / Ali`, `Ahmad / bin / Ismail`). The other construct prefixes register
both words (`abdul` 62% / 22%, `mohd` 81% / 9%) and stay.

## Refreshing the table: only clear evidence changes a surface (since 0.7.1)

A rebuild from a newer snapshot decides every surface by that day's evidence. Pairs that ORCID registrants write about
as often whole as split (`rodrigues gomes`, `rodrigues oliveira`, `silva santana`, `srinivasa rao`: 40-60 % of the
registrants carrying the pair register both words) then cross the majority threshold in either direction. Rebuilding
every tier from the 2026-11-12 data with unchanged thresholds fixed the family name of 4,489 ORCID people but broke
3,108, far below the 5:1 acceptance bar of step 7.

`scripts/stabilize_compound_surnames.py` therefore compares the rebuilt table with the shipped one, surface by surface.
Its evidence is, per surface, the ORCID people whose source-supplied mentions end their `middle + last` words with the
surface, split by their registered family name in the ORCID public data file: `join` when it is the surface or ends
with it, `tail` when it is the surface without its first word (the script's docstring gives the exact folding and the
input schema).

- `join + tail >= 20` and `join >= 0.67` of them: listed;
- `join + tail >= 20` and `tail >= 0.67` of them: not listed;
- anything else (ambiguous, too few registrants, or no registry evidence): the shipped status.

Thresholds were chosen on held-out people: evidence from ORCID half A (last iD character not 1 3 5 7 9 X), scored on
half B against the 0.7.0 table. Keeping the shipped status for ambiguous pairs at a 0.67 margin and 20 registrants
changed 3 held-out family names for the better and none for the worse; a 0.6 margin let the churn back in (support 5:
300 fixed, 254 broken). The alternative of dropping ambiguous pairs, so that sinonym follows each source's own field
split (`--ambiguous abstain`), lost at every setting tried (0.67 and 20: 2,702 fixed, 3,642 broken), because sources
split pairs their authors registered whole about as often as the reverse. A surface that stays listed keeps its
shipped row.

The shipped 0.7.1 table was decided on the half-A evidence as well (unlike the `HALF=all` gate of step 5), so half B
stays a held-out set for the next refresh. It is the 0.7.0 table without `silva silva` (half A: 10 registrants with
both words, 25 with `Silva`; half B: 8 and 26).
