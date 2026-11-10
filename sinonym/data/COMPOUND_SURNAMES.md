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

- `registry`: surfaces decided by the ORCID registry's self-declared family names (built after the corpus tiers and
  before the hyphen tier). For each ORCID person in one half of the registry (last ORCID character not in
  1 3 5 7 9 X, so the other half can score the result) and each 2–3-word surface that ends the `middle + last` words
  of one of their source names, the person counts as a join when the registered family name is the surface or ends
  with it as whole words, and as a tail when it is the surface without its first word. Surfaces with a one-letter
  word are not considered. A surface not already in the corpus tiers is admitted when joins ≥ 2 and outnumber tails,
  every non-final word is used as a compound-surname head at least half the time, and it passes the suffix, East
  Asian, Chinese compound and Vietnamese gates. A surface already in the corpus tiers is vetoed when joins + tails ≥ 3
  and it fails the join test or a gate. The shipped build vetoed 576 surfaces and admitted 19,422; 19,412 remain after
  the excluded heads.

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
  2026-10-31 build from the October 2026 name counts and partition priors: 112,671 surfaces not already in another
  tier (112,601 after the excluded heads). Rows whose `first_name` is an East Asian surname (two or
  more letters) are not evidence: a surname-first source puts the hyphenated given name in `last_name`
  (`Wang / Cai-hon`, `Sakamoto / Shin-ichi`).

Construct prefixes (`abdul`, `abd`, `abdel`, `abou`, `ait`, `md`, `ab`, `mat`, `bani`, `mohd`, `mohamad`, `roy`) are exempt from the head-word given-name gate like particles: `Abdul Rahman`, `Mohd Noor`,
`Md Nor`, `Ait Brahim`, `Roy Chowdhury` open a family name even though each word is also a frequent given name.
Chosen on the ORCID registry: of the 23k split people carrying such surfaces, 73–100 % per prefix registered the
two-word family name; `ul`, `do`, `ali`, `ahmad`, `maria` (24–68 %) stay out.

Columns: `surface` (folded: lower-case, accents stripped, single spaces), `tier`, `mentions`, `whole_share`
(share of mentions that wrote the surface whole in `last`), `given_names` (distinct given names seen with it),
`people` (distinct recurring-ORCID people carrying it; listed tier only), `people_whole`, `people_split_today`
(people seen with both shapes, i.e. split across production block keys before this rule). Registry rows leave
`mentions`, `whole_share` and `given_names` empty; their `people` is the registrants with a family name,
`people_whole` the joins and `people_split_today` the tails. No names of people and no identifiers.

## How it was built and how to rebuild it

The shipped table (sinonym 0.6.0 onward) was built from the October 2026 Semantic Scholar author snapshot in four
stages,
each run on the previous stage's output:

1. **Corpus tiers** (`listed`, `vocabulary`, `corpus`) with `scripts/build_compound_surnames.py`, from source-name
   counts (one row per distinct source triple: all `original_*` fields when `original_last` is set, else the stored
   fields, never a per-field coalesce; 113.6M triples, 702.9M mentions), the surname-partition statistics derived
   from them (per 2–4-word surname tail, how often the corpus puts k words in `middle`; per word, mentions as a
   given name vs a surname word) and the person-level ORCID gate (per surface, distinct ORCID people with the whole
   form, the split form and both).
2. **Registry tier**: the ORCID-registry vetoes and admissions described above.
3. **Hyphen tier** with `scripts/build_hyphen_compound_surnames.py`.
4. **Excluded heads**: surfaces opened by `EXCLUDED_HEADS` (below) are removed. The shipped build removed 1,186 here;
   the current stage-1 builder already leaves these heads out, so a rebuild removes only the few that stages 2 and 3
   admit.

The internal rebuild pipeline, [scholar `scripts/sinonym_assets`](https://github.com/allenai/scholar/tree/main/scripts/sinonym_assets), runs every stage with the recorded
parameters, documents the inputs, and checks a rebuild against the shipped table (row counts per tier, a row-level
diff, an order-independent checksum). Before shipping a rebuild, apply the shipped rule to a held-out ORCID half and
score people united / split under the production block key (target at least 5 united per 1 split, no growth in
shared-block ORCID pairs), score the resulting surnames against the ORCID registry's self-declared family names,
and run the test suite.

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
