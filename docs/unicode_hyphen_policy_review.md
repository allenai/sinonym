# Unicode hyphen policy review

This review decides which of the 31 previously proposed “hyphen-like” code
points may receive early structural ASCII-hyphen treatment in the scalar
Roman-name path, and which must remain visible only as later validation cues.
The decision is based on Unicode semantics, a full corpus census, manual
review, and executable negative controls. Membership in Unicode's `Dash`
property, or having `HYPHEN`/`MINUS` in a character name, is not by itself
evidence that two characters have the same meaning.

## Decision

The scalar Roman-name policy is:

| Action | Code points | Basis |
| --- | --- | --- |
| Baseline | U+002D | ASCII hyphen-minus |
| Fold early to `-` | U+2010, U+2011, U+FE63, U+FF0D | Hyphen semantics or compatibility-equivalent hyphen-minus |
| Fold early to `-` as reviewed metadata substitutions | U+2012, U+2013, U+2014, U+2043, U+2212 | Repeated author-metadata use as an intra-name boundary, supported by the census and manual examples below |
| Fold only after structural preprocessing | U+00AD, U+2015, U+2027, U+208B, U+FE58 | Preserve an internal boundary cue for ethnicity and validity gates without granting camel-case/concatenated-name parity or claiming Unicode equivalence |
| Do not add to either scalar fold | U+058A, U+05BE, U+1400, U+1806, U+2E17, U+2E1A, U+2E3A, U+2E3B, U+2E40, U+2E5D, U+301C, U+3030, U+30A0, U+FE31, U+FE32, U+10EAD | Script-specific punctuation, distinct semantic use, negative/ambiguous evidence, or no corpus evidence |

Only the ten baseline/early-fold characters are structural Roman hyphens. The
five later cues are deliberately absent from that set. Letting U+2015, U+2027,
U+208B, or U+FE58 reach the existing generic separator regex would turn them
into spaces before validation; deleting U+00AD early would erase the evidence
needed to reject a Korean name such as `Seung<U+00AD>Ji Lim`. The later fold
retains the current safe rejection behavior while avoiding a claim that these
characters are interchangeable with ASCII punctuation.

“Do not add” means the scalar Chinese-name expansion does not reinterpret the
mark as a Roman hyphen. It does not narrow the generic person normalizer: that
service retains its pre-existing broad Unicode heuristic for all legacy inputs.
The comparison-key deletion table retains its prior members and adds U+00AD so
all-Han and normalized-key checks still ignore that discretionary control.

The standards-backed core, corpus-backed metadata substitutions, and
validation-only cues are kept as separate constants in code. This makes each
reason visible rather than implying that Unicode defines em dashes, minus
signs, and hyphens as interchangeable.

## Corpus and method

The full scan used the current review corpus, not a sample:

- 106,873,287 distinct `nm` rows and 690,997,259 weighted mentions
- Parquet size 1,881,981,435 bytes, 533 row groups
- SHA-256 `36d6e49e31dfc09b3e98144a6d1bc44c4884abe003a0672bb00ec4c9be991ad1`
- source object: `s3://s2-atalyaa/disambiguation/author-names/sinonym-0-4-x-review/corpus/distinct_author_names.parquet`
- matching 106,873,287-row Chinese-label map SHA-256 `4992da8513e23855f51662583e015237edea0071bc907222fdfc38462bd09dd4`
- 9,520,175 rows in that aligned map have `chinese=true`
- scan revision `b39bdbe35e18104bb77bff21014e1921342e18d2`; DuckDB 1.5.5; elapsed 269.712 seconds

The exact older 48.8-million-row artifact mentioned in historical notes was
not available, so it was not silently substituted. This is deliberately a
review of the current expansion against the current authoritative corpus.

For each non-ASCII candidate, the census counts distinct names, weighted
mentions, letter-flanked occurrences, and matches in the aligned Chinese-label
map. Counts are per code point; a name containing two different marks can
therefore contribute to two rows.

The scan found 649,389 non-ASCII code-point/name hits representing 649,184
unique names; 204 names contained more than one audited mark. Their weighted
totals were 1,324,488 code-point hits and 1,324,241 unique-name mentions.

| Code point | Unicode name | Distinct | Weighted | Letter-flanked distinct | Chinese-labelled distinct | Manual J/N/U | Scalar action |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| U+002D | HYPHEN-MINUS | 7,946,361 | 41,253,235 | — | — | baseline | keep |
| U+00AD | SOFT HYPHEN | 8,376 | 10,639 | 6,927 | 129 | 10/20/0 | validation cue only |
| U+05BE | HEBREW PUNCTUATION MAQAF | 26 | 30 | 25 | 0 | 8/1/0 | no fold |
| U+2010 | HYPHEN | 544,028 | 1,182,251 | 513,794 | 162,705 | 30/0/0 | fold |
| U+2011 | NON-BREAKING HYPHEN | 10,815 | 12,527 | 10,378 | 2,685 | 30/0/0 | fold |
| U+2012 | FIGURE DASH | 858 | 1,738 | 151 | 47 | 18/3/8 | fold |
| U+2013 | EN DASH | 36,053 | 49,103 | 19,426 | 3,786 | 20/9/1 | fold |
| U+2014 | EM DASH | 5,172 | 5,713 | 2,495 | 539 | 12/18/0 | fold |
| U+2015 | HORIZONTAL BAR | 1,409 | 1,585 | 1,282 | 9 | 16/11/1 | validation cue only |
| U+2027 | HYPHENATION POINT | 124 | 133 | 101 | 10 | 1/24/0 | validation cue only |
| U+2043 | HYPHEN BULLET | 643 | 744 | 513 | 626 | 30/0/0 | fold |
| U+208B | SUBSCRIPT MINUS | 5 | 5 | 3 | 0 | 3/2/0 | validation cue only |
| U+2212 | MINUS SIGN | 37,241 | 54,120 | 35,114 | 283 | 21/9/0 | fold |
| U+301C | WAVE DASH | 24 | 24 | 12 | 1 | 0/17/1 | no fold |
| U+30A0 | KATAKANA-HIRAGANA DOUBLE HYPHEN | 1 | 2 | 1 | 0 | 1/0/0 | no fold |
| U+FE63 | SMALL HYPHEN-MINUS | 4 | 4 | 4 | 1 | 4/0/0 | fold |
| U+FF0D | FULLWIDTH HYPHEN-MINUS | 4,610 | 5,870 | 4,503 | 2,122 | 29/1/0 | fold |
| U+FE58 | SMALL EM DASH | 0 | 0 | 0 | 0 | no examples | validation cue only |
| U+058A, U+1400, U+1806, U+2E17, U+2E1A, U+2E3A, U+2E3B, U+2E40, U+2E5D, U+3030, U+FE31, U+FE32, U+10EAD | remaining candidates | 0 each | 0 each | 0 each | 0 each | no examples | no fold |

`J/N/U` means manually labelled joiner, not-joiner, or uncertain. A joiner
label says what the mark is doing in that record; it does not automatically
authorize a global scalar fold or establish that the person is Chinese. The
aligned `chinese` field is a useful sampling stratum, not a gold ethnicity
label.

## Manual adjudication

All 359 selected rows were read and labelled manually: 233 joiners, 115
not-joiners, and 11 uncertain. There were no missing decisions or duplicate
packet IDs; 345 decisions were high-confidence and 14 medium-confidence.

The first packet contained 256 rows. Within each code point and
letter-flanked/non-letter-flanked stratum, it combined up to five
highest-weighted records with five records selected by a deterministic hash.
The second packet added 103 previously unseen records from the aligned
Chinese-labelled population, again combining highest-weighted and
deterministic-hash selections. This deliberately exposed both common dirty
metadata and the target population; a sample of only apparent person names
would have hidden the negative uses.

The rubric was:

- `joiner`: the mark separates components or syllables inside a plausible
  personal name and an ASCII hyphen preserves that authored boundary;
- `not_joiner`: the mark is mathematical notation, a range, prose/layout
  punctuation, an apostrophe/middle-dot substitute, corruption, or
  script-specific punctuation that should not be recast as a Roman hyphen;
- `uncertain`: the record alone supports more than one plausible reading.

Representative positive decisions include:

| Mark | Examples | Reason |
| --- | --- | --- |
| U+2010/U+2011 | `Chen‐Yang Cai`, `Chen‑Yang Cai` | ordinary and non-breaking hyphens preserve an explicit given-name boundary |
| U+2012 | `Jang‒Woo Han` (`U+2012:6b35d2378989`), `Jae‒Jin Kim` | repeated letter-flanked author-name use despite the character's figure-dash semantics |
| U+2013 | `Ting–Ting Zhang` (`U+2013:36b787b2fd0c`), `Bi–Ni Jiang` | repeated hyphen substitution in the target metadata |
| U+2014 | `Yan—Tuan Li` (`U+2014:caf5e0022955`), `Chen Huan—chun` | all 10 targeted Chinese-labelled examples were name joiners |
| U+2043 | `Ru⁃quan Han` (`U+2043:3cc07f12684a`), `Feng⁃zeng Jian` | 30/30 reviewed examples were joiners and 626/643 corpus rows were Chinese-labelled |
| U+2212 | letter-flanked Chinese and Korean names | name use was common, while mathematical negatives were separable by existing validity gates |
| U+FE63/U+FF0D | fullwidth/small-width author names | compatibility forms of hyphen-minus; 33/34 reviewed rows were joiners |

Representative negative decisions include:

| Mark | Example class | Why it must not justify a global equivalence claim |
| --- | --- | --- |
| U+00AD | invisible marks inside otherwise intact words, for example `Antonio Pérez Martí<U+00AD>n` (`U+00AD:71d5e10b069d`) | Unicode soft hyphen is discretionary and default-ignorable, so it is not granted early structural equivalence. Ten targeted records used it as a punctuation joiner, but `Seung<U+00AD>Ji Lim` showed that deleting it before validation erases decisive Korean structure. |
| U+05BE | Hebrew names versus a mathematical expression | genuine Hebrew maqaf use belongs to the generic person path, not a Roman-name scalar expansion |
| U+2013 | `British Drama 1533–1642: A Catalogue` (`U+2013:0dca539fa47a`) | en dash frequently retains range semantics |
| U+2014 | `Subpart A—General` (`U+2014:4d346b12f966`), `CONGRESSIONAL RECORD—SENATE`, corrupted vowels | the general packet was mostly prose, roles, ranges, or corruption; only the target-population evidence supports the narrowly documented metadata substitution |
| U+2015 | layout records and quotation-style bars | only 9 corpus rows were Chinese-labelled, so the mixed manual result is insufficient for early structural expansion; keeping it as a later cue rejects `Jong―Ho Lee` as Korean |
| U+2027 | `Giuseppe D‧Alessio` and transliterated middle-dot names | mostly an apostrophe/middle-dot separator, not a hyphen; keeping it as a later cue prevents a Cyrillic component in `Д‧谢泽` from being silently discarded |
| U+208B/U+2212 | formulas and numeric notation | minus characters retain mathematical semantics outside valid names |
| U+301C | Japanese prose, ranges, and corrupted records | 17/18 reviewed examples were negative; the remaining record was uncertain |
| U+FF0D | a numeric range | compatibility equivalence does not make every containing record a person; validity gates still reject semantic negatives |

Regression tests encode ASCII parity for every approved early fold at both an
ordinary boundary and a boundary that must be visible before camel-case and
concatenated-surname preprocessing. They separately prove that the five later
cues preserve Korean/Cyrillic validity decisions in scalar and batch paths,
and encode unapproved expansions plus range, mathematics, and legal-citation
negatives. The 16 dropped marks account for only 51 corpus rows, while the
generic person path still handles their script-specific joiner use.

An exact behavioral replay of all 359 adjudicated records against revision
`b39bdbe`'s broad 31-character fold found one scalar semantic change: reviewed
joiner `Cheng—meiLi` (`U+2014:0cb44f320c44`) changes from “needs at least 2
Roman tokens” to `Mei-Li Cheng`. The Korean, Cyrillic, soft-hyphen, and
separator negatives remain unchanged.

## Verification

The frozen performance comparison used alternating clean-baseline/current
subprocesses. Revision `b39bdbe` is runtime-equivalent to the immediate
pre-policy head `823f8e7` for the files and benchmark involved.

| Path | Baseline | Current | Median delta | Median paired delta |
| --- | ---: | ---: | ---: | ---: |
| Scalar ASCII, cold | 6,105/s | 6,198/s | +1.53% | -0.31% |
| Scalar ASCII, warm | 6,939/s | 6,973/s | +0.48% | -0.31% |
| Scalar reviewed Unicode packet, warm | 3,225/s | 3,230/s | +0.17% | -0.03% |
| Generic person ASCII, warm | 4,615/s | 4,642/s | +0.58% | +0.50% |
| Generic person reviewed Unicode packet, warm | 3,239/s | 3,233/s | -0.19% | -0.08% |

Both paths are neutral at this resolution. Replacing the generic-person
normalizer's Python per-character callback with an equivalent `str.translate`
table removed the preliminary 1.1–1.3% slowdown. Across 3,000 ASCII inputs,
scalar and person outputs were byte-for-byte identical.
Every subprocess produced deterministic checksums on a second verification
pass.

The repository status gate matched all 15 known test-failure signatures and
passed its performance test at 5,521 names/second. The focused punctuation
suite has 158 passing tests, repository-wide Ruff passes, and `git diff
--check` is clean.

## Unicode basis

Unicode supplies character properties and normalization mappings, not an
application-specific instruction to replace every dash with ASCII `-`:

- [UAX #44](https://www.unicode.org/reports/tr44/) defines property semantics
  and cautions against deriving behavior from character names.
- [Unicode `PropList.txt`](https://www.unicode.org/Public/17.0.0/ucd/PropList.txt)
  identifies `Dash`, `Hyphen`, and other properties; those sets are not the
  same and neither is a name-normalization recipe.
- [UAX #15](https://www.unicode.org/reports/tr15/) defines normalization
  behavior: NFKC maps U+FE63/U+FF0D to U+002D and U+2011 to U+2010.
- [Unicode 17, Chapter 6](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-6/)
  describes U+00AD as a discretionary line-breaking control. That excludes it
  from structural equivalence; retaining it later as an internal validation
  cue is an application-specific author-metadata decision.
