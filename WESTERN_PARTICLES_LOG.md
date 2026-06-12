# Western surname-particle canonicalization — experiment log

> **v2** — confidence tiers + structured input (cut FP/FN). **v3** — batch routing
> for mixed Chinese+Western lists. See "## Enhancement v2" / "## Enhancement v3"
> below. v1 (string-only router) is described first for context.

Scope: sinonym changes only. This is the Western leg of a multi-stage locale router
that extends the (Chinese-first) normalizer to canonicalize Western names whose
**surname particle** (`van`, `de`, `von`, `del`, `da`, `du`, `al`, …) is split off
from the surname by some upstream source. Opt-in; the default Chinese contract is
unchanged.

## Motivation
A corpus study of 9.6M multi-source papers found **797K authors** where a surname
particle sits in the `middle`/given field in one source (`mid=[van] last=Hout`) but
inside `last` in another (`last="van Hout"`); `de` + `van` alone = 36% of these.
Two jobs followed: (A) align them for SELECTION in the upstream Scala harness
(stage-1, done there), and (B) — this work — **canonicalize the emitted name** so the
convention-correct compound form (`van Hout`) is produced. Job B lives here because
it is per-name. The two share one particle list.

## What changed
- `sinonym/data/surname_particles.csv` — 26 curated nobiliary/patronymic particles
  (van, von, der, den, vander, de, del, della, di, da, dos, das, do, du, la, le, el,
  al, ten, ter, des, bin, ibn, abu, ben, st). Source of truth shared with the
  stage-1 harness. Single letters and real surnames (kumar/silva/garcia) excluded.
- `sinonym/data/western_surnames.csv` — 60,442 high-confidence surnames (see Split).
- `sinonym/services/western_particle.py` — `load_western_particles()`,
  `load_western_surnames()`, `try_western_particle(tokens, particles, surnames)`.
  Pure-rule, ML-free.
- `sinonym/detector.py` — new `ChineseNameDetector(enable_western_particles=False)`
  flag; when ON, `normalize_name` routes through the Western router BEFORE the
  non-Chinese rejection, and is SKIPPED for all-Chinese input (so romanized pinyin
  syllables like `de`/`da` are not hijacked).
- `tests/data/western_particle_cases.csv` (72 rows) + `tests/test_western_surname_particles.py`.

## Behaviour (flag ON)
- `Roeland van Hout` → `Roeland van Hout`; `Vincent Van Gogh` → `Vincent van Gogh`
  (particle lowercased, other tokens keep input casing).
- `Charles de Gaulle`, `Leonardo da Vinci`, `Otto von Bismarck`, `Juan de la Cruz`
  (consecutive particles), `Ahmed bin Salman`, … all canonicalize.
- DEFAULT (flag OFF): every one of these is still rejected — contract unchanged,
  zero existing-test flips.

### Router rule
Pattern = `[given …] PARTICLE [surname core …]`. The FIRST particle token (that has
a non-particle given before it and a non-particle core after it) is the particle
boundary. Particle(s) from there on are lowercased; all other tokens keep casing.
The formatted string is `" ".join(tokens)` with that lowercasing — **order
preserving**.

## Split refinement (Lusophone/Hispanic two-surname)
Problem: pre-particle tokens may be SURNAMES, not given names —
`Osmar Luiz Ferreira de Carvalho`: `Ferreira` is a paternal surname, so the ideal
split is given `Osmar Luiz` / surname `Ferreira de Carvalho`, but the particle
boundary alone treats everything before `de` as given.

Fix: a corpus-derived surname dictionary. `try_western_particle` walks LEFT from the
particle, absorbing high-confidence surnames into the surname (keeping ≥1 given). It
changes ONLY the parsed `given_name`/`surname` components — the formatted STRING is
unaffected (token order preserved).

### Surname dictionary — data + calibration
Built from `paper_authors_full` (**691M** author rows): per token, count occurrences
in the `first_name` vs `last_name` field; keep tokens with `last_ct ≥ 1000` AND
`last/(first+last) ≥ 0.90` → 60,442 surnames (`data/western_surnames.csv`).

The corpus's per-row first/last split is itself noisy upstream-splitter output, so
it is NOT treated as ground truth — only as an aggregate frequency signal. The high
ratio gate is robust to that noise because the bulk of occurrences are correct and
the classes separate cleanly:

| token | last/(first+last) | class |
|---|--:|---|
| ferreira / carvalho / silva / santos / souza / oliveira | 0.976–0.990 | surname |
| garcia / lopez / gonzalez / martinez / rodriguez | 0.977–0.978 | surname |
| esch / hout / gogh / casabianca | 0.99 | surname |
| **luiz / antonio** | **0.05** | given |
| **carlos / jose / maria / osmar** | **0.02–0.04** | given |

A clean gulf (surnames > 0.93, givens < 0.18, nothing between), so the gate admits
only confident surnames; ambiguous tokens stay given (conservative — the heuristic
only ADDS surname reassignments, never removes given safety). Verified: no
given-name leakage at the gate.

Result: `Osmar Luiz Ferreira de Carvalho` → `Osmar Luiz` / `Ferreira de Carvalho`;
`Maria Garcia de Souza` → `Maria` / `Garcia de Souza`; `Pedro Silva dos Santos` →
`Pedro` / `Silva dos Santos`. Germanic unchanged (`Roeland van Hout` → `Roeland` /
`van Hout`; `Maximilian Pierer von Esch` — `Pierer` not a surname → stays given).

## Tests
`tests/test_western_surname_particles.py`, data-driven from two CSVs
(`western_particle_cases.csv` flat, `western_particle_parts_cases.csv` structured) +
structural tests. **161 passed, 10 xfail.** Full suite: **224 passed, 10 xfail,
9 failed** — the 9 failures are PRE-EXISTING on a clean checkout (env / sklearn-pickle
version mismatch), confirmed via `git stash`; zero regressions from this work.

Coverage: nobiliary folds across 9+ particle languages; 2–3 given names then a
particle; casing folds; the split refinement; the structured path (folds + already-
compound + **mis-split → fail-closed**); tier-gated ambiguous particles; negatives
incl. the DB non-particle tail (`kumar/garcia/silva/perez/gonzalez/rodriguez/martinez`
— all REJECTED, no false fold); structural (default rejects both entry points, parsed
components, Chinese not hijacked, flag isolation, middle as str or list).

## Enhancement v2 — confidence tiers + structured input (cut FP and FN)
v1 took a flat STRING and re-guessed the field split, treating every particle as
equally certain → the documented false positives/negatives. v2 adds two levers,
both **fail-closed** (when unsure, return failure → caller keeps its own split).

**(a) Particle confidence tiers (data-driven).** `surname_particles.csv` gains
`first_ratio` + `tier` columns (from the 691M-row corpus): a particle is `ambiguous`
when it is a given name >=30% of the time, else `safe`. AMBIGUOUS =
`{di, do, le, des, bin, ibn, abu, ben, st}` (e.g. `ben` 0.62, `bin` 0.78 — `Bin` is a
common Chinese given name); SAFE = the rest (`de` 0.09, `van` 0.12, `den` 0.05, …).
Both paths refuse to fold a run containing an ambiguous particle. Effect: the flat
path now REJECTS `John Ben Carter`/`Nguyen Le Minh` (were false positives) instead of
mis-folding. Tradeoff: Arabic patronymics `bin`/`ibn`/`abu` are ambiguous in this
(academic, China-heavy) corpus, so `Ahmed bin Salman` fails closed rather than folds.

**(b) Structured entry point** `ChineseNameDetector.normalize_name_parts(first,
middle, last)` (opt-in; `try_western_particle_parts`). Uses the upstream
first/middle/last split as a strong PRIOR — **not** ground truth, because sources
mis-split/mis-order just like Chinese name order. It folds a trailing-middle particle
run into the surname (or normalizes an already-compound last) only when corpus stats
corroborate, and DETECTS a mis-split (a given name in the surname field, a surname in
the first field) → fails closed. New `western_givennames.csv` (18,716 high-confidence
given names, first-ratio >=0.90 over the corpus) backs the conflict detection.

A `use_prior` flag (default `True`) toggles this: `use_prior=False` IGNORES the
supplied split, concatenates the fields, and re-derives via the flat string router
(== `normalize_name` of the joined name) — for callers whose upstream split is
unreliable. E.g. `(Della,"","de Souza")` → `Della de Souza` with the prior, but
fails (flat leading-particle FN) without it.

Effect on the original failures:
- FN `Della de Souza` — fixed: arriving as `(Della, "", "de Souza")`, `Della` is never
  mistaken for a boundary → already-compound → `Della de Souza`.
- FP `John Ben Carter` — fixed: `(John, [Ben], Carter)` → `ben` ambiguous → fail closed.
- Mis-split `(Silva, "", "Maria")` / `(Ferreira, "", "John")` → fail closed (never a
  wrong canonical) instead of emitting garbage.
- `(Osmar,"Luiz Ferreira","de Carvalho")` → unchanged string; source split trusted.

The contract is **success ⇔ a confident fold (or already-canonical compound)**;
everything else returns `ParseResult.failure(reason)`. No silent / low-confidence
folds on either path.

## Enhancement v3 — batch routing for mixed Chinese + Western lists
The in-process batch APIs (`analyze_name_batch`/`process_name_batch`) run a separate
Chinese-oriented parser and didn't invoke the Western router, so a mixed batch
normalized the Chinese names but FAILED the Western ones. Fix reuses the rejection
the batch parser already produces: `batch_analysis` runs an ethnicity pre-filter per
name and emits `classify_ethnicity`'s `"no Chinese evidence found"` for non-Chinese
names (and excludes them from the surname-first/given-first vote — so no format-vote
skew). A small ADDITIVE post-pass in the detector wrapper
(`_apply_western_to_batch`, using `_western_fallback`) replaces a per-name
non-Chinese failure with a CONFIDENT Western fold when the flag is on:
```
process_name_batch(["Zhang Wei","Roeland van Hout","Charles de Gaulle",
                    "Wang Xiaoli","John Smith","Ahmed bin Salman"])
  Zhang Wei        -> Wei Zhang          (chinese)
  Roeland van Hout -> Roeland van Hout   (western fold)
  Charles de Gaulle-> Charles de Gaulle  (western fold)
  Wang Xiaoli      -> Xiao-Li Wang       (chinese)
  John Smith       -> FAIL               (default kept — neither)
  Ahmed bin Salman -> FAIL               (ambiguous particle, fail closed)
```
**Default outcome preserved:** a name that is neither confidently Chinese nor a
confident fold keeps its existing default (the failure); the post-pass never passes
through an identity result and never changes the flag-off contract. `process_name_batch`
inherits via `analyze_name_batch`; `process_name_batch_multiprocess` already routed
through `normalize_name` (Western worked there already). Out of scope: a structured
batch (list of first/middle/last) / `use_prior` in batch.

## Limitations — fixed in v2 vs residual
A token like `de` / `van` / `ben` / `le` is BOTH a surname particle AND, for some
people, a real first/middle given name or an integral capitalized surname element.
The closed list alone cannot tell which role applies; v2 mitigates with tiers +
structured input + fail-closed, but some residue remains.

**FIXED in v2:**
- FP from AMBIGUOUS particles — `John Ben Carter`, `Nguyen Le Minh`: `ben`/`le` are
  ambiguous tier → no fold → REJECTED (flat + structured). ✓
- FN from a leading particle-token-given — `Della de Souza`, `Ben de Vries`: the FLAT
  path still rejects (leading particle → boundary 0), but the STRUCTURED path fixes
  it (`(Della,"","de Souza")` → `Della de Souza`). ✓
- Mis-split source — `(Silva,"","Maria")` etc.: detected → fail closed (never a wrong
  canonical). ✓

**RESIDUAL (still xfail):**
- FP from a SAFE particle that is a given name in context — `John Van Smith` →
  `John van Smith`, `Tran Van Thanh` (Vietnamese middle). `van` is 88% surname-side in
  the corpus so it stays SAFE and folds; statistically defensible but wrong here.
- CASING — integral capitalized surnames: `Robert De Niro`→`de Niro`,
  `Jean-Claude Van Damme`→`van Damme` (safe `de`/`van` fold + lowercase). `Le Guin`/
  `Le Roy` now REJECT (le ambiguous) rather than fold — still not the ideal
  caps-preserving fold.
- FN that needs reorder — flat `Della de Souza` (only the structured path fixes it).

All residual cases are recorded as `xfail(strict=False)` so the suite stays green
while tracking the
gap; if the logic later improves they xpass.

## Files
- `sinonym/services/western_particle.py` (loaders, `try_western_particle` flat,
  `try_western_particle_parts` structured)
- `sinonym/detector.py` (flag, router wiring, `normalize_name_parts`, imports)
- `sinonym/data/surname_particles.csv` (+ `first_ratio`,`tier`),
  `sinonym/data/western_surnames.csv`, `sinonym/data/western_givennames.csv`
- `tests/test_western_surname_particles.py`,
  `tests/data/western_particle_cases.csv` (flat),
  `tests/data/western_particle_parts_cases.csv` (structured)
