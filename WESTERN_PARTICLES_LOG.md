# Western surname-particle canonicalization — experiment log

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
`tests/test_western_surname_particles.py` (data-driven from the CSV) + structural
tests. **135 passed, 12 xfail.** Full suite: **210 passed, 12 xfail, 9 failed** —
the 9 failures are PRE-EXISTING on a clean checkout (env / sklearn-pickle version
mismatch), confirmed via `git stash`; zero regressions from this work.

Coverage: nobiliary folds across 9+ particle languages; 2–3 given names then a
particle; casing folds; the split refinement; negatives including the DB
non-particle tail (`kumar/garcia/silva/perez/gonzalez/rodriguez/martinez` — all
correctly REJECTED, no false fold); structural (default still rejects every
fold-case, parsed components, Chinese names not hijacked, flag isolation).

## Known limitations (xfail — irreducible token ambiguity, would need per-name knowledge)
A token like `de` / `van` / `ben` / `al` / `le` is BOTH a surname particle AND, for
some people, a real **first/middle** given name or an integral capitalized surname
element. The rule splits a name into `given` (→ first + middle) and `surname`
(→ last) at the particle; when the particle token is actually playing a given/last
role, that split lands in the wrong field. The closed list cannot tell which role
applies without per-name knowledge.

Notation below: the INTENDED parse vs what the rule produces, by field.

**(1) FALSE POSITIVE — a particle-token sitting in the MIDDLE (a given/middle name)
is pulled into LAST and lowercased.**

| input | intended first / middle / last | rule's first / middle / last → output | error |
|---|---|---|---|
| `John Ben Carter` | John / **Ben** / Carter | John / — / **ben Carter** → `John ben Carter` | middle given `Ben` demoted into surname + lowercased |
| `John Van Smith` | John / **Van** / Smith | John / — / **van Smith** → `John van Smith` | same |
| `Tran Van Thanh` (Vietnamese) | given Thanh / **Van** / surname Tran | Tran / — / **van Thanh** → `Tran van Thanh` | `Van` is a Vietnamese middle, not a particle (router is Western-only) |
| `Nguyen Le Minh` (Vietnamese) | given Minh / **Le** / surname Nguyen | Nguyen / — / **le Minh** → `Nguyen le Minh` | same |

**(2) FALSE NEGATIVE — a particle-token used as the FIRST name lands at index 0, so
the "first particle" is the given itself → no given remains before it → REJECT.**

| input | intended first / middle / last | rule | error |
|---|---|---|---|
| `Della de Souza` | **Della** / — / de Souza | first particle = `della` @ idx0 → boundary<1 → **rejected** | a real `de Souza` fold is dropped because the first name equals a particle |
| `Ben de Vries` | **Ben** / — / de Vries | `ben` @ idx0 → **rejected** | same |
| `Al de Roma` | **Al** / — / de Roma | `al` @ idx0 → **rejected** | same |

(The "first particle = boundary" rule is REQUIRED for `de la Cruz` — given `Juan`,
last `de la Cruz` — but that same rule mis-fires here. The two requirements conflict;
no single positional rule satisfies both.)

**(3) CASING — an integral, conventionally-capitalized surname element is lowercased.**

| input | intended first / middle / last | rule's output | error |
|---|---|---|---|
| `Ursula Le Guin` | Ursula / — / **Le Guin** | `Ursula le Guin` | `Le` is integral to the surname (English caps), not a lowercase particle |
| `Robert De Niro` | Robert / — / **De Niro** | `Robert de Niro` | same |
| `Mary Le Roy` | Mary / — / **Le Roy** | `Mary le Roy` | same |
| `Jean-Claude Van Damme` | Jean-Claude / — / **Van Damme** | `Jean-Claude van Damme` | same |

(By-design lowercase-particle convention; semantically wrong for these few where the
particle is a fixed part of the surname.)

All recorded as `xfail(strict=False)` so the suite stays green while tracking the
gap; if the logic later improves they xpass.

## Files
- `sinonym/services/western_particle.py`
- `sinonym/detector.py` (flag + router wiring + imports)
- `sinonym/data/surname_particles.csv`, `sinonym/data/western_surnames.csv`
- `tests/test_western_surname_particles.py`, `tests/data/western_particle_cases.csv`
