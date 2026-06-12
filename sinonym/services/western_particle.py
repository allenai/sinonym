"""Western surname-particle canonicalisation (opt-in, rule-based, ML-free).

Sinonym is Chinese-first and by default REJECTS non-Chinese names. When the
detector is constructed with ``enable_western_particles=True``, a name that looks
like a Western *given + nobiliary/patronymic particle + surname* (e.g.
"Roeland van Hout", "Osmar Luiz Ferreira de Carvalho", "Maximilian Pierer von Esch")
is canonicalised: the particle is folded into the surname so the convention-correct
compound form is emitted.

Two entry points:
- ``try_western_particle(tokens, particles, surnames)`` — FLAT string path. Re-guesses
  the given/surname boundary from a token list; used by ``normalize_name``.
- ``try_western_particle_parts(first, middle, last, particles, surnames, givens)`` —
  STRUCTURED path. Uses the upstream first/middle/last split as a strong PRIOR
  (NOT ground truth) and folds only when corpus token statistics corroborate; used by
  ``normalize_name_parts``.

Both are purely rule-based and FAIL CLOSED: when not confident (ambiguous particle,
contradictory split, or nothing to fold) they return None / ParseResult.failure so the
caller keeps its own name unchanged. Output convention: particle(s) lowercased, other
tokens keep their input casing.

Confidence inputs (all corpus-derived, shared with the stage-1 harness where relevant):
- particles: dict ``particle -> tier`` (``safe`` | ``ambiguous``) from
  surname_particles.csv. A particle is ``ambiguous`` when it is a given name >=30% of
  the time in the corpus (ben/bin/le/di/do/des/ibn/abu/st) — folding it needs more
  than the closed list.
- surnames: high-confidence surname set (western_surnames.csv).
- givens: high-confidence given-name set (western_givennames.csv) — used to detect a
  mis-split source (a given name sitting in the surname field, etc.).
"""
from __future__ import annotations

from sinonym.coretypes import ParseResult
from sinonym.coretypes.results import ParsedName
from sinonym.resources import open_csv_reader

_PARTICLES_CACHE: dict[str, str] | None = None
_SURNAMES_CACHE: frozenset[str] | None = None
_GIVENNAMES_CACHE: frozenset[str] | None = None


def load_western_particles() -> dict[str, str]:
    """Load the shared particle set as ``{particle: tier}`` (cached). Source of truth:
    author-names/data/surname_particles.csv, vendored to sinonym/data/. `tier` is
    ``safe`` or ``ambiguous`` (column added from corpus first-name ratio)."""
    global _PARTICLES_CACHE
    if _PARTICLES_CACHE is None:
        _PARTICLES_CACHE = {
            row["particle"].strip().lower(): (row.get("tier") or "safe").strip().lower()
            for row in open_csv_reader("surname_particles.csv")
            if row.get("particle", "").strip()
        }
    return _PARTICLES_CACHE


def load_western_surnames() -> frozenset[str]:
    """High-confidence Western surname set (cached). From paper_authors_full (691M
    rows): tokens with last_name-field count >=1000 AND last/(first+last) ratio >=0.90.
    Robust to the corpus's noisy per-row split (true surnames sit far above given
    names). See data/western_surnames.csv."""
    global _SURNAMES_CACHE
    if _SURNAMES_CACHE is None:
        _SURNAMES_CACHE = frozenset(
            row["surname"].strip().lower()
            for row in open_csv_reader("western_surnames.csv")
            if row.get("surname", "").strip()
        )
    return _SURNAMES_CACHE


def load_western_givennames() -> frozenset[str]:
    """High-confidence Western given-name set (cached). From paper_authors_full:
    tokens with first_name-field count >=1000 AND first/(first+last) ratio >=0.90.
    Used to detect a mis-split source (e.g. a given name in the surname field)."""
    global _GIVENNAMES_CACHE
    if _GIVENNAMES_CACHE is None:
        _GIVENNAMES_CACHE = frozenset(
            row["given"].strip().lower()
            for row in open_csv_reader("western_givennames.csv")
            if row.get("given", "").strip()
        )
    return _GIVENNAMES_CACHE


def _toks(value) -> list[str]:
    """Normalise a field (str | list | None) to a list of whitespace tokens."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parts = []
        for v in value:
            parts.extend(str(v).split())
        return [t for t in parts if t.strip()]
    return [t for t in str(value).split() if t.strip()]


def join_name_parts(first, middle=None, last="") -> str:
    """Flatten first/middle/last fields back into a single ordered name string
    (used for the no-prior path that ignores the supplied split)."""
    return " ".join(_toks(first) + _toks(middle) + _toks(last))


def _surname_like(tok: str, surnames: frozenset[str], givens: frozenset[str]) -> bool:
    """A token is surname-like if it is a known surname, or at least not a known
    given name (conservative — unknown tokens are allowed as surname cores)."""
    return tok in surnames or tok not in givens


def try_western_particle(
    tokens,
    particles: dict[str, str],
    surnames: frozenset[str] = frozenset(),
) -> ParseResult | None:
    """FLAT path. If `tokens` form a Western given+particle+surname name, fold the
    particle into the surname and return a canonical ParseResult; otherwise None.

    Pattern: >=1 given token, then a PARTICLE (the first one = the particle boundary),
    then a non-particle surname core. Particle(s) from the boundary on are lowercased.
    Tier gate: if ANY particle in the folded run is AMBIGUOUS, return None (fail
    closed) — the flat path has no extra evidence to disambiguate it.

    `surnames` (optional) refines the given/surname split for multi-surname names:
    high-confidence pre-particle surnames are pulled into the surname (keeps >=1
    given). This never changes the formatted string (token order preserved).
    """
    toks = [t for t in tokens if t and t.strip()]
    if len(toks) < 3:
        return None

    low = [t.lower() for t in toks]
    boundary = next((i for i, t in enumerate(low) if t in particles), None)
    if boundary is None or boundary < 1:
        return None
    if not any(low[j] not in particles for j in range(boundary + 1, len(low))):
        return None

    # Tier gate: any ambiguous particle from the boundary onward → fail closed.
    if any(particles.get(low[j]) == "ambiguous" for j in range(boundary, len(low)) if low[j] in particles):
        return None

    # Refine split: absorb high-confidence pre-particle surnames (keep >=1 given).
    split = boundary
    while split > 1 and low[split - 1] in surnames and low[split - 1] not in particles:
        split -= 1

    given_tokens = list(toks[:split])
    surname_tokens = [t.lower() if t.lower() in particles else t for t in toks[split:]]
    formatted = " ".join(given_tokens + surname_tokens)
    parsed = ParsedName(
        surname=" ".join(surname_tokens),
        given_name=" ".join(given_tokens),
        surname_tokens=surname_tokens,
        given_tokens=given_tokens,
        order=["given", "surname"],
    )
    return ParseResult.success_with_name(formatted, parsed=parsed)


def try_western_particle_parts(
    first,
    middle,
    last,
    particles: dict[str, str],
    surnames: frozenset[str],
    givens: frozenset[str],
) -> ParseResult:
    """STRUCTURED path. Canonicalise a name already split into first/middle/last.

    The source split is a strong PRIOR, NOT ground truth (sources mis-split/mis-order
    just like Chinese name order). We combine the source position with corpus token
    stats and FAIL CLOSED (return failure → caller keeps its split) unless confident.

    Returns success only when a particle is confidently folded, or the surname is
    already a valid compound. Returns failure for: a contradictory split (§3b), an
    ambiguous particle, a non-surname-like core, or nothing to fold.
    """
    first_toks, mid_toks, last_toks = _toks(first), _toks(middle), _toks(last)
    if not first_toks or not last_toks:
        return ParseResult.failure("structured input needs both given and surname")

    low_first = [t.lower() for t in first_toks]
    low_last = [t.lower() for t in last_toks]
    last_core = low_last[-1]
    first_core = low_first[0]

    # --- §3b: detect a mis-split source and fail closed (never emit a wrong canonical)
    if last_core in givens and last_core not in surnames:
        return ParseResult.failure("suspect split: surname field holds a given name")
    if first_core in surnames and last_core in givens:
        return ParseResult.failure("suspect split: surname-in-first / given-in-last")

    def _success(given_tokens, surname_tokens):
        formatted = " ".join(given_tokens + surname_tokens)
        parsed = ParsedName(
            surname=" ".join(surname_tokens),
            given_name=" ".join(given_tokens),
            surname_tokens=surname_tokens,
            given_tokens=given_tokens,
            order=["given", "surname"],
        )
        return ParseResult.success_with_name(formatted, parsed=parsed)

    # --- already-compound: last begins with a SAFE particle run, core surname-like.
    if low_last[0] in particles and len(last_toks) >= 2:
        i = 0
        while i < len(last_toks) and low_last[i] in particles:
            i += 1
        run_low = low_last[:i]
        core = low_last[-1]
        if any(particles.get(p) == "ambiguous" for p in run_low) or not _surname_like(core, surnames, givens):
            return ParseResult.failure("compound surname not confidently canonical")
        new_last = [t.lower() for t in last_toks[:i]] + list(last_toks[i:])  # particles lowercased
        return _success(list(first_toks) + list(mid_toks), new_last)

    # --- split pattern: a particle run stranded at the END of the middle field.
    if mid_toks:
        low_mid = [t.lower() for t in mid_toks]
        j = len(mid_toks)
        while j > 0 and low_mid[j - 1] in particles:
            j -= 1
        run_low = low_mid[j:]
        if run_low:
            if any(particles.get(p) == "ambiguous" for p in run_low):
                return ParseResult.failure("ambiguous particle in middle; not folded")
            if not _surname_like(last_core, surnames, givens):
                return ParseResult.failure("surname core not surname-like")
            new_given = list(first_toks) + list(mid_toks[:j])
            new_surname = [t.lower() for t in mid_toks[j:]] + list(last_toks)
            return _success(new_given, new_surname)

    return ParseResult.failure("no particle to canonicalize")
