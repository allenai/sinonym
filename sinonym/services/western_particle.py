"""Western surname-particle canonicalization (opt-in, rule-based, ML-free).

Sinonym is Chinese-first and by default REJECTS non-Chinese names. When the
detector is constructed with ``enable_western_particles=True``, a name that looks
like a Western *given + nobiliary/patronymic particle + surname* (e.g.
"Roeland van Hout", "Osmar Luiz Ferreira de Carvalho", "Maximilian Pierer von Esch")
is routed here BEFORE the non-Chinese rejection and canonicalised: the particle is
folded into the surname so the convention-correct compound form is emitted.

This is the Western leg of the multi-stage locale router. It is purely rule-based
(a closed particle list shared with the stage-1 harness —
``sinonym/data/surname_particles.csv``); it never touches the ML classifier and is
inert unless the flag is set, so the Chinese contract is unchanged.

Output convention (per spec): the particle token(s) are lowercased; every other
token keeps its given (input) casing.
"""
from __future__ import annotations

from sinonym.coretypes import ParseResult
from sinonym.coretypes.results import ParsedName
from sinonym.resources import open_csv_reader

_PARTICLES_CACHE: frozenset[str] | None = None
_SURNAMES_CACHE: frozenset[str] | None = None


def load_western_particles() -> frozenset[str]:
    """Load the shared particle set from package data (cached). Source of truth:
    author-names/data/surname_particles.csv, vendored to sinonym/data/."""
    global _PARTICLES_CACHE
    if _PARTICLES_CACHE is None:
        _PARTICLES_CACHE = frozenset(
            row["particle"].strip().lower()
            for row in open_csv_reader("surname_particles.csv")
            if row.get("particle", "").strip()
        )
    return _PARTICLES_CACHE


def load_western_surnames() -> frozenset[str]:
    """Load the high-confidence Western surname set (cached). Derived from the
    691M-row paper_authors_full corpus: tokens with last_name-field frequency
    >=1000 AND last/(first+last) ratio >=0.90 (see data/western_surnames.csv).
    Used ONLY to refine the given/surname boundary for multi-surname names
    (Lusophone/Hispanic), never the formatted string. The high ratio gate makes it
    robust to the corpus's noisy per-row split (given-name tokens like luiz/carlos
    score ~0.05 and are excluded)."""
    global _SURNAMES_CACHE
    if _SURNAMES_CACHE is None:
        _SURNAMES_CACHE = frozenset(
            row["surname"].strip().lower()
            for row in open_csv_reader("western_surnames.csv")
            if row.get("surname", "").strip()
        )
    return _SURNAMES_CACHE


def try_western_particle(
    tokens,
    particles: frozenset[str],
    surnames: frozenset[str] = frozenset(),
) -> ParseResult | None:
    """If `tokens` form a Western given+particle+surname name, fold the particle
    into the surname and return a canonical ParseResult; otherwise return None
    (caller falls through to the normal Chinese / rejection path).

    Pattern: at least one GIVEN token, then a PARTICLE, then at least one surname
    CORE token. The first such particle marks the PARTICLE boundary; particle(s)
    from there on are lowercased, all other tokens keep their input casing.

    `surnames` (optional) refines the given/surname SPLIT for multi-surname names:
    pre-particle tokens that are high-confidence surnames are pulled into the
    surname (e.g. "Osmar Luiz Ferreira de Carvalho" → given "Osmar Luiz", surname
    "Ferreira de Carvalho"). This NEVER changes the formatted string (token order is
    preserved) — only the parsed components. At least one given token is always kept.
    """
    toks = [t for t in tokens if t and t.strip()]
    if len(toks) < 3:
        # Need given + particle + core. A bare "van Hout" (no given) is a surname
        # fragment, not a full name to canonicalise — leave it.
        return None

    low = [t.lower() for t in toks]
    # Particle boundary = the FIRST particle token. Require a real given token
    # before it (so a leading-particle fragment like "de la Cruz" is NOT folded)
    # and a non-particle surname core after it.
    boundary = next((i for i, t in enumerate(low) if t in particles), None)
    if boundary is None or boundary < 1:
        return None
    if not any(low[j] not in particles for j in range(boundary + 1, len(low))):
        return None

    # Refine the given/surname split: walk left from the particle, absorbing
    # high-confidence pre-particle surnames into the surname. Keep >=1 given token.
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
