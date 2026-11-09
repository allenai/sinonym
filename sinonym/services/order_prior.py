"""Corpus order prior for two-token Latin names.

`two_token_order_prior.csv.gz` holds, per unordered token pair, how many pre-normalization author mentions in the
S2 corpus (May 2026 snapshot, 691M rows) were written as ``first=a last=b`` versus ``first=b last=a``. Only pairs
with >= 100 mentions and a >= 80% majority that the Chinese parser can read are kept. The prior settles the
orientation of an ambiguous two-token Chinese name (`Xi Lu` / `Lu Xi`) by how the corpus itself writes that pair,
which keeps every mention of the pair on one surname. In TIMO it applies only when the router picks a parse without a
venue pool (PP-only routing); the canonical APIs use it to keep the source order of a reversed two-token Chinese name. On
abstain the source order stands: the corpus majority overrode the person's own surname far more often than it fixed it
(ORCID registry: 19,574 broken vs 2,275 fixed). With a venue pool the router has already weighed the venue's convention,
and the prior overrode it wrongly about twice as often as rightly even when it required an 86% majority (publisher
ORCID-linked author rows routed with production-style venue pools, against the registered family name: 2,675 surnames
fixed vs 5,685 broken). The cost is block consistency: without the prior some mentions of a pair keep the other
orientation, so more of a person's papers land in different disambiguation blocks (863 more people split in the same
evaluation).
"""
from __future__ import annotations

import csv
import gzip
import io
import unicodedata
from functools import lru_cache

from sinonym.resources import read_bytes

PRIOR_MIN_MENTIONS = 100
PRIOR_MIN_SHARE = 0.8


def _fold(token: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", token).casefold() if ch.isalnum())


@lru_cache(maxsize=1)
def _table() -> dict[tuple[str, str], tuple[str, float, int]]:
    """(a, b) with a < b -> (majority orientation 'ab' | 'ba', share, mentions)."""
    rows: dict[tuple[str, str], tuple[str, float, int]] = {}
    text = gzip.decompress(read_bytes("two_token_order_prior.csv.gz")).decode("utf-8")
    for row in csv.DictReader(io.StringIO(text)):
        n_ab, n_ba = int(float(row["n_ab"])), int(float(row["n_ba"]))
        total = n_ab + n_ba
        if total < PRIOR_MIN_MENTIONS:
            continue
        majority, share = ("ab", n_ab / total) if n_ab >= n_ba else ("ba", n_ba / total)
        if share >= PRIOR_MIN_SHARE:
            rows[(row["a"], row["b"])] = (majority, share, total)
    return rows


def load_order_prior() -> None:
    _table()


def corpus_surname_for(given: str, surname: str) -> str | None:
    """Return the token the corpus writes as the surname for this pair, or None when the prior is silent."""
    g, s = _fold(given), _fold(surname)
    if not g or not s or g == s:
        return None
    a, b = (g, s) if g < s else (s, g)
    entry = _table().get((a, b))
    if entry is None:
        return None
    majority = entry[0]
    corpus_surname = b if majority == "ab" else a
    return surname if corpus_surname == s else given
