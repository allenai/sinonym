"""Dotted initials, including Slavic digraph initials (``Ya.``, ``Zh.``), before a single surname word.

``Ya. B. Beikin`` has exactly one reading: ``Ya.`` is a transliterated Я initial. Read as Chinese, ``Ya`` becomes the
surname and ``Beikin`` is cut into pinyin syllables. Shared by the TIMO writer and the canonical APIs.
"""
from __future__ import annotations

import re

DIGRAPH_INITIALS = frozenset({"Ya", "Yu", "Ye", "Yo", "Zh", "Ch", "Sh", "Kh", "Ts", "Ia", "Iu", "Dz", "Dj", "Lj", "Nj"})
_DOTTED_SHORT_INITIAL_RE = re.compile(r"[A-Z][a-z]?\.")
_PLAIN_SURNAME_TOKEN_RE = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", re.UNICODE)


def is_dotted_initial(token: str) -> bool:
    """``A.`` or a dotted Slavic digraph initial (``Ya.``, ``Zh.``); never a title (``Dr.``, ``St.``)."""
    if not _DOTTED_SHORT_INITIAL_RE.fullmatch(token):
        return False
    return len(token) == 2 or token[:-1] in DIGRAPH_INITIALS  # noqa: PLR2004


def is_digraph_initials_surname(personal: list[str], surname: str) -> bool:
    """Every personal token is a dotted initial, at least one a digraph, and the surname is one plain word.

    Single-letter initials alone stay with the Chinese pipeline (``J. J. Zhu`` -> ``J.-J. Zhu``).
    """
    if not personal or not surname or " " in surname:
        return False
    if not all(is_dotted_initial(token) for token in personal):
        return False
    if not any(token[:-1] in DIGRAPH_INITIALS for token in personal):
        return False
    return bool(_PLAIN_SURNAME_TOKEN_RE.fullmatch(surname)) and len(surname) >= 3  # noqa: PLR2004
