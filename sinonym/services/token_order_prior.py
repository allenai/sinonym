"""Corpus role prior for single Latin name tokens.

``token_role_prior.csv.gz`` lists tokens the S2 corpus (May 2026 pre-normalization snapshot, 691M mentions) uses
almost only as a given name (``giulia``, ``sophie``) or almost only as a surname (``ferrari``, ``smith``):
at least 200 mentions and a given-name share of at most 10% or at least 90%. A two-token Latin name whose first
token is a surname word and whose last token is a given-name word was written surname-first by its source
(``Ferrari / Giulia``); swapping it keeps every mention of the person on one surname. Built by
``scripts/build_token_order_prior.py``.
"""
from __future__ import annotations

import csv
import gzip
import io
import unicodedata
from functools import lru_cache

from sinonym.resources import read_bytes


def _fold(token: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", token).casefold() if ch.isalnum())


@lru_cache(maxsize=1)
def _roles() -> dict[str, str]:
    text = gzip.decompress(read_bytes("token_role_prior.csv.gz")).decode("utf-8")
    return {row["token"]: row["role"] for row in csv.DictReader(io.StringIO(text))}


def token_role(token: str) -> str | None:
    """``"given"``, ``"surname"`` or ``None`` when the corpus has no strong opinion about this token."""
    return _roles().get(_fold(token))


def surname_first_pair(given: str, surname: str) -> bool:
    """True when the corpus says ``given`` is a surname word and ``surname`` a given-name word."""
    return token_role(given) == "surname" and token_role(surname) == "given"
