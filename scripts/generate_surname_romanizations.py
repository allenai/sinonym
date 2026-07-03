#!/usr/bin/env python3
# ruff: noqa: PLR2004, SLF001
"""Generate the romanization-conditional surname table (surname_romanizations.csv).

Enumerates, from the repo's own romanization rule tables and name inventories,
every as-written spelling whose surname lookup inherits the frequency mass of a
different Mandarin syllable, and assigns each spelling the share of that target
mass it keeps when scored as a surname:

- ``CANTONESE_SURNAMES`` keys and the curated Wade-Giles/Taiwanese surname
  spellings below are attested surname romanizations -> full share (1.0).
- Every other spelling reachable only through romanization remapping, with no
  as-written surname attestation, gets the penalty share e^-4 (~1.8%), the
  operating point measured on the 795-case eval (fixes "Leung Ka Fai" and
  "Kong Kung" with zero regressions; 2.0 log-units flips Leung Ka Fai, Kong
  Kung needs more than 3.0).

Column semantics are documented in ``sinonym/data/README.md``. The scoring
hook (``NameParsingService``) consumes only ``spelling`` and ``target_share``;
``mandarin_target`` and ``surname_ppm_as_written`` document the seeding so a
future corpus-derived estimate (ORCID-with-country) can replace the numbers
without a schema change.

Usage:
    uv run python scripts/generate_surname_romanizations.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from sinonym import ChineseNameDetector
from sinonym.chinese_names_data import CANTONESE_SURNAMES, ROMANIZATION_EXCEPTIONS, SYLLABLE_RULES

OUTPUT_NAME = "surname_romanizations.csv"

# Measured operating point: remap-only spellings keep e^-4 of the target mass.
PENALTY_SHARE = math.exp(-4.0)

# Wade-Giles/Taiwanese/Hokkien spellings that genuinely romanize surnames but
# are never listed as-written in the surname tables (spelling -> primary
# Mandarin surname target).
WADE_GILES_SURNAME_TARGETS = {
    "tsai": "cai",  # 蔡 (Taiwan's most common surname romanization)
    "tsao": "cao",  # 曹
    "tseng": "zeng",  # 曾
    "chou": "zhou",  # 周
    "chiang": "jiang",  # 蒋
    "kuo": "guo",  # 郭
    "hsieh": "xie",  # 谢
    "hsiung": "xiong",  # 熊
    "yeh": "ye",  # 叶
    "teng": "deng",  # 邓
    "chien": "qian",  # 钱/简
    "chin": "jin",  # 秦/金
    "chia": "jia",  # 贾
    "tuan": "duan",  # 段
    "ting": "ding",  # 丁 (Taiwan)
    "chuang": "zhuang",  # 庄 (Taiwan)
    # NOT included: 'kung' (vanishingly rare as a modern as-written surname vs
    # its given-syllable use), 'fai', 'wah', 'tat', 'man', 'tim', 'to'
    # (Cantonese given-name syllables).
}


def build_rows(detector: ChineseNameDetector) -> list[dict[str, object]]:
    """Enumerate table rows from the repo's own romanization tables."""
    data = detector._data
    norm = detector._normalizer.norm
    norm_light = detector._normalizer.norm_light

    rows: dict[str, dict[str, object]] = {}

    # Attested Cantonese/Hokkien/Korean surname romanizations: current behavior
    # (alias to the Mandarin target's mass) is kept, i.e. full target share.
    for cantonese_key, (mandarin, _han) in CANTONESE_SURNAMES.items():
        if " " in cantonese_key:
            continue  # compound surnames are handled by COMPOUND_VARIANTS
        assert norm_light(cantonese_key) == cantonese_key, cantonese_key
        rows[cantonese_key] = {
            "spelling": cantonese_key,
            "mandarin_target": mandarin,
            "surname_ppm_as_written": round(data.surname_frequencies.get(cantonese_key, 0.0), 4),
            "target_share": 1.0,
        }

    # Curated Wade-Giles/Taiwanese surname spellings: full target share.
    for spelling, target in WADE_GILES_SURNAME_TARGETS.items():
        assert spelling not in rows, spelling
        rows[spelling] = {
            "spelling": spelling,
            "mandarin_target": target,
            "surname_ppm_as_written": round(data.surname_frequencies.get(target, 0.0), 4),
            "target_share": 1.0,
        }

    # Everything else reachable by remap with zero as-written surname
    # attestation: penalty share.
    candidates = set(ROMANIZATION_EXCEPTIONS) | set(SYLLABLE_RULES)
    candidates |= {key for key in data.given_names if len(key) <= 7}
    for candidate in sorted(candidates):
        light = norm_light(candidate)
        if len(light) < 2 or not (light.isascii() and light.isalpha()) or light in rows:
            continue
        remapped = norm(candidate)
        if remapped == light:
            continue  # spelling keeps its own identity
        target_mass = data.surname_frequencies.get(remapped, 0.0)
        if target_mass <= 0:
            continue  # inherits no surname mass -> harmless
        if light in data.surnames:
            continue  # attested as-written -> keeps full mass
        rows[light] = {
            "spelling": light,
            "mandarin_target": remapped,
            "surname_ppm_as_written": round(target_mass * PENALTY_SHARE, 4),
            "target_share": round(PENALTY_SHARE, 10),
        }

    return sorted(rows.values(), key=lambda row: str(row["spelling"]))


def main() -> None:
    detector = ChineseNameDetector()
    rows = build_rows(detector)

    out_path = Path(__file__).resolve().parent.parent / "sinonym" / "data" / OUTPUT_NAME
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["spelling", "mandarin_target", "surname_ppm_as_written", "target_share"],
        )
        writer.writeheader()
        writer.writerows(rows)

    full_share = sum(1 for row in rows if row["target_share"] == 1.0)
    print(f"Wrote {len(rows)} spellings to {out_path}")
    print(f"  full share (attested romanizations): {full_share}")
    print(f"  penalty share ({PENALTY_SHARE:.4f}): {len(rows) - full_share}")


if __name__ == "__main__":
    main()
