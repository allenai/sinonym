#!/usr/bin/env python3
# ruff: noqa: PLR2004, SLF001
"""Generate the romanization-conditional surname table (surname_romanizations.csv).

Enumerates, from the repo's own romanization rule tables and name inventories,
every as-written spelling whose surname lookup inherits the frequency mass of a
different Mandarin syllable, and assigns each spelling the share of that target
mass it keeps when scored as a surname:

- ``CANTONESE_SURNAMES`` keys and the curated Wade-Giles/Taiwanese surname
  spellings below are attested surname romanizations -> full share (1.0),
  except the Korean-dominant spellings in
  ``KOREAN_DOMINANT_FULL_SHARE_EXCLUSIONS``, which are demoted to the penalty
  share (their real-but-minor Chinese usage keeps discounted as-written mass
  instead of the Mandarin target's full mass).
- Every other spelling reachable only through romanization remapping, with no
  as-written surname attestation, gets the penalty share e^-4 (~1.8%), the
  operating point measured on the 795-case eval (fixes "Leung Ka Fai" and
  "Kong Kung" with zero regressions; 2.0 log-units flips Leung Ka Fai, Kong
  Kung needs more than 3.0).

Column semantics are documented in ``sinonym/data/README.md``. Every column is
runtime-consumed: ``NameParsingService`` scores single-token surname candidates
from ``target_share``, and ``surname_ppm_as_written`` is loaded as the effective
surname frequency for penalty rows (``initialization.py`` seeds
``surname_frequencies`` from it and ``get_surname_freq_as_written`` returns
``max(direct, as_written_ppm)``). ``mandarin_target`` names the remap target and
also drives that seeding. ``surname_ppm_as_written`` is therefore load-bearing,
not documentation, so a corrupted committed value has runtime effect and this
generator must not regenerate it *from itself*.

Regeneration idempotence invariant
----------------------------------
Running this script against a healthy committed CSV must reproduce that CSV
byte-for-byte. The generator reads ``detector._data``, which initialization
builds partly *from* this same CSV, so any column computed from a value the CSV
itself supplied for that spelling would be a self-referential fixed point that
silently propagates a corrupted committed value into the regenerated output.
To avoid that for the mass columns:

- Full-share ``CANTONESE_SURNAMES`` rows read the frequency from the *Mandarin
  target* key (which comes from the ``familyname_orcid.csv`` base data), not from
  the as-written spelling's own frequency entry (which the CSV seeds).
- The remap-only loop excludes the table's own as-written aliases from the
  "attested -> keep full mass" skip, so a corrupted CSV cannot suppress penalty
  rows by injecting spurious spellings into ``surnames``.

Residual circularity (documented, not fixed): 13 Mandarin targets are themselves
full-share spellings in this table (they are ``CANTONESE_SURNAMES`` keys, e.g.
``li``, ``huang``, ``yu``). Corrupting *their own* row can still perturb their
``surname_frequencies`` entry (via ``initialization._build_surname_log_probabilities``,
which overwrites with the CSV's ``as_written_ppm`` when it exceeds the base mass),
which then flows to rows that target them (e.g. corrupting ``li`` reaches both
``li`` and ``lee``). Fully breaking this would require the generator to read the
raw ``familyname_orcid.csv`` base frequencies instead of the initialized data;
that is a larger change and is left as a known limitation.

Usage:
    uv run python scripts/generate_surname_romanizations.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import TYPE_CHECKING

from sinonym import ChineseNameDetector
from sinonym.chinese_names_data import CANTONESE_SURNAMES, ROMANIZATION_EXCEPTIONS, SYLLABLE_RULES

if TYPE_CHECKING:
    from collections.abc import Sequence

OUTPUT_NAME = "surname_romanizations.csv"

# Measured operating point: remap-only spellings keep e^-4 of the target mass.
PENALTY_SHARE = math.exp(-4.0)

# Korean-dominant CANTONESE_SURNAMES keys demoted from full share (aliasing
# the Mandarin target's full mass) to the penalty share (~1.8% of the target,
# scoring as-written like a genuinely rare Chinese surname). Each decision was
# fixed by measurement: the "delta" quoted is the attributed net change in
# decisive pp-vys string-level correctness when ALL 18 Korean-overlap
# full-share spellings were trimmed at once and the fixture evidence refreshed
# (near-per-spelling by case-insensitive token match, see the commit message).
# Full-share rows granting real Chinese surname mass (Cantonese/HK/SG/MY/TW
# usage) stay full-share; a blanket trim would be very wrong for them.
#
# TRIMMED (5) - Korean-dominant, trimming is neutral-or-positive on fixtures:
#   jung   delta +2  Korean 정; jung->zheng. Fixes "Jung Im Na"/"Jung Mi Cha".
#   im     delta +1  Korean 임; im->lin (Chinese 林 is romanized lin/lim, not
#                     im). Fixes "Jung Im Na"; kept in the trim set over jang
#                     because it carries this measured fix.
#   pak    delta +1  Korean 박; pak->bai. Fixes "Pak Siau Wei".
#   moon   delta  0  Korean 문; moon->wen. No Chinese as-written usage in the
#                     fixtures/eval to lose; linguistically Korean-dominant.
#   kyeong delta  0  Korean 경; kyeong->jing. Absent from fixtures/eval;
#                     linguistically Korean-dominant.
#
# KEPT full-share (13) - measured cost, Chinese dominance, or eval-baseline:
#   lee    delta -2  HK 李, top Chinese surname; trimming breaks Lee Yan Ying/Lee Si.
#   choi   delta -2  HK 蔡; trimming breaks Changsun Choi/Mi Ran Choi.
#   han    delta  0  Chinese 韩 common; net-neutral (Han Liang +1, Dai Hoon Han -1).
#   ho     delta +1  HK/Cantonese 何, top HK surname; the sole +1 is one
#                     Cantonese-context row (Ho-fung Hung), outweighed by ho's
#                     dominant legit Chinese usage - exactly the resolver's target.
#   chang  delta  0  WG/TW 张; only cosmetic either-string flips, no accuracy change.
#   lim    delta  0  SG/MY 林; Chinese-dominant, no decisive change.
#   yu     delta  0  yu->yu is the Mandarin surname 于/俞/余 itself; trimming would
#                     penalize a real full-mass Chinese surname. Keep firmly.
#   jang   delta  0  Korean 장 (would otherwise be trimmed), but the {im,jang,jung}
#                     trio flips the borderline eval case "Li Zheng" via a global
#                     surname-rank shift; eval must hold at the 16-name baseline,
#                     so one of {im,jang} is retained and im carries the measured
#                     fix while jang has zero attributed fixture delta.
#   koo    delta  0  eval-baseline: trimming rejects "Koo Ming"; also Chinese-attested
#                     (Zhang Koo). Korean 구/Chinese 顾/古 - keep.
#   shin   delta  0  bicultural (Korean 신 / Chinese 沈/辛); no Chinese-attested
#                     fixture/eval usage but default-keep (conservative); trimming
#                     only diverges the either row So Ra Shin.
#   son    delta  0  bicultural (Korean 손 / Chinese 孙); no fixture/eval presence, default-keep.
#   soo    delta  0  bicultural (Korean 수 / Chinese 苏); only a cosmetic identical-string
#                     flip (Soo Han), default-keep.
#   suh    delta  0  bicultural (Korean 서 / Chinese 徐); only a freq change on S. Suh,
#                     no outcome change, default-keep.
KOREAN_DOMINANT_FULL_SHARE_EXCLUSIONS = frozenset(
    {
        "im",
        "jung",
        "kyeong",
        "moon",
        "pak",
    },
)

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

    # Attested Cantonese/Hokkien surname romanizations: current behavior
    # (alias to the Mandarin target's mass) is kept, i.e. full target share.
    # Korean-dominant exclusions are demoted to the penalty share of the
    # target's mass (computed from the target so regeneration is idempotent).
    for cantonese_key, (mandarin, _han) in CANTONESE_SURNAMES.items():
        if " " in cantonese_key:
            continue  # compound surnames are handled by COMPOUND_VARIANTS
        assert norm_light(cantonese_key) == cantonese_key, cantonese_key
        if cantonese_key in KOREAN_DOMINANT_FULL_SHARE_EXCLUSIONS:
            target_mass = data.surname_frequencies.get(mandarin, 0.0)
            rows[cantonese_key] = {
                "spelling": cantonese_key,
                "mandarin_target": mandarin,
                "surname_ppm_as_written": round(target_mass * PENALTY_SHARE, 4),
                "target_share": round(PENALTY_SHARE, 10),
            }
            continue
        # Read the frequency from the Mandarin *target* key, not the as-written
        # Cantonese key. In a healthy state initialization sets
        # frequencies[cantonese_key] == frequencies[mandarin] for full-share
        # rows, so this is value-identical; but the target key does not route
        # through surname_romanizations.csv, so a corrupted committed value for
        # THIS spelling's own row no longer feeds back into its regenerated
        # mass (see the idempotence invariant in the module docstring).
        rows[cantonese_key] = {
            "spelling": cantonese_key,
            "mandarin_target": mandarin,
            "surname_ppm_as_written": round(data.surname_frequencies.get(mandarin, 0.0), 4),
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
        if light in data.surnames and light not in data.surname_as_written_aliases:
            # Attested as-written with its own (familyname-derived) surname mass
            # -> keeps full mass. Exclude spellings the table itself injected into
            # `surnames` (the full-share/penalty as-written aliases): those are
            # this generator's own output, so skipping on them would let a
            # corrupted committed CSV suppress penalty rows it should still emit.
            # In a healthy state every alias is already in `rows` (produced by the
            # CANTONESE_SURNAMES / WADE_GILES loops above) and skipped earlier, so
            # this guard is value-identical while breaking that circularity.
            continue
        rows[light] = {
            "spelling": light,
            "mandarin_target": remapped,
            "surname_ppm_as_written": round(target_mass * PENALTY_SHARE, 4),
            "target_share": round(PENALTY_SHARE, 10),
        }

    return sorted(rows.values(), key=lambda row: str(row["spelling"]))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    _parse_args(argv)
    detector = ChineseNameDetector()
    if detector._data is None:
        message = (
            "detector initialization failed (likely a missing/corrupt data CSV); "
            "restore a working baseline via `git checkout -- sinonym/data/<file>` before regenerating"
        )
        raise RuntimeError(message)
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
