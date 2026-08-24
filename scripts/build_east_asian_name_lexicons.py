# ruff: noqa: INP001
"""Build deterministic East Asian name-order lexicons from pinned open data."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from sinonym.text_processing.text_normalizer import exact_name_surface_key

COUNTRY_COMMIT = "eb62e13d4d62dd96cdfae79d293a02066352205f"
JAPANESE_COMMIT = "c5220278652e7bae05b06cfaf527f1b09a100de6"
USER_AGENT = "sinonym-east-asian-lexicon-builder/1.0"
MAX_ATTEMPTS = 4
RETRYABLE_HTTP_CODES = frozenset({429, 502, 503, 504})
MIN_GIVEN_FIELDS = 2
ASSET_SCHEMA_VERSION = 3
TOP_VIETNAMESE_SURNAME_RANK = 4
REVIEWED_SURFACE_TOKEN_COUNT = 2
GZIP_OS_BYTE_OFFSET = 9
GZIP_UNKNOWN_OS = 255
REVIEWED_POSSIBLE_SURNAMES_PATH = Path(__file__).with_name("data") / "reviewed_japanese_possible_surnames.csv"
REVIEWED_POSSIBLE_SURNAME_FIELDS = (
    "surname_key",
    "evidence_id",
    "example_surface",
    "evidence_url",
    "given_first_exact_surface",
)


@dataclass(frozen=True)
class Source:
    """One hash-pinned upstream CSV."""

    name: str
    url: str
    sha256: str
    license: str
    repository: str


SOURCES = (
    Source(
        name="country_surnames",
        url=(
            "https://raw.githubusercontent.com/sigpwned/"
            f"popular-names-by-country-dataset/{COUNTRY_COMMIT}/common-surnames-by-country.csv"
        ),
        sha256="32cb28bea558a9d353feeef097da03c6488a4e7ba9398e5983cee2f0b9caa91c",
        license="CC0-1.0",
        repository="https://github.com/sigpwned/popular-names-by-country-dataset",
    ),
    Source(
        name="japanese_surnames",
        url=(
            "https://raw.githubusercontent.com/shuheilocale/"
            f"japanese-personal-name-dataset/{JAPANESE_COMMIT}/"
            "japanese_personal_name_dataset/dataset/last_name_org.csv"
        ),
        sha256="02706bd06932d6bde121e6fbaf9bc2a88c43e6f015128215eecb213223523d30",
        license="MIT",
        repository="https://github.com/shuheilocale/japanese-personal-name-dataset",
    ),
    Source(
        name="japanese_male_given_names",
        url=(
            "https://raw.githubusercontent.com/shuheilocale/"
            f"japanese-personal-name-dataset/{JAPANESE_COMMIT}/"
            "japanese_personal_name_dataset/dataset/first_name_man_org.csv"
        ),
        sha256="c14a67047f6101fa3234e87c3f643f68ec1807f3fdb545855e57ab9bd5cd5a3b",
        license="MIT",
        repository="https://github.com/shuheilocale/japanese-personal-name-dataset",
    ),
    Source(
        name="japanese_female_given_names",
        url=(
            "https://raw.githubusercontent.com/shuheilocale/"
            f"japanese-personal-name-dataset/{JAPANESE_COMMIT}/"
            "japanese_personal_name_dataset/dataset/first_name_woman_org.csv"
        ),
        sha256="de8ce6f8f430e14e0036ca209656a1ec06e9bcd83e9defffe1b0654dc636f9a9",
        license="MIT",
        repository="https://github.com/shuheilocale/japanese-personal-name-dataset",
    ),
)


def parse_args() -> argparse.Namespace:
    """Parse explicit output paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--roman-output", required=True, type=Path)
    parser.add_argument("--native-output", required=True, type=Path)
    return parser.parse_args()


def fetch(source: Source) -> bytes:
    """Fetch and verify one pinned source with bounded retries."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = urllib.request.Request(source.url, headers={"User-Agent": USER_AGENT})  # noqa: S310
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                payload = response.read()
        except urllib.error.HTTPError as error:
            if error.code not in RETRYABLE_HTTP_CODES or attempt == MAX_ATTEMPTS:
                raise
            delay_seconds = 5 * attempt
            print(
                f"retry source={source.name} attempt={attempt}/{MAX_ATTEMPTS} http={error.code} delay={delay_seconds}s",
                file=sys.stderr,
            )
            time.sleep(delay_seconds)
            continue
        digest = hashlib.sha256(payload).hexdigest()
        if digest != source.sha256:
            message = f"source hash mismatch for {source.name}: expected={source.sha256} actual={digest}"
            raise ValueError(message)
        return payload
    message = f"bounded attempts exhausted for {source.name}"
    raise AssertionError(message)


def fold(value: str) -> str:
    """Return a lowercase accent-insensitive lookup key."""
    translated = value.translate(str.maketrans({"\u0110": "D", "\u0111": "d"}))
    return "".join(
        character for character in unicodedata.normalize("NFD", translated).casefold() if not unicodedata.combining(character)
    )


def japanese_roman_keys(value: str) -> set[str]:
    """Return exact and common long-vowel-neutral Japanese keys."""
    exact = fold(value)
    return {exact, exact.replace("ou", "o").replace("oo", "o").replace("uu", "u")}


def slash_variants(value: str) -> list[str]:
    """Expand slash-separated source variants."""
    return [part.strip() for part in value.split("/") if part.strip()]


def is_hangul(value: str) -> bool:
    """Return whether a source form is entirely modern Hangul."""
    return bool(value) and all("\uac00" <= character <= "\ud7a3" for character in value)


def source_metadata() -> list[dict[str, str]]:
    """Return serializable provenance for every input."""
    return [
        {
            "name": source.name,
            "url": source.url,
            "sha256": source.sha256,
            "license": source.license,
            "repository": source.repository,
        }
        for source in SOURCES
    ]


def reviewed_possible_surnames() -> tuple[list[str], list[str], dict[str, str]]:
    """Load reviewed ambiguity-only surnames and optional exact repairs."""
    payload = REVIEWED_POSSIBLE_SURNAMES_PATH.read_bytes()
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    if tuple(reader.fieldnames or ()) != REVIEWED_POSSIBLE_SURNAME_FIELDS:
        message = f"invalid reviewed possible surname columns in {REVIEWED_POSSIBLE_SURNAMES_PATH}"
        raise ValueError(message)
    rows = list(reader)
    keys = [row["surname_key"].strip() for row in rows]
    evidence_surfaces = [fold(row["example_surface"].strip()) for row in rows]
    surfaces = [
        exact_name_surface_key(row["given_first_exact_surface"]) for row in rows if row["given_first_exact_surface"].strip()
    ]
    if not keys or any(key != fold(key) or not key.isalpha() for key in keys) or len(keys) != len(set(keys)):
        message = f"invalid reviewed possible surname keys in {REVIEWED_POSSIBLE_SURNAMES_PATH}"
        raise ValueError(message)
    if any(
        len(surface.split()) != REVIEWED_SURFACE_TOKEN_COUNT or key not in japanese_roman_keys(surface.split()[-1])
        for key, surface in zip(keys, evidence_surfaces, strict=True)
    ):
        message = f"invalid reviewed surname evidence surfaces in {REVIEWED_POSSIBLE_SURNAMES_PATH}"
        raise ValueError(message)
    exact_rows = [row for row in rows if row["given_first_exact_surface"].strip()]
    if any(
        len(surface.split()) != REVIEWED_SURFACE_TOKEN_COUNT
        or row["surname_key"].strip() not in japanese_roman_keys(surface.split()[-1])
        for row, surface in zip(exact_rows, surfaces, strict=True)
    ) or len(surfaces) != len(set(surfaces)):
        message = f"invalid reviewed exact surfaces in {REVIEWED_POSSIBLE_SURNAMES_PATH}"
        raise ValueError(message)
    return (
        sorted(keys),
        sorted(surfaces),
        {
            "name": "reviewed_possible_japanese_surnames",
            "url": "scripts/data/reviewed_japanese_possible_surnames.csv",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "license": "MIT",
            "repository": "https://github.com/allenai/sinonym",
        },
    )


def encode(payload: dict[str, object]) -> bytes:
    """Return deterministic gzip-compressed JSON."""
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    compressed = bytearray(gzip.compress(serialized, compresslevel=9, mtime=0))
    # CPython 3.11-3.12 expose zlib's platform-specific OS byte when mtime=0,
    # while 3.10 and 3.13+ write 255. Canonicalize it for cross-version builds.
    compressed[GZIP_OS_BYTE_OFFSET] = GZIP_UNKNOWN_OS
    return bytes(compressed)


def main() -> None:
    """Build the two runtime assets and print their counts and hashes."""
    args = parse_args()
    fetched = {source.name: fetch(source) for source in SOURCES}
    country_rows = list(
        csv.DictReader(io.StringIO(fetched["country_surnames"].decode("utf-8-sig"))),
    )
    japanese_surname_rows = list(
        csv.reader(io.StringIO(fetched["japanese_surnames"].decode("utf-8-sig"))),
    )
    japanese_given_rows = [
        row
        for source_name in (
            "japanese_male_given_names",
            "japanese_female_given_names",
        )
        for row in csv.reader(io.StringIO(fetched[source_name].decode("utf-8-sig")))
        if len(row) >= MIN_GIVEN_FIELDS and row[0].strip() and row[1].strip()
    ]

    japanese_surnames_roman = sorted(
        {
            key
            for row in japanese_surname_rows
            if len(row) == 4 and row[3].strip()  # noqa: PLR2004
            for key in japanese_roman_keys(row[3].strip())
        },
    )
    japanese_surnames_native = sorted(
        {
            row[0].strip()
            for row in japanese_surname_rows
            if len(row) == 4 and row[0].strip()  # noqa: PLR2004
        },
    )
    japanese_given_roman = sorted(
        {key for row in japanese_given_rows for key in japanese_roman_keys(row[1].strip())},
    )
    japanese_given_native = sorted(
        {native.strip() for row in japanese_given_rows for native in row[2:] if native.strip()},
    )
    japanese_country_surnames_roman = sorted(
        {
            key
            for row in country_rows
            if row["Country"] == "JP"
            for romanized in slash_variants(row["Romanized Name"])
            for key in japanese_roman_keys(romanized)
        },
    )
    reviewed_surnames, reviewed_surfaces, reviewed_provenance = reviewed_possible_surnames()
    japanese_possible_surnames_roman = sorted(
        set(japanese_surnames_roman) | set(japanese_country_surnames_roman) | set(reviewed_surnames),
    )
    korean_surnames_roman = sorted(
        {
            fold(romanized)
            for row in country_rows
            if row["Country"] == "KR" and is_hangul(row["Localized Name"])
            for romanized in slash_variants(row["Romanized Name"])
        },
    )
    vietnamese_surnames_roman = sorted(
        {
            fold(romanized)
            for row in country_rows
            if row["Country"] == "VN"
            for romanized in slash_variants(row["Romanized Name"])
        },
    )
    vietnamese_top4_surnames_roman = sorted(
        {
            fold(romanized)
            for row in country_rows
            if row["Country"] == "VN" and int(row["Rank"]) <= TOP_VIETNAMESE_SURNAME_RANK
            for romanized in slash_variants(row["Romanized Name"])
        },
    )
    if len(vietnamese_top4_surnames_roman) != TOP_VIETNAMESE_SURNAME_RANK:
        message = (
            "pinned Vietnamese surname source no longer yields exactly "
            f"{TOP_VIETNAMESE_SURNAME_RANK} top-ranked forms: {vietnamese_top4_surnames_roman!r}"
        )
        raise ValueError(message)

    provenance = source_metadata()
    roman_payload: dict[str, object] = {
        "schema_version": ASSET_SCHEMA_VERSION,
        "sources": [*provenance, reviewed_provenance],
        "japanese_surnames": japanese_surnames_roman,
        "japanese_possible_surnames": japanese_possible_surnames_roman,
        "japanese_given_first_exact_surfaces": reviewed_surfaces,
        "japanese_given_names": japanese_given_roman,
        "korean_surnames": korean_surnames_roman,
        "vietnamese_surnames": vietnamese_surnames_roman,
        "vietnamese_top4_surnames": vietnamese_top4_surnames_roman,
    }
    native_payload: dict[str, object] = {
        "schema_version": ASSET_SCHEMA_VERSION,
        "sources": provenance,
        "japanese_surnames": japanese_surnames_native,
        "japanese_given_names": japanese_given_native,
    }
    outputs = (
        (args.roman_output, roman_payload),
        (args.native_output, native_payload),
    )
    report: dict[str, object] = {"counts": {}}
    for path, payload in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = encode(payload)
        path.write_bytes(encoded)
        report[path.name] = {
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    report["counts"] = {
        "japanese_surnames_roman": len(japanese_surnames_roman),
        "japanese_country_surnames_roman": len(japanese_country_surnames_roman),
        "japanese_possible_surnames_roman": len(japanese_possible_surnames_roman),
        "japanese_given_first_exact_surfaces": len(reviewed_surfaces),
        "japanese_given_roman": len(japanese_given_roman),
        "japanese_surnames_native": len(japanese_surnames_native),
        "japanese_given_native": len(japanese_given_native),
        "korean_surnames_roman": len(korean_surnames_roman),
        "vietnamese_surnames_roman": len(vietnamese_surnames_roman),
        "vietnamese_top4_surnames_roman": len(vietnamese_top4_surnames_roman),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
