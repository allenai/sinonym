"""Regression coverage for confirmed normalization review findings."""

from typing import cast

import pytest

from sinonym import ChineseNameDetector
from sinonym.name_punctuation import fold_spaced_transliteration_apostrophes
from sinonym.services.normalization import LazyNormalizationMap, NormalizationService
from sinonym.services.person_name_normalization import PersonNameNormalizationService, PersonNameOutcome


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("Sa 'di Ahmed", "Sa'di Ahmed"),
        ("Ts 'ai Ing-wen", "Ts'ai Ing-wen"),
        ("Ma 'ayan Hillel", "Ma'ayan Hillel"),
        ("Cui 'e Zheng", "Cui'e Zheng"),
        ("P 'eng Wang", "P'eng Wang"),
        ("SA 'di Ahmed", "SA'di Ahmed"),
    ],
)
def test_spaced_transliteration_apostrophes_join_the_preceding_token(raw_name: str, expected: str) -> None:
    """Multi-letter transliteration prefixes retain their apostrophe boundary."""
    result = PersonNameNormalizationService().normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == expected


@pytest.mark.parametrize(
    "raw_name",
    [
        "Zhang 'Wei",
        "john 'smith",
        "Sa 'Di Ahmed",
        "John 'Jack' Smith",
        "Claude Bigar' PhD",
        "Gerard't Hooft",
        "can't stop",
    ],
)
def test_unreviewed_apostrophe_spacing_is_left_unchanged(raw_name: str) -> None:
    """The transliteration repair is an exact allowlist, not a casing heuristic."""
    assert fold_spaced_transliteration_apostrophes(raw_name) == raw_name


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("Zhang 'Wei", "Wei Zhang"),
        ("Li 'Na", "Na Li"),
        ("Chen 'Long Fei", "Long-Fei Chen"),
        ("Zhang \u2019Wei", "Wei Zhang"),
    ],
)
def test_unreviewed_spaced_apostrophe_preserves_chinese_token_boundaries(
    detector: ChineseNameDetector,
    raw_name: str,
    expected: str,
) -> None:
    """Stray apostrophes cannot fuse otherwise separate Chinese name tokens."""
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == expected


@pytest.mark.parametrize(
    "raw_name",
    [
        "Gerard 't Hooft",
        "Gerard' t Hooft",
        "Gerard ' t Hooft",
    ],
)
def test_dutch_t_particle_has_one_canonical_spacing(raw_name: str) -> None:
    """Source spacing cannot fuse the Dutch particle to the given name."""
    result = PersonNameNormalizationService().normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == "Gerard 't Hooft"
    assert result.canonical_name.normalized.surname_tokens == ("'t", "Hooft")


@pytest.mark.parametrize(
    ("raw_name", "expected_text", "expected_suffix"),
    [
        ("Claude Bigar' PhD", "Claude Bigar'", ""),
        ("Claude Bigar' Jr.", "Claude Bigar' Jr.", "Jr."),
    ],
)
def test_terminal_surname_apostrophe_does_not_absorb_credentials_or_suffixes(
    raw_name: str,
    expected_text: str,
    expected_suffix: str,
) -> None:
    """A terminal semantic apostrophe stays attached only to the surname."""
    result = PersonNameNormalizationService().normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == expected_text
    assert result.canonical_name.normalized.surname == "Bigar'"
    assert result.canonical_name.normalized.suffix == expected_suffix


def test_spaced_chinese_apostrophe_uses_the_shared_public_normalization(
    detector: ChineseNameDetector,
) -> None:
    """The Chinese override must not turn the repaired boundary into a hyphen."""
    scalar = detector.normalize_name("Cui 'e Zheng")
    canonical = detector.normalize_person_name("Cui 'e Zheng")

    assert scalar.success
    assert scalar.result == "Cui'e Zheng"
    assert canonical is not None
    assert canonical.text == "Cui'e Zheng"
    quoted = detector.normalize_person_name("John 'Jack' Smith")
    particle = detector.normalize_person_name("Gerard 't Hooft")
    assert quoted is not None
    assert particle is not None
    assert quoted.text == "John Jack Smith"
    assert particle.text == "Gerard 't Hooft"


@pytest.mark.parametrize(
    ("raw_name", "expected_text", "expected_surname", "expected_suffix"),
    [
        ("Junior, Maria", "Maria Junior", "Junior", ""),
        ("Filho, Jose", "Jose Filho", "Filho", ""),
        ("Neto, Antonio", "Antonio Neto", "Neto", ""),
        ("Smith MD, John", "John Smith", "Smith", ""),
        ("Smith Junior, John", "John Smith Jr.", "Smith", "Jr."),
        ("Smith, John Junior", "John Smith Jr.", "Smith", "Jr."),
    ],
)
def test_comma_names_require_a_surviving_family_token_before_demoting_surname_like_suffixes(
    raw_name: str,
    expected_text: str,
    expected_surname: str,
    expected_suffix: str,
) -> None:
    """Family-only Junior/Filho/Neto remain surnames without weakening real suffixes."""
    result = PersonNameNormalizationService().normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == expected_text
    assert result.canonical_name.normalized.surname == expected_surname
    assert result.canonical_name.normalized.suffix == expected_suffix


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("Annelies Hommens - van de Steeg", "Annelies Hommens-van de Steeg"),
        ("Suzan Cochius - den Otter", "Suzan Cochius-den Otter"),
        ("Dr. Mohammad Moshfaq - ur Rahman", "Mohammad Moshfaq-ur Rahman"),
        ("Annemarieke Spitzen - van der Sluijs", "Annemarieke Spitzen-van der Sluijs"),
    ],
)
def test_spaced_hyphen_before_particle_surname_is_not_a_multi_name_separator(raw_name: str, expected: str) -> None:
    """Reviewed particle surnames survive the conservative spaced-hyphen gate."""
    result = PersonNameNormalizationService().normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == expected


@pytest.mark.parametrize(
    ("raw_name", "expected_reason"),
    [
        ("John Smith - Mary Jones", "multiple-name separator"),
        ("John Smith - Mary van der Berg", "multiple-name separator"),
        ("John van der Berg - Mary Jones", "multiple-name separator"),
        ("John A. Smith - Mary Jones", "multiple-name separator"),
        ("Smith John, Mary van der Berg", "comma separates two complete names"),
    ],
)
def test_separator_between_complete_names_remains_non_person(raw_name: str, expected_reason: str) -> None:
    """Particles and initials within complete sides cannot hide a second person."""
    result = PersonNameNormalizationService().normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.NON_PERSON
    assert result.reason == expected_reason


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("van der Berg, John Adam", "John Adam van der Berg"),
        ("Smith Jones, John A.", "John A. Smith Jones"),
    ],
)
def test_comma_name_particle_and_initial_controls_remain_person(raw_name: str, expected: str) -> None:
    """The comma-specific particle and initial guards still protect name forms."""
    result = PersonNameNormalizationService().normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == expected


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("Zhang et al.", "Zhang"),
        ("Et al. Wang, Wei", "Wei Wang"),
    ],
)
def test_et_al_cleanup_accepts_one_name_token_plus_external_context(raw_name: str, expected: str) -> None:
    """Exact edge citation markers are removed when one complete name remains."""
    result = PersonNameNormalizationService().normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert result.canonical_name.text == expected


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("lee young", "Young Lee"),
        ("LEE YOUNG", "Young Lee"),
        ("choi woong", "Woong Choi"),
        ("CHOI WOONG", "Woong Choi"),
    ],
)
def test_atomic_korean_tokens_are_title_cased(detector: ChineseNameDetector, raw_name: str, expected: str) -> None:
    """Atomic Korean bypasses splitting, not output capitalization."""
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == expected


@pytest.mark.parametrize(
    ("compact", "spaced"),
    [
        ("李Wěi", "李 Wěi"),
        ("李Ān", "李 Ān"),
        ("Zhāng伟", "Zhāng 伟"),
        ("Ān李", "Ān 李"),
    ],
)
def test_compact_mixed_tokens_preserve_supported_accented_roman_letters(
    detector: ChineseNameDetector,
    compact: str,
    spaced: str,
) -> None:
    """Compact and spaced Han/Roman surfaces normalize identically."""
    compact_result = detector.normalize_name(compact)
    spaced_result = detector.normalize_name(spaced)

    assert compact_result.success
    assert spaced_result.success
    assert compact_result.result == spaced_result.result


@pytest.mark.parametrize("raw_name", ["'Zhang Wei", "Zhang Wei''"])
def test_chinese_output_strips_stray_apostrophes_at_token_boundaries(
    detector: ChineseNameDetector,
    raw_name: str,
) -> None:
    """Affirmative Chinese output cannot leak unmatched boundary apostrophes."""
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == "Wei Zhang"


@pytest.mark.parametrize("raw_name", ["P'eng Wang", "Cui'e Zheng"])
def test_chinese_output_preserves_internal_apostrophes(detector: ChineseNameDetector, raw_name: str) -> None:
    """The formatter boundary cleanup retains internal transliteration marks."""
    result = detector.normalize_name(raw_name)

    assert result.success
    assert result.result == raw_name


@pytest.mark.parametrize("raw_name", ["P'eng Wang", "Cui'e Zheng", "Claude Bigar'"])
def test_person_normalization_preserves_semantic_apostrophes(raw_name: str) -> None:
    """The Chinese-output cleanup does not alter generic semantic apostrophes."""
    result = PersonNameNormalizationService().normalize_text(raw_name)

    assert result.outcome is PersonNameOutcome.PERSON
    assert result.canonical_name is not None
    assert "'" in result.canonical_name.text


class _StubTextNormalizer:
    """Minimal token normalizer for the lazy-map contract test."""

    @staticmethod
    def normalize_token(token: str) -> str:
        """Return a visibly normalized test value."""
        return token.casefold()


class _StubNormalizationService:
    """Expose the collaborator consumed by ``LazyNormalizationMap``."""

    _text_normalizer = _StubTextNormalizer()


def test_lazy_normalization_map_get_honors_membership_and_default() -> None:
    """``get`` follows mapping defaults while indexed lookup remains lazy."""
    normalizer = cast("NormalizationService", _StubNormalizationService())
    values = LazyNormalizationMap(("KNOWN",), normalizer)

    assert values.get("KNOWN") == "known"
    assert values.get("MISSING") is None
    assert values.get("MISSING", "fallback") == "fallback"
    assert values["MISSING"] == "missing"
