"""
Chinese Name Detection and Normalization Module

This module provides sophisticated detection and normalization of Chinese names from various
romanization systems, with robust filtering to prevent false positives from Western, Korean,
Vietnamese, and Japanese names.

## Overview

The core functionality is provided by the `ChineseNameDetector` class, which uses a multi-stage
pipeline to process names:

1. **Input Preprocessing**: Handles mixed scripts, normalizes romanization variants
2. **Ethnicity Classification**: Filters non-Chinese names using linguistic patterns
3. **Probabilistic Parsing**: Identifies surname/given name boundaries using frequency data
4. **Compound Name Splitting**: Splits fused given names using tiered confidence system
5. **Output Formatting**: Produces standardized "Given-Name Surname" format

## Architecture

### Clean Service Separation
- **NormalizationService**: Pure centralized normalization with lazy computation
- **PinyinCacheService**: Isolated cache management with persistent storage
- **DataInitializationService**: Immutable data structure initialization
- **ChineseNameDetector**: Main detection engine with dependency injection

### Scala-Compatible Design
- **Immutable Data Structures**: All core data is frozen/immutable for thread safety
- **Functional Error Handling**: ParseResult with Either-like success/failure semantics
- **Pure Functions**: Side-effect free normalization suitable for Scala interop
- **Dependency Injection**: Clean separation of concerns, no circular dependencies

### Performance Optimizations
- **Lazy Normalization**: On-demand token processing reduces memory usage
- **Early Exit Patterns**: Non-Chinese names detected quickly without full processing
- **Persistent Caching**: Han→Pinyin mappings cached to disk for fast startup
- **Single-Pass Processing**: Minimized regex operations and string transformations

## Key Features

### Comprehensive Romanization Support
- **Pinyin**: Standard mainland Chinese romanization
- **Wade-Giles**: Traditional romanization system with aspirated consonants
- **Cantonese**: Hong Kong and southern Chinese romanizations
- **Mixed Scripts**: Handles names with both Han characters and Roman letters

### Advanced Name Splitting
The module uses a sophisticated **tiered confidence system** for splitting compound given names:

- **Gold Standard**: Both parts are high-confidence Chinese syllables (anchors)
- **Silver Standard**: One part is high-confidence, one is plausible
- **Bronze Standard**: Both parts are plausible with cultural validation

This prevents incorrect splitting of Western names (e.g., "Julian" → "Jul", "ian") while
correctly handling Chinese compounds (e.g., "Weiming" → "Wei", "Ming").

### Robust False Positive Prevention
- **Forbidden Phonetic Patterns**: Blocks Western consonant clusters (th, dr, br, gl, etc.)
- **Korean Name Detection**: Identifies Korean surnames and given name patterns
- **Vietnamese Name Detection**: Recognizes Vietnamese naming conventions
- **Cultural Validation**: Applies frequency analysis and phonetic rules

### Data-Driven Approach
- **Surname Database**: ~1400 Chinese surnames with frequency data
- **Given Name Database**: ~3000 Chinese given name syllables with probabilities
- **Compound Syllables**: ~400 valid Chinese syllable components for splitting
- **Ethnicity Markers**: Curated lists of non-Chinese name patterns

## Usage Examples

```python
# from s2and.chinese_names import ChineseNameDetector  # Original import - now internal

# Basic usage
detector = ChineseNameDetector()
result = detector.normalize_name("Zhang Wei")
# Returns: ParseResult(success=True, result="Wei Zhang")

# Compound given names
result = detector.normalize_name("Li Weiming")
# Returns: ParseResult(success=True, result="Wei-Ming Li")

# Mixed scripts
result = detector.normalize_name("张Wei Ming")
# Returns: ParseResult(success=True, result="Wei-Ming Zhang")

# Non-Chinese names (correctly rejected)
result = detector.normalize_name("John Smith")
# Returns: ParseResult(success=False, error_message="surname not recognised")

result = detector.normalize_name("Kim Min-jun")
# Returns: ParseResult(success=False, error_message="appears to be Korean name")

# Access result data
if result.success:
    print(f"Formatted name: {result.result}")
else:
    print(f"Error: {result.error_message}")

# Advanced usage - access normalization service directly
normalized_token = detector._normalizer.norm("wei")  # Returns: "wei"
normalized_token = detector._normalizer.norm("ts'ai")  # Returns: "cai" (Wade-Giles conversion)

# Get cache information
cache_info = detector.get_cache_info()
print(f"Cache size: {cache_info.cache_size} characters")
```

## Architecture

### Core Classes

- **ChineseNameDetector**: Main detection engine with caching and data management
- **PinyinCacheService**: Fast Han character to Pinyin conversion with disk caching
- **DataInitializationService**: Loads and processes surname/given name databases
- **ChineseNameConfig**: Configuration and regex patterns

### Data Sources

- **familyname_orcid.csv**: Chinese surnames with frequency data
- **givenname_orcid.csv**: Chinese given names with usage statistics
- **han_pinyin_cache.pkl**: Precomputed Han character to Pinyin mappings

### Processing Pipeline

1. **Preprocessing**: Clean input, normalize punctuation, handle compound surnames
2. **Tokenization**: Split into tokens, convert Han characters to Pinyin
3. **Ethnicity Check**: Score for Korean/Vietnamese/Japanese patterns vs Chinese evidence
4. **Parse Generation**: Create all valid (surname, given_name) combinations
5. **Scoring**: Rank parses using frequency data and cultural patterns
6. **Formatting**: Split compound names, capitalize, format as "Given-Name Surname"

## Error Handling

The module provides detailed error messages for debugging:
- `"surname not recognised"`: No valid Chinese surname found
- `"appears to be Korean name"`: Korean linguistic patterns detected
- `"appears to be Vietnamese name"`: Vietnamese naming conventions identified
- `"given name tokens are not plausibly Chinese"`: Given name validation failed

## Performance

- **Production Ready**: ~0.16ms average per name (comprehensive benchmark validated)
- **Cold start**: ~100ms (initial data loading with persistent cache)
- **Warm processing**: Sub-millisecond for most names with early exit optimization
- **Memory efficiency**: Lazy normalization reduces peak usage by ~60%
- **Cache optimization**: Persistent disk cache for Han→Pinyin mappings
- **Scalability**: Thread-safe design suitable for high-throughput processing

## API

The main class is `ChineseNameDetector`:
- `ChineseNameDetector()`: Main detector class
- `detector.normalize_name(name) -> ParseResult`: Returns structured result with success/error
- `ParseResult.success`: Boolean indicating if name was recognized as Chinese
- `ParseResult.result`: Formatted name if successful
- `ParseResult.error_message`: Error description if failed

## Thread Safety

The module is thread-safe after initialization. The caching layer uses immutable
data structures and the detector can be safely used from multiple threads.
"""

import logging
import math
import platform
import string
import threading
from dataclasses import replace
from typing import Literal

from sinonym.chinese_names_data import (
    COMPOUND_VARIANTS,
    HAN_SURNAME_POSITION_READINGS,
)
from sinonym.coretypes import (
    BatchFormatPattern,
    BatchParseResult,
    CanonicalName,
    NameComponents,
    NameFormat,
)
from sinonym.coretypes.results import ParsedName
from sinonym.coretypes.routing_resolution import (
    ApplyAssignment,
    EastAsianEvidenceReason,
    EvidenceFailure,
    HardScalarConstraint,
    HardScalarMaterializationFailure,
    PreserveBaseline,
    ResolutionReason,
    east_asian_evidence_resolution_reason,
)
from sinonym.services import (
    BatchAnalysisDependencies,
    BatchAnalysisOptions,
    BatchAnalysisService,
    CacheInfo,
    ChineseNameConfig,
    DataInitializationService,
    EthnicityClassificationService,
    NameDataStructures,
    NameFormattingService,
    NameParsingService,
    NonPersonInputDetectionService,
    NormalizationService,
    NormalizedInput,
    ParseResult,
    PinyinCacheService,
    ServiceContext,
    SurnameResolver,
)
from sinonym.services.batch_analysis import RelatedBatchParseResult
from sinonym.services.east_asian_name_order import (
    EastAsianNameOrderDecision,
    EastAsianNameOrderPreservation,
    EastAsianNameOrderService,
)
from sinonym.services.ethnicity import INITIAL_ONLY_CROSS_CULTURAL_SURNAMES
from sinonym.services.formatting import REVIEWED_UNBOUNDED_PREFIX_GIVEN_FORMS, SINGLE_LETTER_PINYIN_SYLLABLES
from sinonym.services.person_name_normalization import (
    DropReason,
    PersonNameNormalizationResult,
    PersonNameNormalizationService,
    PersonNameOutcome,
)
from sinonym.services.process_pool import PersistentMultiprocessNormalizer
from sinonym.utils.string_manipulation import StringManipulationUtils

LOGGER = logging.getLogger(__name__)

ParallelMode = Literal["auto", "never", "always"]
AUTO_MULTIPROCESS_MIN_NAMES = 25000
AUTO_MULTIPROCESS_MIN_BATCHES = 4000
LINUX_AUTO_MULTIPROCESS_MIN_NAMES = 10000
LINUX_AUTO_MULTIPROCESS_MIN_BATCHES = 2000
BILINGUAL_SURNAME_STRENGTH_RATIO_MIN = 5.0
BILINGUAL_ROMAN_HAN_SOURCE_ORDER_RATIO_MAX = 12.0
BILINGUAL_ENDPOINT_PAIR_COUNT = 2
TWO_TOKEN_NAME_COUNT = 2
SPACED_ALL_CHINESE_GROUP_COUNT = 2
THREE_CHARACTER_ALL_CHINESE_TOKEN_COUNT = 3
SPACED_HAN_PREFIX_SURNAME_RATIO_MIN = 5.0
CAMEL_CASE_LAST_SURNAME_RATIO_MIN = 5.0
LEADING_ET_AL_TOKEN_COUNT = 2
LEADING_ET_AL_MARKER = "et al."
CURATED_COMPOUND_SURNAME_FORMS = frozenset((*COMPOUND_VARIANTS.keys(), *COMPOUND_VARIANTS.values()))
HAN_SURNAME_POSITION_SOURCE_READINGS = frozenset(source_reading for _character, source_reading in HAN_SURNAME_POSITION_READINGS)
CompactHanRomanCandidate = tuple[list[str], list[str], list[str], float, bool]


def _should_use_multiprocessing(
    *,
    item_count: int,
    parallel: ParallelMode,
    auto_threshold: int | None,
    default_auto_threshold: int,
    linux_auto_threshold: int,
    max_workers: int | None,
) -> bool:
    """Return whether a high-level wrapper should use a process pool."""
    if parallel not in ("auto", "never", "always"):
        message = "parallel must be 'auto', 'never', or 'always'"
        raise ValueError(message)
    if item_count == 0 or parallel == "never":
        return False
    if parallel == "always":
        return True
    resolved_threshold = auto_threshold
    if resolved_threshold is None:
        resolved_threshold = linux_auto_threshold if platform.system() == "Linux" else default_auto_threshold
    if resolved_threshold < 1:
        message = "auto multiprocessing threshold must be >= 1"
        raise ValueError(message)
    return max_workers != 1 and item_count >= resolved_threshold


def _auto_start_method(mp_start_method: str) -> str:
    """Resolve the high-level auto start method."""
    if mp_start_method != "auto":
        return mp_start_method
    return "spawn"


def _validate_batch_policy(format_threshold: float, minimum_batch_size: int | None = None) -> None:
    """Validate public batch-policy arguments before analysis starts."""
    if not math.isfinite(format_threshold) or not 0.0 <= format_threshold <= 1.0:
        message = "format_threshold must be finite and between 0.0 and 1.0"
        raise ValueError(message)
    if minimum_batch_size is not None and minimum_batch_size < 1:
        message = "minimum_batch_size must be >= 1"
        raise ValueError(message)


# ════════════════════════════════════════════════════════════════════════════════
# MAIN CHINESE NAME DETECTOR CLASS
# ════════════════════════════════════════════════════════════════════════════════


class ChineseNameDetector:
    """Main Chinese name detection and normalization service."""

    def __init__(self, config: ChineseNameConfig | None = None, weights: list[float] | None = None):
        self._config = config or ChineseNameConfig.create_default()
        self._cache_service = PinyinCacheService(self._config)
        self._normalizer = NormalizationService(self._config, self._cache_service)
        self._person_name_normalizer = PersonNameNormalizationService()
        self._east_asian_name_order = EastAsianNameOrderService()
        self._data_service = DataInitializationService(self._config, self._cache_service, self._normalizer)
        self._initialization_lock = threading.RLock()
        self._data: NameDataStructures | None = None
        self._surname_resolver: SurnameResolver | None = None
        self._weights = weights  # Store weights to pass to parsing service

        # Service instances (initialized after data loading)
        self._ethnicity_service: EthnicityClassificationService | None = None
        self._parsing_service: NameParsingService | None = None
        self._formatting_service: NameFormattingService | None = None
        self._non_person_input_service: NonPersonInputDetectionService | None = None
        self._batch_analysis_service: BatchAnalysisService | None = None

        # Initialize data structures
        self._initialize()

    def _initialize(self) -> None:
        """Initialize cache and data structures."""
        try:
            self._ensure_initialized()
        except Exception as e:  # noqa: BLE001 - construction keeps lazy initialization fallback semantics.
            LOGGER.warning("Failed to initialize at construction: %s. Will initialize lazily.", e)

    def _initialize_services(self, data: NameDataStructures) -> None:
        """Build dependent services and publish them only after all succeed."""
        context = ServiceContext(self._config, self._normalizer, data)
        surname_resolver = SurnameResolver(data, self._normalizer)
        ethnicity_service = EthnicityClassificationService(context)
        parsing_service = NameParsingService(context, weights=self._weights)
        formatting_service = NameFormattingService(context)
        non_person_input_service = NonPersonInputDetectionService(self._config, self._normalizer, data)
        batch_analysis_service = BatchAnalysisService(
            parsing_service,
            ethnicity_service=ethnicity_service,
            dependencies=BatchAnalysisDependencies(
                min_tokens_required=self._config.min_tokens_required,
                # Batch policy consumes only Chinese parse/evidence fields.
                # Public batch APIs attach canonical sidecars after batch
                # analysis; the terminal TIMO resolver performs its sole
                # scalar resolution later.
                individual_parser=self._normalize_chinese_name,
                input_failure=self._initial_input_failure,
                classification_input=self._chinese_classification_input,
                surname_resolver=surname_resolver,
            ),
        )
        self._surname_resolver = surname_resolver
        self._ethnicity_service = ethnicity_service
        self._parsing_service = parsing_service
        self._formatting_service = formatting_service
        self._non_person_input_service = non_person_input_service
        self._batch_analysis_service = batch_analysis_service
        self._data = data

    def _ensure_initialized(self) -> None:
        """Ensure data is initialized (lazy initialization)."""
        if self._data is not None:
            return
        with self._initialization_lock:
            if self._data is not None:
                return
            data = self._data_service.initialize_data_structures()
            try:
                self._normalizer.set_data_context(data)
                self._initialize_services(data)
            except Exception:
                self._normalizer.set_data_context(None)
                raise

    def _require_surname_resolver(self) -> SurnameResolver:
        """Return the initialized surname resolver."""
        if self._surname_resolver is not None:
            return self._surname_resolver
        message = "surname resolver is not initialized"
        raise RuntimeError(message)

    def _is_surname_group(self, tokens: list[str], normalized_cache: dict[str, str]) -> bool:
        """Return whether a romanized Han token group is a surname component."""
        if len(tokens) == 1:
            return self._require_surname_resolver().parser_is_surname(tokens)

        normalized_tokens = [self._normalizer.get_normalized(token, normalized_cache) for token in tokens]
        spaced = " ".join(normalized_tokens)
        compact = "".join(normalized_tokens)

        return (
            spaced in self._data.compound_surnames
            or spaced in self._data.compound_surnames_normalized
            or compact in self._data.compound_original_format_map
        )

    def _is_compound_surname_group(self, tokens: list[str], normalized_cache: dict[str, str]) -> bool:
        """Return whether a romanized Han token group is a compound surname."""
        return len(tokens) > 1 and self._is_surname_group(tokens, normalized_cache)

    def _is_curated_compound_surname_group(self, tokens: list[str], normalized_cache: dict[str, str]) -> bool:
        """Return whether a romanized Han token group is an explicitly curated compound surname."""
        if len(tokens) <= 1:
            return False

        normalized_tokens = [self._normalizer.get_normalized(token, normalized_cache) for token in tokens]
        spaced = " ".join(normalized_tokens)
        compact = "".join(normalized_tokens)
        return spaced in CURATED_COMPOUND_SURNAME_FORMS or compact in CURATED_COMPOUND_SURNAME_FORMS

    def is_curated_compound_surname(self, surname: str) -> bool:
        """Return whether a complete string is an explicitly curated compound surname."""
        return self._is_curated_compound_surname_group(surname.split(), {})

    def _surname_group_strength(self, pinyin_tokens: list[str] | tuple[str, ...], han_group: str = "") -> float:
        """Return the strongest surname-frequency signal for a Han-backed pinyin group."""
        if not pinyin_tokens:
            return 0.0
        if not han_group:
            message = "_surname_group_strength requires Han provenance for pinyin tokens"
            raise ValueError(message)

        surname_resolver = self._require_surname_resolver()
        frequencies = [surname_resolver.parser_frequency((han_group,))]
        if len(pinyin_tokens) > 1:
            frequencies.append(surname_resolver.parser_frequency(pinyin_tokens))
            frequencies.append(surname_resolver.parser_frequency(("".join(pinyin_tokens),)))
        return max(frequencies, default=0.0)

    def _has_only_cjk_token_groups(self, normalized_input: NormalizedInput) -> bool:
        """Return whether separator-delimited input tokens are all CJK characters."""
        return len(normalized_input.tokens) > 1 and all(
            token and all(self._config.cjk_pattern.search(char) for char in token) for token in normalized_input.tokens
        )

    def _format_parse_result(
        self,
        surname_tokens: list[str],
        given_tokens: list[str],
        normalized_input: NormalizedInput,
        original_order: list[str],
        *,
        original_compound_format: str | None = None,
    ) -> ParseResult:
        """Format parsed components and attach stable structured name fields."""
        try:
            given_tokens = self._native_bound_given_tokens(normalized_input, given_tokens)
            surname_tokens = self._han_surname_position_readings(
                surname_tokens,
                normalized_input,
                original_order,
            )
            allow_surname_like_given_split = self._allows_surname_like_given_split(normalized_input)
            syllabic_single_letter_tokens = self._native_single_letter_given_tokens(normalized_input, given_tokens)
        except ValueError as error:
            return ParseResult.failure(str(error))

        selected_format = NameFormat.GIVEN_FIRST if original_order and original_order[0] == "given" else NameFormat.SURNAME_FIRST
        return self._formatting_service.materialize_parse_result(
            surname_tokens,
            given_tokens,
            selected_format,
            normalized_input.norm_map,
            normalized_input.compound_metadata,
            original_compound_format=original_compound_format,
            allow_surname_like_given_split=allow_surname_like_given_split,
            syllabic_single_letter_tokens=syllabic_single_letter_tokens,
        )

    def _han_surname_position_readings(
        self,
        surname_tokens: list[str],
        normalized_input: NormalizedInput,
        original_order: list[str],
    ) -> list[str]:
        """Correct a pypinyin reading only when its source Han is the assigned surname."""
        if not any(token.lower() in HAN_SURNAME_POSITION_SOURCE_READINGS for token in surname_tokens):
            return surname_tokens
        if not all(
            token and all(self._config.cjk_pattern.search(character) for character in token) for token in normalized_input.tokens
        ):
            return surname_tokens

        han_characters = "".join(normalized_input.tokens)
        surname_length = len(surname_tokens)
        if original_order[0] == "surname":
            surname_start = 0
        elif original_order[-1] == "surname":
            surname_start = len(han_characters) - surname_length
        else:
            return surname_tokens

        surname_end = surname_start + surname_length
        source_readings = normalized_input.roman_tokens[surname_start:surname_end]
        if tuple(token.lower() for token in surname_tokens) != tuple(token.lower() for token in source_readings):
            return surname_tokens

        surname_characters = han_characters[surname_start:surname_end]
        return [
            HAN_SURNAME_POSITION_READINGS.get((character, token.lower()), token)
            for character, token in zip(surname_characters, surname_tokens, strict=True)
        ]

    def _allows_surname_like_given_split(self, normalized_input: NormalizedInput) -> bool:
        """Return whether surname-like fused given tokens may be gold-split."""
        return not any(self._config.cjk_pattern.search(char) for token in normalized_input.tokens for char in token)

    def _native_single_letter_given_tokens(
        self,
        normalized_input: NormalizedInput,
        given_tokens: list[str],
    ) -> frozenset[str]:
        """Return one-letter given syllables proven by aligned source script."""
        given_keys = {self._normalizer.norm(token) for token in given_tokens}
        given_light_keys = {self._normalizer.norm_light(token) for token in given_tokens}
        syllables: set[str] = set()

        pairs = self._normalizer.aligned_bilingual_pairs(normalized_input)
        if pairs is not None:
            for pair in pairs:
                if self._normalizer.norm(pair.roman_token) not in given_keys:
                    continue
                if len(pair.roman_token) == 1 and pair.roman_token.isalpha():
                    syllables.add(pair.roman_token)
                syllables.update(part for part in pair.han_pinyin if len(part) == 1 and part.isalpha())

        source_characters = self._normalizer.han_roman_source_characters(normalized_input)
        if len(source_characters) == len(normalized_input.roman_tokens):
            syllables.update(
                token
                for token, _source_character in zip(
                    normalized_input.roman_tokens,
                    source_characters,
                    strict=True,
                )
                if len(token) == 1 and token.isalpha() and self._normalizer.norm(token) in given_keys
            )
        for han_pinyin in self._native_han_pinyin_sequences(normalized_input):
            if "".join(self._normalizer.norm_light(part) for part in han_pinyin) in given_light_keys:
                syllables.update(part for part in han_pinyin if len(part) == 1 and part.isalpha())
        return frozenset(syllables)

    def _native_bound_given_tokens(
        self,
        normalized_input: NormalizedInput,
        given_tokens: list[str],
    ) -> list[str]:
        """Insert given-name boundaries proved by an aligned Han reading."""
        sequences = self._native_han_pinyin_sequences(normalized_input)
        if not sequences:
            return given_tokens

        bound_tokens: list[str] = []
        for token in given_tokens:
            if token.count("'") == 1 and not token.startswith("'") and not token.endswith("'"):
                # The source already carries an explicit boundary or aspiration mark.
                # Let the formatter retain it while preserving split-token lineage.
                bound_tokens.append(token)
                continue
            target = self._normalizer.norm_light(token)
            matches = {
                "-".join(sequence[start:end])
                for sequence in sequences
                for start in range(len(sequence))
                for end in range(start + 2, len(sequence) + 1)
                if "".join(self._normalizer.norm_light(part) for part in sequence[start:end]) == target
            }
            bound_tokens.append(matches.pop() if len(matches) == 1 else token)
        return bound_tokens

    def _native_han_pinyin_sequences(self, normalized_input: NormalizedInput) -> tuple[tuple[str, ...], ...]:
        """Return pinyin sequences for source tokens that are entirely Han."""
        return tuple(
            tuple(self._cache_service.han_to_pinyin_fast(token))
            for token in normalized_input.tokens
            if self._is_han_source_token(token)
        )

    def _normalize_camel_case_pair(self, normalized_input: NormalizedInput) -> ParseResult | None:
        """Parse a whole-input camelCase pair using surname-first provenance."""
        tokens = list(normalized_input.roman_tokens)
        if len(tokens) != BILINGUAL_ENDPOINT_PAIR_COUNT or self._is_compound_surname_group(tokens, normalized_input.norm_map):
            return None

        first, last = tokens
        surname_resolver = self._require_surname_resolver()
        first_is_surname = surname_resolver.evidence_is_surname(first)
        last_is_surname = surname_resolver.evidence_is_surname(last)
        if not first_is_surname and not last_is_surname:
            return None

        first_freq = surname_resolver.evidence_frequency(first)
        last_freq = surname_resolver.evidence_frequency(last)
        last_wins = last_is_surname and (
            not first_is_surname
            or last.isupper()
            or (first_freq == 0 and last_freq > 0)
            or (first_freq > 0 and last_freq >= CAMEL_CASE_LAST_SURNAME_RATIO_MIN * first_freq)
        )
        if last_wins:
            surname_tokens, given_tokens = [last], [first]
            original_order = ["given", "surname"]
        else:
            surname_tokens, given_tokens = [first], [last]
            original_order = ["surname", "given"]

        return self._format_parse_result(surname_tokens, given_tokens, normalized_input, original_order)

    def _normalize_aligned_bilingual_name(self, normalized_input: NormalizedInput) -> ParseResult | None:
        """Parse explicit Roman/Han aligned names using Han surname identity."""
        pairs = self._normalizer.aligned_bilingual_pairs(normalized_input)
        if pairs is None:
            return None

        surname_strengths = [self._bilingual_pair_surname_strength(pair) for pair in pairs]
        source_order_result = self._normalize_weak_roman_han_pair_order(normalized_input, pairs, surname_strengths)
        if source_order_result is not None:
            return source_order_result

        return self._normalize_bilingual_pairs_by_han_identity(normalized_input, pairs, surname_strengths)

    def _normalize_bilingual_pairs_by_han_identity(
        self,
        normalized_input: NormalizedInput,
        pairs,
        surname_strengths: list[float],
    ) -> ParseResult | None:
        """Parse aligned bilingual pairs by the strongest Han surname signal."""
        best_strength = max(surname_strengths)
        if best_strength <= 0:
            return None

        best_index = surname_strengths.index(best_strength)
        next_best = max((strength for index, strength in enumerate(surname_strengths) if index != best_index), default=0.0)
        if next_best > 0 and best_strength / next_best < BILINGUAL_SURNAME_STRENGTH_RATIO_MIN:
            return None
        if best_index not in (0, len(pairs) - 1):
            return None

        surname_tokens = [pairs[best_index].roman_token]
        given_tokens = [pair.roman_token for index, pair in enumerate(pairs) if index != best_index]
        original_order = ["surname", "given"] if best_index == 0 else ["given", "surname"]

        return self._format_parse_result(surname_tokens, given_tokens, normalized_input, original_order)

    def _normalize_weak_roman_han_pair_order(
        self,
        normalized_input: NormalizedInput,
        pairs,
        surname_strengths: list[float],
    ) -> ParseResult | None:
        """Use source order for weak two-pair Latin-Han annotations."""
        if len(pairs) != BILINGUAL_ENDPOINT_PAIR_COUNT or any(len(pair.han_pinyin) != 1 for pair in pairs):
            return None
        if not self._is_roman_han_bilingual_pair_input(normalized_input):
            return None

        first_strength, last_strength = surname_strengths
        use_source_order = False
        if last_strength > 0:
            stronger = max(first_strength, last_strength)
            weaker = min(first_strength, last_strength)
            use_source_order = last_strength >= first_strength or stronger / weaker < BILINGUAL_ROMAN_HAN_SOURCE_ORDER_RATIO_MAX

        if not use_source_order:
            return None

        surname_tokens = [pairs[-1].roman_token]
        given_tokens = [pair.roman_token for pair in pairs[:-1]]
        return self._format_parse_result(surname_tokens, given_tokens, normalized_input, ["given", "surname"])

    def _is_roman_han_bilingual_pair_input(self, normalized_input: NormalizedInput) -> bool:
        """Return whether the source is exactly two Roman-Han aligned pairs."""
        tokens = list(normalized_input.tokens)
        if len(tokens) != BILINGUAL_ENDPOINT_PAIR_COUNT * 2:
            return False

        return all(
            self._is_roman_source_token(tokens[index]) and self._is_han_source_token(tokens[index + 1])
            for index in range(0, len(tokens), 2)
        )

    def _is_han_source_token(self, token: str) -> bool:
        """Return whether the original source token is entirely CJK."""
        return bool(token) and all(self._config.cjk_pattern.search(char) for char in token)

    def _is_roman_source_token(self, token: str) -> bool:
        """Return whether the original source token contains Roman letters and no CJK."""
        return bool(
            token and self._config.ascii_alpha_pattern.search(token) and not self._config.cjk_pattern.search(token),
        )

    def _bilingual_pair_surname_strength(self, pair) -> float:
        """Return surname strength from the Han side of an aligned bilingual pair."""
        surname_resolver = self._require_surname_resolver()
        han_freq = surname_resolver.parser_frequency((pair.han_token,))
        if len(pair.han_pinyin) == 1:
            return han_freq

        return max(
            han_freq,
            surname_resolver.parser_frequency(pair.han_pinyin),
            surname_resolver.parser_frequency(("".join(pair.han_pinyin),)),
        )

    def _normalize_compact_han_roman_name(self, normalized_input: NormalizedInput) -> ParseResult | None:
        """Parse compact Han names followed by an exact Roman transliteration."""
        compact_components = self._compact_han_roman_components(normalized_input)
        if compact_components is None:
            return None
        surname_tokens, given_tokens, original_order = compact_components

        return self._format_parse_result(surname_tokens, given_tokens, normalized_input, original_order)

    def _compact_han_roman_components(
        self,
        normalized_input: NormalizedInput,
    ) -> tuple[list[str], list[str], list[str]] | None:
        """Return parsed components for exact compact Han/Roman transliterations."""
        han_groups = [token for token in normalized_input.tokens if self._is_han_source_token(token)]
        roman_tokens = [
            clean_token
            for token in normalized_input.tokens
            if self._is_roman_source_token(token)
            for clean_token in [self._clean_source_roman_token(token)]
            if clean_token
        ]

        source_token_count = len(han_groups) + len(roman_tokens)
        if len(han_groups) == 1 and source_token_count == len(normalized_input.tokens) and roman_tokens:
            han_group = han_groups[0]
            han_pinyin = tuple(self._cache_service.han_to_pinyin_fast(han_group))
            if len(han_pinyin) >= self._config.min_tokens_required:
                surname_pinyin_length = self._han_surname_prefix_length(han_group, han_pinyin)
                if 0 < surname_pinyin_length < len(han_pinyin):
                    reversed_components = self._split_reversed_roman_tokens_for_han_prefix(
                        roman_tokens,
                        han_pinyin,
                        surname_pinyin_length,
                    )
                    if reversed_components is not None:
                        given_tokens, surname_tokens = reversed_components
                        return surname_tokens, given_tokens, ["given", "surname"]

            if len(han_pinyin) >= self._config.min_tokens_required and self._roman_tokens_match_han_pinyin(
                roman_tokens,
                han_pinyin,
            ):
                prefix_candidate: CompactHanRomanCandidate | None = None
                surname_pinyin_length = self._han_surname_prefix_length(han_group, han_pinyin)
                if 0 < surname_pinyin_length < len(han_pinyin):
                    split_tokens = self._split_roman_tokens_for_han_prefix(
                        roman_tokens,
                        han_pinyin,
                        surname_pinyin_length,
                    )
                    if split_tokens is not None:
                        surname_tokens, given_tokens = split_tokens
                        prefix_tokens = han_pinyin[:surname_pinyin_length]
                        prefix_candidate = (
                            surname_tokens,
                            given_tokens,
                            ["surname", "given"],
                            self._surname_group_strength(prefix_tokens, han_group[:surname_pinyin_length]),
                            self._is_curated_compound_surname_group(list(prefix_tokens), {}),
                        )

                suffix_candidate: CompactHanRomanCandidate | None = None
                surname_pinyin_length = self._han_surname_suffix_length(han_group, han_pinyin)
                if 0 < surname_pinyin_length < len(han_pinyin):
                    split_tokens = self._split_roman_tokens_for_han_suffix(
                        roman_tokens,
                        han_pinyin,
                        surname_pinyin_length,
                    )
                    if split_tokens is not None:
                        given_tokens, surname_tokens = split_tokens
                        suffix_tokens = han_pinyin[-surname_pinyin_length:]
                        suffix_candidate = (
                            surname_tokens,
                            given_tokens,
                            ["given", "surname"],
                            self._surname_group_strength(suffix_tokens, han_group[-surname_pinyin_length:]),
                            self._is_curated_compound_surname_group(list(suffix_tokens), {}),
                        )

                selected_candidate = self._select_han_roman_candidate(prefix_candidate, suffix_candidate)
                if selected_candidate is not None:
                    surname_tokens, given_tokens, original_order, _strength, _is_curated = selected_candidate
                    return surname_tokens, given_tokens, original_order
        return None

    def _select_han_roman_candidate(
        self,
        prefix_candidate: CompactHanRomanCandidate | None,
        suffix_candidate: CompactHanRomanCandidate | None,
    ) -> CompactHanRomanCandidate | None:
        """Choose between viable compact Han/Roman endpoint parses."""
        if prefix_candidate is None:
            return suffix_candidate
        if suffix_candidate is None:
            return prefix_candidate

        prefix_strength = prefix_candidate[3]
        suffix_strength = suffix_candidate[3]
        prefix_is_curated = prefix_candidate[4]
        suffix_is_curated = suffix_candidate[4]

        if prefix_is_curated != suffix_is_curated:
            return prefix_candidate if prefix_is_curated else suffix_candidate
        if suffix_strength >= prefix_strength * BILINGUAL_SURNAME_STRENGTH_RATIO_MIN:
            return suffix_candidate
        return prefix_candidate

    def _clean_source_roman_token(self, token: str) -> str:
        """Clean a source Roman token while preserving source capitalization."""
        folded = token.translate(self._config.roman_punctuation_fold_tr)
        return self._config.clean_roman_pattern.sub("", folded)

    def _roman_tokens_match_han_pinyin(self, roman_tokens: list[str], han_pinyin: tuple[str, ...]) -> bool:
        """Return whether Roman source tokens exactly transliterate the Han pinyin."""
        roman_joined = "".join(self._normalizer.norm_light(token) for token in roman_tokens)
        han_joined = "".join(self._normalizer.norm_light(token) for token in han_pinyin)
        return roman_joined == han_joined

    def _han_surname_prefix_length(self, han_group: str, han_pinyin: tuple[str, ...]) -> int:
        """Return the Han surname prefix length in pinyin tokens."""
        if len(han_pinyin) >= self._config.min_tokens_required:
            first_two_pinyin = list(han_pinyin[:2])
            first_two_han = han_group[:2]
            if self._require_surname_resolver().parser_frequency((first_two_han,)) > 0 or self._is_compound_surname_group(
                first_two_pinyin,
                {},
            ):
                return 2

        first_han = han_group[0]
        if self._require_surname_resolver().parser_frequency((first_han,)) > 0:
            return 1
        return 0

    def _han_surname_suffix_length(self, han_group: str, han_pinyin: tuple[str, ...]) -> int:
        """Return the Han surname suffix length in pinyin tokens."""
        if len(han_pinyin) >= self._config.min_tokens_required:
            last_two_pinyin = list(han_pinyin[-2:])
            last_two_han = han_group[-2:]
            if self._require_surname_resolver().parser_frequency((last_two_han,)) > 0 or self._is_compound_surname_group(
                last_two_pinyin,
                {},
            ):
                return 2

        last_han = han_group[-1]
        if self._require_surname_resolver().parser_frequency((last_han,)) > 0:
            return 1
        return 0

    def _split_roman_tokens_for_han_prefix(
        self,
        roman_tokens: list[str],
        han_pinyin: tuple[str, ...],
        prefix_length: int,
    ) -> tuple[list[str], list[str]] | None:
        """Split Roman tokens at the boundary matching a Han pinyin prefix."""
        prefix_target = "".join(self._normalizer.norm_light(token) for token in han_pinyin[:prefix_length])
        current = ""
        for index, token in enumerate(roman_tokens, start=1):
            current += self._normalizer.norm_light(token)
            if current == prefix_target:
                return roman_tokens[:index], roman_tokens[index:]
            if not prefix_target.startswith(current):
                return None
        return None

    def _split_roman_tokens_for_han_suffix(
        self,
        roman_tokens: list[str],
        han_pinyin: tuple[str, ...],
        suffix_length: int,
    ) -> tuple[list[str], list[str]] | None:
        """Split Roman tokens at the boundary matching a Han pinyin suffix."""
        suffix_target = "".join(self._normalizer.norm_light(token) for token in han_pinyin[-suffix_length:])
        current = ""
        for index in range(len(roman_tokens) - 1, -1, -1):
            current = self._normalizer.norm_light(roman_tokens[index]) + current
            if current == suffix_target:
                return roman_tokens[:index], roman_tokens[index:]
            if not suffix_target.endswith(current):
                return None
        return None

    def _split_reversed_roman_tokens_for_han_prefix(
        self,
        roman_tokens: list[str],
        han_pinyin: tuple[str, ...],
        prefix_length: int,
    ) -> tuple[list[str], list[str]] | None:
        """Match given-first Roman components to surname-first compact Han."""
        surname_target = "".join(self._normalizer.norm_light(token) for token in han_pinyin[:prefix_length])
        given_target = "".join(self._normalizer.norm_light(token) for token in han_pinyin[prefix_length:])
        for index in range(1, len(roman_tokens)):
            roman_given = "".join(self._normalizer.norm_light(token) for token in roman_tokens[:index])
            roman_surname = "".join(self._normalizer.norm_light(token) for token in roman_tokens[index:])
            if roman_given == given_target and roman_surname == surname_target:
                return roman_tokens[:index], roman_tokens[index:]
        return None

    def _normalize_surname_first_unbounded_prefix_given(
        self,
        normalized_input: NormalizedInput,
    ) -> ParseResult | None:
        """Keep an A/E-prefixed whole given token behind an evidenced surname."""
        if len(normalized_input.tokens) != TWO_TOKEN_NAME_COUNT or not all(
            self._is_roman_source_token(token) for token in normalized_input.tokens
        ):
            return None

        surname, given = normalized_input.roman_tokens
        if not self._require_surname_resolver().evidence_is_surname(surname):
            return None
        if self._normalizer.norm_light(given) not in REVIEWED_UNBOUNDED_PREFIX_GIVEN_FORMS:
            return None
        if not self._is_unbounded_single_letter_syllable_shape(given, leading_only=True):
            return None

        return self._format_parse_result([surname], [given], normalized_input, ["surname", "given"])

    def _normalize_spaced_all_chinese_name(self, normalized_input: NormalizedInput) -> ParseResult | None:
        """Parse all-Han names whose whitespace already separates name components."""
        if (
            len(normalized_input.tokens) != SPACED_ALL_CHINESE_GROUP_COUNT
            or len(normalized_input.roman_tokens) <= SPACED_ALL_CHINESE_GROUP_COUNT
        ):
            return None

        first_group = self._cache_service.han_to_pinyin_fast(normalized_input.tokens[0])
        last_group = self._cache_service.han_to_pinyin_fast(normalized_input.tokens[1])
        if not first_group or not last_group:
            return None

        # OCR/noisy spacing can split a compound surname across the group boundary.
        boundary_compound = first_group + last_group[:1] if len(first_group) == 1 and len(last_group) > 1 else []
        if boundary_compound and self._is_surname_group(boundary_compound, normalized_input.norm_map):
            surname_tokens = boundary_compound
            given_tokens = last_group[1:]
            original_order = ["surname", "given"]
        else:
            first_is_surname = self._is_surname_group(first_group, normalized_input.norm_map)
            last_is_surname = self._is_surname_group(last_group, normalized_input.norm_map)
            last_is_compound_surname = self._is_compound_surname_group(last_group, normalized_input.norm_map)
            first_strength = self._surname_group_strength(
                first_group,
                normalized_input.tokens[0],
            )
            last_strength = self._surname_group_strength(
                last_group,
                normalized_input.tokens[1],
            )
            last_compound_surname_wins = last_is_compound_surname and (
                self._is_curated_compound_surname_group(last_group, normalized_input.norm_map)
                or (last_strength > 0 and last_strength >= first_strength * BILINGUAL_SURNAME_STRENGTH_RATIO_MIN)
            )

            if (
                last_is_surname
                and not first_is_surname
                and len(first_group) > 1
                and len(last_group) == 1
                and self._spaced_han_prefers_prefix_surname(first_group, last_group)
            ):
                return None
            if last_is_surname and (not first_is_surname or last_compound_surname_wins):
                surname_tokens = last_group
                given_tokens = first_group
                original_order = ["given", "surname"]
            elif first_is_surname:
                surname_tokens = first_group
                given_tokens = last_group
                original_order = ["surname", "given"]
            else:
                return None

        return self._format_parse_result(surname_tokens, given_tokens, normalized_input, original_order)

    def _spaced_han_prefers_prefix_surname(self, first_group: list[str], last_group: list[str]) -> bool:
        """Return whether noisy spacing likely split a surname-first Han name's given name."""
        if not first_group or not last_group:
            return False

        surname_resolver = self._require_surname_resolver()
        first_freq = surname_resolver.parser_frequency((first_group[0],))
        last_freq = surname_resolver.parser_frequency((last_group[0],))
        if last_freq <= 0:
            return first_freq > 0
        return first_freq / last_freq >= SPACED_HAN_PREFIX_SURNAME_RATIO_MIN

    # Public API methods
    def get_cache_info(self) -> CacheInfo:
        """Get cache information."""
        return self._cache_service.get_cache_info()

    def _initial_input_failure(self, raw_name: str) -> ParseResult | None:
        """Return an early failure before normalization, or initialize services."""
        if not raw_name or len(raw_name) > self._config.max_name_length:
            return ParseResult.failure("invalid input length")

        if all(c in string.punctuation + string.whitespace for c in raw_name):
            return ParseResult.failure("name contains only punctuation/whitespace")

        if self._normalizer._text_preprocessor.contains_non_chinese_scripts(raw_name):
            return ParseResult.failure("contains non-Chinese characters")

        self._ensure_initialized()

        if self._non_person_input_service is None:
            return None

        non_person_reason = self._non_person_input_service.failure_reason(raw_name)
        if non_person_reason is None:
            return None
        return ParseResult.failure(non_person_reason)

    def _normalize_chinese_name(self, raw_name: str) -> ParseResult:
        """
        Run the legacy Chinese detection and normalization pipeline unchanged.

        Returns ParseResult with:
        - success=True, result=formatted_name if Chinese name detected
        - success=False, error_message=reason if not Chinese name
        """
        initial_failure = self._initial_input_failure(raw_name)
        if initial_failure is not None:
            return initial_failure

        raw_name = self._chinese_classification_input(raw_name)

        # Use new normalization service for cleaner pipeline
        normalized_input = self._normalizer.apply(raw_name)

        if len(normalized_input.roman_tokens) < self._config.min_tokens_required:
            return ParseResult.failure(f"needs at least {self._config.min_tokens_required} Roman tokens")

        # Check if this is an all-Chinese input first
        is_all_chinese = not normalized_input.cleaned.isascii() and self._normalizer._text_preprocessor.is_all_chinese_input(
            normalized_input.cleaned,
        )

        # Exact alternating Roman/Han alignment is stronger evidence than the
        # Roman-only ethnicity gate, especially for polyphonic Han surnames.
        if self._config.cjk_pattern.search(raw_name) and self._config.ascii_alpha_pattern.search(raw_name):
            aligned_bilingual_result = self._normalize_aligned_bilingual_name(normalized_input)
            if aligned_bilingual_result is not None:
                return aligned_bilingual_result

        # Check for non-Chinese ethnicity using normalized tokens (consistent for all inputs)
        non_chinese_result = self._ethnicity_service.classify_ethnicity(
            normalized_input.roman_tokens,
            normalized_input.norm_map,
            raw_name,
        )

        if non_chinese_result.success is False:
            return non_chinese_result

        contextual_taiwan_result = self._normalize_contextual_taiwan_name(normalized_input)
        if contextual_taiwan_result is not None:
            return contextual_taiwan_result

        compact_han_roman_result = self._normalize_compact_han_roman_name(normalized_input)
        if compact_han_roman_result is not None:
            return compact_han_roman_result

        surname_first_prefix_result = self._normalize_surname_first_unbounded_prefix_given(normalized_input)
        if surname_first_prefix_result is not None:
            return surname_first_prefix_result

        # Try parsing in both orders - for all-Chinese inputs, choose best scoring parse

        if self._has_only_cjk_token_groups(normalized_input):
            grouped_result = self._normalize_spaced_all_chinese_name(normalized_input)
            if grouped_result is not None:
                return grouped_result

        if is_all_chinese and len(normalized_input.roman_tokens) == self._config.min_tokens_required:
            # For all-Chinese 2-token inputs, ALWAYS assume surname-first order
            # Two-character Chinese names are always (surname, given_name)
            tokens = list(normalized_input.roman_tokens)
            token1, token2 = tokens[0], tokens[1]

            # Check if first token can be a surname
            surname_resolver = self._require_surname_resolver()
            token1_is_surname = surname_resolver.parser_is_surname((token1,))

            # For 2-character all-Chinese names, use surname-first if token1 is a valid surname
            if token1_is_surname:
                best_result = ([token1], [token2])
            else:
                # Fallback: if token1 is not a surname, try token2 as surname (less common but possible)
                token2_is_surname = surname_resolver.parser_is_surname((token2,))
                best_result = ([token2], [token1]) if token2_is_surname else None

            if best_result:
                surname_tokens, given_tokens = best_result
                original_order = ["surname", "given"] if token1_is_surname else ["given", "surname"]
                return self._format_parse_result(surname_tokens, given_tokens, normalized_input, original_order)
        elif is_all_chinese and len(normalized_input.roman_tokens) == THREE_CHARACTER_ALL_CHINESE_TOKEN_COUNT:
            # For 3-character all-Chinese names: check compound surname vs single surname
            tokens = list(normalized_input.roman_tokens)

            # Try both possibilities and see which one the parsing service accepts
            # Option 1: First two tokens as compound surname + third as given
            compound_parse = self._parsing_service.parse_name_order_tokens(
                tokens,
                normalized_input.norm_map,
                normalized_input.compound_metadata,
                normalized_input.spaced_compound_spans,
            )

            if (
                compound_parse is not None
                and len(compound_parse[0]) == BILINGUAL_ENDPOINT_PAIR_COUNT
                and len(compound_parse[1]) == 1
            ):
                # Parsing service recognized first two as compound surname
                best_result = (compound_parse[0], compound_parse[1])
            else:
                # Option 2: First token as single surname + last two as given name
                best_result = ([tokens[0]], tokens[1:])

            if best_result:
                surname_tokens, given_tokens = best_result
                # For 3-character all-Chinese, original order is surname-first.
                return self._format_parse_result(surname_tokens, given_tokens, normalized_input, ["surname", "given"])
        else:
            if normalized_input.from_camel_case_pair:
                camel_result = self._normalize_camel_case_pair(normalized_input)
                if camel_result is not None:
                    return camel_result

            # Evaluate both order hypotheses and pick the best-scoring parse
            original_tokens = list(normalized_input.roman_tokens)
            best_candidate = None

            # For two tokens the parser already evaluates both endpoint surname
            # candidates. Reversing repeats the same candidate set; uncertain
            # fallback parses and parenthetical-order hints retain the full path.
            if len(original_tokens) == TWO_TOKEN_NAME_COUNT and not normalized_input.surname_first_parenthetical_hint:
                direct_parse = self._parsing_service._best_parse_tokens(
                    original_tokens,
                    normalized_input.norm_map,
                    normalized_input.compound_metadata,
                    normalized_input.spaced_compound_spans,
                )
                if direct_parse is not None:
                    surname_tokens, given_tokens, original_compound_surname = direct_parse
                    best_candidate = {
                        "surname_tokens": surname_tokens,
                        "given_tokens": given_tokens,
                        "score": 0.0,
                        "order_tokens": original_tokens,
                        "used_original": True,
                        "original_compound_format": original_compound_surname,
                    }

            if best_candidate is None:
                for order in (normalized_input.roman_tokens, normalized_input.roman_tokens[::-1]):
                    order_tokens = list(order)
                    spaced_compound_spans = normalized_input.spaced_compound_spans
                    if order_tokens != original_tokens:
                        token_count = len(order_tokens)
                        spaced_compound_spans = tuple(span.reversed(token_count) for span in spaced_compound_spans)
                    parse_result = self._parsing_service.parse_name_order_tokens(
                        order_tokens,
                        normalized_input.norm_map,
                        normalized_input.compound_metadata,
                        spaced_compound_spans,
                    )
                    if parse_result is None:
                        continue

                    surname_tokens, given_tokens, original_compound_surname = parse_result
                    score = self._parsing_service.calculate_parse_score(
                        surname_tokens,
                        given_tokens,
                        original_tokens,
                        normalized_input.norm_map,
                        is_all_chinese=False,
                        original_compound_format=original_compound_surname,
                        surname_first_parenthetical_hint=normalized_input.surname_first_parenthetical_hint,
                    )
                    used_original = order_tokens == original_tokens

                    candidate = {
                        "surname_tokens": surname_tokens,
                        "given_tokens": given_tokens,
                        "score": score,
                        "order_tokens": order_tokens,
                        "used_original": used_original,
                        "original_compound_format": original_compound_surname,
                    }

                    if (
                        best_candidate is None
                        or candidate["score"] > best_candidate["score"]
                        or (
                            candidate["score"] == best_candidate["score"]
                            and candidate["used_original"]
                            and not best_candidate["used_original"]
                        )
                    ):
                        best_candidate = candidate

            if best_candidate is not None:
                surname_tokens = best_candidate["surname_tokens"]
                given_tokens = best_candidate["given_tokens"]
                order_tokens = best_candidate["order_tokens"]
                used_original = best_candidate["used_original"]
                # Determine original input order relative to detected parse.
                k = len(surname_tokens)
                is_surname_first_in_this_order = list(order_tokens[:k]) == surname_tokens
                is_surname_last_in_this_order = list(order_tokens[-k:]) == surname_tokens

                # Compound surname fallback: surname_tokens may be sub-tokens
                # of a single original token (e.g. ['Ou','yang'] from 'Ouyang')
                if not is_surname_first_in_this_order and not is_surname_last_in_this_order:
                    joined_surname = "".join(surname_tokens).lower()
                    if order_tokens[0].lower() == joined_surname:
                        is_surname_first_in_this_order = True
                    elif order_tokens[-1].lower() == joined_surname:
                        is_surname_last_in_this_order = True

                original_is_given_first = is_surname_last_in_this_order if used_original else is_surname_first_in_this_order
                original_order = ["given", "surname"] if original_is_given_first else ["surname", "given"]
                return self._format_parse_result(
                    surname_tokens,
                    given_tokens,
                    normalized_input,
                    original_order,
                    original_compound_format=best_candidate["original_compound_format"],
                )

        return ParseResult.failure("name not recognised as Chinese")

    def _normalize_contextual_taiwan_name(self, normalized_input: NormalizedInput) -> ParseResult | None:
        """Parse the narrowly admitted Taiwan romanization contexts."""
        given_parts = self._ethnicity_service.contextual_taiwan_given_parts(normalized_input.roman_tokens)
        if given_parts is None:
            return None

        norm_map = dict(normalized_input.norm_map)
        # BOCA's ``Jr`` corresponds to the Mandarin syllable ``Zhi``. Keep the
        # visible Taiwan spelling while supplying internal given-name evidence.
        if given_parts == ("tsung", "jr"):
            norm_map["jr"] = "zhi"
        contextual_input = replace(normalized_input, norm_map=norm_map)
        return self._format_parse_result(
            [normalized_input.roman_tokens[-1]],
            list(given_parts),
            contextual_input,
            ["given", "surname"],
        )

    def _leading_et_al_normalization(self, raw_name: str) -> PersonNameNormalizationResult | None:
        """Return the audited normalization for one exact leading citation marker."""
        prefix = raw_name.lstrip()
        if (
            len(prefix) <= len(LEADING_ET_AL_MARKER)
            or prefix[: len(LEADING_ET_AL_MARKER)].casefold() != LEADING_ET_AL_MARKER
            or not prefix[len(LEADING_ET_AL_MARKER)].isspace()
        ):
            return None
        normalized = self._person_name_normalizer.normalize_text(raw_name)
        if normalized.canonical_name is None or len(normalized.dropped_tokens) < LEADING_ET_AL_TOKEN_COUNT:
            return None
        first, second = normalized.dropped_tokens[:LEADING_ET_AL_TOKEN_COUNT]
        has_exact_prefix = (
            first.text.casefold() == "et"
            and second.text.casefold() == "al."
            and first.reason is DropReason.CONNECTOR
            and second.reason is DropReason.CONNECTOR
        )
        return normalized if has_exact_prefix else None

    def _chinese_classification_input(self, raw_name: str) -> str:
        """Remove only an audited leading ``Et al.`` citation contaminant."""
        if self._leading_et_al_normalization(raw_name) is None:
            return raw_name
        return raw_name.lstrip()[len(LEADING_ET_AL_MARKER) :].lstrip()

    @staticmethod
    def _canonical_components_from_parsed(parsed: ParsedName) -> NameComponents:
        """Convert legacy parsed components to immutable canonical components."""
        if parsed.order == ["given", "middle", "surname"]:
            expanded_order = (
                ("given",) * len(parsed.given_tokens)
                + ("middle",) * len(parsed.middle_tokens)
                + ("surname",) * len(parsed.surname_tokens)
            )
        else:
            counts = {
                "given": len(parsed.given_tokens),
                "middle": len(parsed.middle_tokens),
                "surname": len(parsed.surname_tokens),
            }
            occurrences = {role: parsed.order.count(role) for role in counts}
            expanded: list[str] = []
            for role in parsed.order:
                count = counts.get(role, 0)
                if occurrences.get(role) == 1:
                    expanded.extend([role] * count)
                elif count:
                    expanded.append(role)
            expanded_order = tuple(expanded)
        return NameComponents(
            given_name=parsed.given_name,
            middle_name=parsed.middle_name,
            surname=parsed.surname,
            given_tokens=tuple(parsed.given_tokens),
            middle_tokens=tuple(parsed.middle_tokens),
            surname_tokens=tuple(parsed.surname_tokens),
            order=expanded_order,
        )

    def _canonical_name_from_chinese_result(self, raw_name: str, result: ParseResult) -> CanonicalName | None:
        """Build canonical metadata from the final selected Chinese parse."""
        if not result.success or result.parsed is None or not isinstance(result.result, str):
            return None
        normalized = self._canonical_components_from_parsed(result.parsed)
        source_parsed = result.parsed_original_order or result.parsed
        source = self._canonical_source_components(raw_name, source_parsed, normalized)
        return CanonicalName(
            source_text=raw_name,
            text=result.result,
            source=source,
            normalized=normalized,
        )

    @staticmethod
    def _component_token_key(token: str) -> str:
        """Return a comparison key for source-to-normalized token lineage."""
        if token.isalnum():
            return token.casefold()
        return "".join(character.casefold() for character in token if character.isalnum())

    @staticmethod
    def _ordered_component_tokens(components: NameComponents) -> list[str]:
        """Rebuild a component token stream from its positional role order."""
        queues = {
            "given": iter(components.given_tokens),
            "middle": iter(components.middle_tokens),
            "surname": iter(components.surname_tokens),
            "suffix": iter(components.suffix_tokens),
        }
        return [next(queues[role]) for role in components.order]

    def _routing_name_tokens(self, baseline: CanonicalName) -> tuple[str, ...]:
        """Return cleaned baseline tokens that carry personal-name semantics."""
        components = baseline.normalized
        ordered = self._ordered_component_tokens(components)
        return tuple(token for role, token in zip(components.order, ordered, strict=True) if role != "suffix")

    def _east_asian_routing_surface(self, raw_name: str, baseline: CanonicalName) -> str:
        """Return cleaned routing text while retaining comma-order authority."""
        if "," in raw_name:
            source_name_order = tuple(role for role in baseline.source.order if role != "suffix")
            if source_name_order and source_name_order[0] == "surname" and "given" in source_name_order:
                normalized = baseline.normalized
                given_span = " ".join(part for part in (normalized.given_name, normalized.middle_name) if part)
                return f"{normalized.surname}, {given_span}"
        return " ".join(self._routing_name_tokens(baseline))

    def _canonical_source_components(
        self,
        raw_name: str,
        parsed_original: ParsedName,
        normalized: NameComponents,
    ) -> NameComponents:
        """Preserve raw Latin token boundaries while retaining parsed roles."""
        citation_normalization = self._leading_et_al_normalization(raw_name)
        if citation_normalization is not None:
            assert citation_normalization.canonical_name is not None
            source_tokens = self._ordered_component_tokens(citation_normalization.canonical_name.source)
            source = self._source_components_from_tokens(
                source_tokens[LEADING_ET_AL_TOKEN_COUNT:],
                parsed_original,
                normalized,
                require_unique_roles=False,
                source_aligned_fallback=True,
            )
            assert source is not None
            return source

        simple_tokens = self._simple_source_tokens(raw_name)
        if simple_tokens is not None:
            simple_source = self._source_components_from_tokens(
                simple_tokens,
                parsed_original,
                normalized,
                require_unique_roles=True,
            )
            if simple_source is not None:
                return simple_source
            aligned_source = self._source_components_from_tokens(
                simple_tokens,
                parsed_original,
                normalized,
                require_unique_roles=False,
                source_aligned_fallback=True,
            )
            assert aligned_source is not None
            return aligned_source

        source_result = self._person_name_normalizer.normalize_text(raw_name)
        if source_result.canonical_name is None:
            return self._canonical_components_from_parsed(parsed_original)

        generic_source = source_result.canonical_name.source
        ordered_tokens = self._ordered_component_tokens(generic_source)
        source = self._source_components_from_tokens(
            ordered_tokens,
            parsed_original,
            normalized,
            require_unique_roles=False,
        )
        assert source is not None
        return source

    def _simple_source_tokens(self, raw_name: str) -> tuple[str, ...] | None:
        """Return already-clean ASCII source tokens, or abstain."""
        return self._normalizer.simple_latin_tokens(raw_name)

    def _source_aligned_fallback_roles(
        self,
        ordered_tokens: list[str] | tuple[str, ...],
        parsed_original: ParsedName,
    ) -> list[str]:
        """Align source tokens to parsed roles while allowing fused source tokens."""
        parsed_components = self._canonical_components_from_parsed(parsed_original)
        normalized_tokens = self._ordered_component_tokens(parsed_components)
        normalized_roles = list(parsed_components.order)
        fallback_roles: list[str] = []
        normalized_index = 0

        for source_index, source_token in enumerate(ordered_tokens):
            if normalized_index >= len(normalized_roles):
                fallback_roles.append("given")
                continue

            role = normalized_roles[normalized_index]
            fallback_roles.append(role)
            remaining_source = len(ordered_tokens) - source_index - 1
            max_end = len(normalized_tokens) - remaining_source
            matched_end = None
            source_key = self._component_token_key(source_token)
            for end in range(normalized_index + 1, max_end + 1):
                if normalized_roles[end - 1] != role:
                    break
                normalized_key = self._component_token_key("".join(normalized_tokens[normalized_index:end]))
                if normalized_key == source_key:
                    matched_end = end
                    break
            normalized_index = matched_end if matched_end is not None else normalized_index + 1

        return fallback_roles

    def _source_components_from_tokens(
        self,
        ordered_tokens: list[str] | tuple[str, ...],
        parsed_original: ParsedName,
        normalized: NameComponents,
        *,
        require_unique_roles: bool,
        source_aligned_fallback: bool = False,
    ) -> NameComponents | None:
        """Assign source tokens to normalized roles without changing token text."""
        normalized_by_role = {
            "given": normalized.given_tokens,
            "middle": normalized.middle_tokens,
            "surname": normalized.surname_tokens,
            "suffix": normalized.suffix_tokens,
        }
        role_keys = {
            role: {
                *(self._component_token_key(token) for token in tokens),
                self._component_token_key("".join(tokens)),
            }
            for role, tokens in normalized_by_role.items()
            if tokens
        }
        fallback_roles: list[str] | None = None
        assigned: list[tuple[str, str]] = []
        for index, token in enumerate(ordered_tokens):
            key = self._component_token_key(token)
            candidates = [role for role, keys in role_keys.items() if key and key in keys]
            if require_unique_roles and len(candidates) != 1:
                return None
            if len(candidates) == 1:
                assigned.append((candidates[0], token))
                continue
            if fallback_roles is None:
                fallback_roles = (
                    self._source_aligned_fallback_roles(ordered_tokens, parsed_original)
                    if source_aligned_fallback
                    else [role for role in parsed_original.order if role in normalized_by_role]
                )
            fallback = fallback_roles[index] if index < len(fallback_roles) else "given"
            assigned.append((fallback, token))

        tokens_by_role = {
            role: tuple(token for assigned_role, token in assigned if assigned_role == role) for role in normalized_by_role
        }
        return NameComponents(
            given_name=" ".join(tokens_by_role["given"]),
            middle_name=" ".join(tokens_by_role["middle"]),
            surname=" ".join(tokens_by_role["surname"]),
            suffix=" ".join(tokens_by_role["suffix"]),
            given_tokens=tokens_by_role["given"],
            middle_tokens=tokens_by_role["middle"],
            surname_tokens=tokens_by_role["surname"],
            suffix_tokens=tokens_by_role["suffix"],
            order=tuple(role for role, _token in assigned),
        )

    def normalize_person_name(self, raw_name: str) -> CanonicalName | None:
        """Return a canonical name for one person-like raw string.

        The result applies hard identity evidence, affirmative Chinese
        formatting, and conservative East Asian order routing without
        requiring the legacy Chinese recognizer to succeed. Invalid or obvious
        non-person inputs return ``None``.
        """
        return self._normalize_person_name_with_chinese_result(raw_name, None)

    def _normalize_person_name_with_chinese_result(
        self,
        raw_name: str,
        chinese_result: ParseResult | None,
    ) -> CanonicalName | None:
        """Normalize one person while reusing an already-computed Chinese result."""
        baseline = self._person_name_baseline(raw_name)
        if baseline is None:
            return None
        return self._canonical_person_name_from_baseline(
            raw_name,
            baseline,
            chinese_result=chinese_result,
        )

    def _canonical_person_name_from_baseline(
        self,
        raw_name: str,
        baseline: CanonicalName,
        *,
        chinese_result: ParseResult | None = None,
    ) -> CanonicalName:
        """Apply hard identity evidence, then Chinese and soft order policy."""
        routing_surface = self._east_asian_routing_surface(raw_name, baseline)
        decision = self._infer_east_asian_name_order_decision(
            routing_surface,
            legacy_raw_name=raw_name,
        )
        if decision is not None and east_asian_evidence_resolution_reason(decision.reason) in {
            ResolutionReason.JAPANESE_ITERATION_MARK_ASSIGNMENT,
            ResolutionReason.IDENTITY_BACKED_EXACT_ASSIGNMENT,
        }:
            routed = self._canonical_name_from_order_decision(baseline, decision)
            if routed is not None:
                return routed

        if self._east_asian_name_order._is_reviewed_japanese_given_first_exact_surface(routing_surface):
            return baseline

        chinese = self._canonical_chinese_name_with_source(
            raw_name,
            baseline.source,
            result=chinese_result,
        )
        if chinese is not None:
            return chinese
        if decision is None:
            return baseline
        return self._canonical_name_from_order_decision(baseline, decision) or baseline

    def _canonical_chinese_name_with_source(
        self,
        raw_name: str,
        source: NameComponents,
        *,
        result: ParseResult | None = None,
    ) -> CanonicalName | None:
        """Return affirmative Chinese normalization without reviving citation tokens."""
        classification_input = self._chinese_classification_input(raw_name)
        if result is None:
            result = self._normalize_chinese_name(classification_input)
        if not self._is_affirmative_chinese_canonical_input(classification_input, result):
            return None
        canonical = self._canonical_name_from_chinese_result(raw_name, result)
        if canonical is None:
            return None
        canonical_source = canonical.source if classification_input != raw_name else source
        return replace(canonical, source=canonical_source)

    def _is_affirmative_chinese_canonical_input(
        self,
        raw_name: str,
        result: ParseResult,
    ) -> bool:
        """Require source-level Chinese evidence for a canonical override."""
        if not result.success or result.parsed is None:
            return False
        if self._config.cjk_pattern.search(raw_name):
            return True
        normalized_input = self._normalizer.apply(raw_name)
        surname_key = self._normalizer.norm_light(result.parsed.surname)
        given_source_tokens = self._source_given_tokens_for_chinese_result(normalized_input, result.parsed)
        compact_initial_bundle = bool(
            len(given_source_tokens) == 1
            and len(result.parsed.given_tokens) >= 2  # noqa: PLR2004
            and all(
                len("".join(character for character in token if character.isalpha())) == 1 for token in result.parsed.given_tokens
            )
            and self._normalizer.norm(given_source_tokens[0])
            == "".join(self._normalizer.norm(token.rstrip(".")) for token in result.parsed.given_tokens),
        )
        if compact_initial_bundle:
            return True
        if surname_key in INITIAL_ONLY_CROSS_CULTURAL_SURNAMES:
            return (
                bool(given_source_tokens)
                and any(len(token) > 1 for token in given_source_tokens)
                and all(self._is_direct_chinese_given_source_token(token) for token in given_source_tokens)
            )
        source_order = result.parsed_original_order
        preserves_one_unbounded_token = bool(
            source_order is not None
            and source_order.order
            and source_order.order[0] == "surname"
            and not result.parsed.middle_tokens
            and len(given_source_tokens) == 1
            and len(result.parsed.given_tokens) == 1
            and self._normalizer.norm(given_source_tokens[0]) == self._normalizer.norm(result.parsed.given_tokens[0])
            and self._is_unbounded_single_letter_syllable_shape(given_source_tokens[0]),
        )
        if preserves_one_unbounded_token:
            return True
        return bool(given_source_tokens) and all(
            self._is_affirmative_chinese_given_source_token(token) for token in given_source_tokens
        )

    def _source_given_tokens_for_chinese_result(
        self,
        normalized_input: NormalizedInput,
        parsed: ParsedName,
    ) -> list[str]:
        """Remove the selected surname edge from the source Roman tokens."""
        tokens = list(normalized_input.roman_tokens)
        surname_key = "".join(self._normalizer.norm(token) for token in parsed.surname_tokens)
        for size in range(1, len(tokens)):
            if "".join(self._normalizer.norm(token) for token in tokens[:size]) == surname_key:
                return tokens[size:]
            if "".join(self._normalizer.norm(token) for token in tokens[-size:]) == surname_key:
                return tokens[:-size]
        return []

    def _is_affirmative_chinese_given_source_token(self, token: str) -> bool:
        """Reject only speculative fused splits that manufacture a lone letter."""
        if self._is_direct_chinese_given_source_token(token):
            return True

        explicit_parts = [part for part in token.replace("\u2019", "-").replace("'", "-").split("-") if part]
        if len(explicit_parts) > 1:
            return all(
                (len(part) == 1 and part.isalpha())
                or self._data.is_given_name(self._normalizer.norm(part))
                or self._normalizer.is_valid_chinese_phonetics(part)
                for part in explicit_parts
            )

        camel_parts = self._config.camel_case_pattern.findall(token)
        if len(camel_parts) > 1 and "".join(camel_parts) == token:
            return all(
                (len(part) == 1 and part.isalpha())
                or self._data.is_given_name(self._normalizer.norm(part))
                or self._normalizer.is_valid_chinese_phonetics(part)
                for part in camel_parts
            )

        split = StringManipulationUtils.split_concatenated_name(
            token,
            None,
            self._data,
            self._normalizer,
            self._config,
        )
        return bool(split) and all(len(part) > 1 for part in split)

    def _is_direct_chinese_given_source_token(self, token: str) -> bool:
        """Return whether one supplied token is itself a syllable or initial."""
        if len(token) == 1 and token.isalpha():
            return True
        normalized = self._normalizer.norm(token)
        return self._data.is_given_name(normalized) or self._normalizer.is_valid_chinese_phonetics(token)

    def _is_unbounded_single_letter_syllable_shape(
        self,
        token: str,
        *,
        leading_only: bool = False,
    ) -> bool:
        """Return whether A/E plus a full syllable occurs without a boundary."""
        if len(token) <= 2 or not token.isalpha():  # noqa: PLR2004
            return False
        candidates = [token[1:]] if token[0].casefold() in SINGLE_LETTER_PINYIN_SYLLABLES else []
        if not leading_only and token[-1].casefold() in SINGLE_LETTER_PINYIN_SYLLABLES:
            candidates.append(token[:-1])
        return any(
            self._data.is_given_name(self._normalizer.norm(candidate)) or self._normalizer.is_valid_chinese_phonetics(candidate)
            for candidate in candidates
        )

    def _person_name_baseline(self, raw_name: str) -> CanonicalName | None:
        """Return a policy-neutral cleaned person name in flattened input order."""
        if not raw_name or len(raw_name) > self._config.max_name_length:
            return None
        if all(character in string.punctuation + string.whitespace for character in raw_name):
            return None
        self._ensure_initialized()
        if self._non_person_input_service is not None:
            non_person_reason = self._non_person_input_service.failure_reason(raw_name)
            if non_person_reason is not None:
                return None
        normalized = self._person_name_normalizer.normalize_text(raw_name)
        if normalized.outcome is not PersonNameOutcome.PERSON:
            return None
        return normalized.canonical_name

    def _routing_input_order_components(self, raw_name: str) -> NameComponents | None:
        """Clean a flattened input while mechanically retaining endpoint order.

        This is an operational fallback, not a semantic inference: the first
        surviving token is emitted as given, the last as surname, and any
        interior tokens as middle. It is used only after a selected candidate
        has been proven to reverse those endpoints.
        """
        baseline = self._person_name_baseline(raw_name)
        if baseline is None:
            return None
        normalized = baseline.normalized
        name_tokens = list(self._routing_name_tokens(baseline))
        if len(name_tokens) < 2:  # noqa: PLR2004 - an endpoint reversal requires two surviving endpoints
            return None
        given_tokens = (name_tokens[0],)
        middle_tokens = tuple(name_tokens[1:-1])
        surname_tokens = (name_tokens[-1],)
        return NameComponents(
            given_name=given_tokens[0],
            middle_name=" ".join(middle_tokens),
            surname=surname_tokens[0],
            suffix=normalized.suffix,
            given_tokens=given_tokens,
            middle_tokens=middle_tokens,
            surname_tokens=surname_tokens,
            suffix_tokens=normalized.suffix_tokens,
            order=("given", *("middle" for _ in middle_tokens), "surname", *("suffix" for _ in normalized.suffix_tokens)),
        )

    def routing_scalar_resolution(
        self,
        raw_name: str,
    ) -> CanonicalName | HardScalarConstraint | None:
        """Resolve TIMO's scalar candidate and its optional hard constraint.

        This deliberately consumes the cleaned flattened name rather than
        source component labels. Unlike the public compatibility API, an
        evidence service failure is typed and propagated so the terminal
        resolver can record an explicit source-preservation outcome. Ordinary
        non-applicability remains ``None``.
        """
        baseline = self._person_name_baseline(raw_name)
        if baseline is None:
            return None
        if self._ethnicity_service is None:
            return baseline

        routing_surface = self._east_asian_routing_surface(raw_name, baseline)
        resolution = self._east_asian_name_order.infer_resolution(
            routing_surface,
            japanese_probability=self._ethnicity_service.japanese_probability,
        )

        if isinstance(resolution, EastAsianNameOrderPreservation):
            return PreserveBaseline(canonical_name=baseline, evidence_reason=resolution.reason)

        decision = resolution
        if self._korean_compact_split_yields_to_chinese(raw_name, decision):
            decision = None

        if decision is not None:
            routed = self._canonical_name_from_order_decision(baseline, decision)
            if east_asian_evidence_resolution_reason(decision.reason) is not None:
                if routed is None:
                    message = f"hard scalar assignment could not be materialized for {raw_name!r}"
                    raise HardScalarMaterializationFailure(message)
                return ApplyAssignment(canonical_name=routed, evidence_reason=decision.reason)
            return routed or baseline

        return baseline

    def routing_reorder_veto(
        self,
        raw_name: str,
        selected: NameComponents,
        *,
        paper_names: list[str],
        focal_index: int,
    ) -> tuple[NameComponents, ResolutionReason | None]:
        """Apply candidate-aware reorder vetoes to selected components."""
        reason = self._east_asian_name_order.reorder_conflict_reason(
            raw_name,
            selected,
            paper_names=paper_names,
            focal_index=focal_index,
        )
        if reason is None:
            return selected, None
        cleaned_input = self._routing_input_order_components(raw_name)
        if cleaned_input is None:
            message = f"reorder veto could not materialize cleaned input order for {raw_name!r}"
            raise RuntimeError(message)
        return cleaned_input, reason

    def _infer_east_asian_name_order_decision(
        self,
        routing_surface: str,
        *,
        legacy_raw_name: str,
    ) -> EastAsianNameOrderDecision | None:
        """Return one canonical order decision while surfacing evidence failure."""
        if self._ethnicity_service is None:
            return None
        try:
            resolution = self._east_asian_name_order.infer_resolution(
                routing_surface,
                japanese_probability=self._ethnicity_service.japanese_probability,
            )
        except EvidenceFailure as error:
            LOGGER.warning(
                "East Asian name-order routing abstained after classifier failure for %r: %s",
                legacy_raw_name,
                error,
            )
            return None
        decision = resolution if isinstance(resolution, EastAsianNameOrderDecision) else None
        if decision is None or self._korean_compact_split_yields_to_chinese(legacy_raw_name, decision):
            return None
        return decision

    def _korean_compact_split_yields_to_chinese(
        self,
        raw_name: str,
        decision: EastAsianNameOrderDecision | None,
    ) -> bool:
        """A unique Korean compact split yields to a successful Chinese parse."""
        return (
            decision is not None
            and decision.reason is EastAsianEvidenceReason.KOREAN_COMPACT_GIVEN_UNIQUE_SPLIT
            and self._normalize_chinese_name(raw_name).success
        )

    def _routed_source_components(
        self,
        baseline: CanonicalName,
        decision: EastAsianNameOrderDecision,
    ) -> NameComponents:
        """Relabel or split matched routing occurrences while preserving source lineage."""
        decision_source = decision.source_components()
        fallback = replace(
            decision_source,
            suffix=baseline.source.suffix,
            suffix_tokens=baseline.source.suffix_tokens,
            order=decision_source.order + ("suffix",) * len(baseline.source.suffix_tokens),
        )

        routing_tokens = self._routing_name_tokens(baseline)
        decision_tokens = self._ordered_component_tokens(decision_source)
        decision_occurrences = list(zip(decision_source.order, decision_tokens, strict=True))
        routed_groups: list[tuple[tuple[str, str], ...]] = []
        decision_index = 0
        for routing_token in routing_tokens:
            routing_key = self._component_token_key(routing_token)
            if not routing_key:
                return fallback
            group_start = decision_index
            joined_key = ""
            while decision_index < len(decision_occurrences):
                joined_key += self._component_token_key(decision_occurrences[decision_index][1])
                decision_index += 1
                if joined_key == routing_key:
                    break
                if not routing_key.startswith(joined_key):
                    return fallback
            if joined_key != routing_key:
                return fallback
            routed_groups.append(tuple(decision_occurrences[group_start:decision_index]))
        if decision_index != len(decision_occurrences):
            return fallback

        source_tokens = self._ordered_component_tokens(baseline.source)
        assigned = list(zip(baseline.source.order, source_tokens, strict=True))
        source_index = 0
        for routing_token, routed_group in zip(routing_tokens, routed_groups, strict=True):
            routing_key = self._component_token_key(routing_token)
            while source_index < len(assigned) and self._component_token_key(assigned[source_index][1]) != routing_key:
                source_index += 1
            if source_index == len(assigned):
                return fallback
            _source_role, source_token = assigned[source_index]
            replacement = routed_group if len(routed_group) > 1 else ((routed_group[0][0], source_token),)
            assigned[source_index : source_index + 1] = replacement
            source_index += len(replacement)

        tokens_by_role = {
            role: tuple(token for assigned_role, token in assigned if assigned_role == role)
            for role in ("given", "middle", "surname", "suffix")
        }
        return NameComponents(
            given_name=" ".join(tokens_by_role["given"]),
            middle_name=" ".join(tokens_by_role["middle"]),
            surname=" ".join(tokens_by_role["surname"]),
            suffix=" ".join(tokens_by_role["suffix"]),
            given_tokens=tokens_by_role["given"],
            middle_tokens=tokens_by_role["middle"],
            surname_tokens=tokens_by_role["surname"],
            suffix_tokens=tokens_by_role["suffix"],
            order=tuple(role for role, _token in assigned),
        )

    def _canonical_name_from_order_decision(
        self,
        baseline: CanonicalName,
        decision: EastAsianNameOrderDecision,
    ) -> CanonicalName | None:
        """Normalize routed components while retaining original source-role lineage."""
        routed = self._person_name_normalizer.normalize_components(
            first_name=decision.first_name,
            middle_name=decision.middle_name,
            last_name=decision.last_name,
            suffix=baseline.normalized.suffix,
        )
        if routed.outcome is not PersonNameOutcome.PERSON or routed.canonical_name is None:
            # Routed components failed to re-normalize (e.g. a stray leading-hyphen/apostrophe
            # token such as "Shin -Ichi" or "O -P Sairanen" yields an empty component). Abstain
            # from the re-ordering and let the caller keep the baseline canonical name, rather
            # than raising and crashing the whole normalize_name() call on one bad input.
            LOGGER.warning(
                "East Asian route %r produced invalid components; keeping baseline order",
                decision.reason,
            )
            return None
        canonical = replace(
            routed.canonical_name,
            source_text=baseline.source_text,
            source=self._routed_source_components(baseline, decision),
        )
        if decision.reason is EastAsianEvidenceReason.IDENTITY_BACKED_EXACT_FULL_SURFACE and len(decision.given_tokens) > 1:
            normalized = canonical.normalized
            normalized = replace(
                normalized,
                given_name="-".join(normalized.given_tokens),
            )
            canonical = replace(
                canonical,
                text=" ".join(
                    part
                    for part in (
                        normalized.given_name,
                        normalized.middle_name,
                        normalized.surname,
                        normalized.suffix,
                    )
                    if part
                ),
                normalized=normalized,
            )
        return canonical

    def normalize_person_name_components(
        self,
        *,
        first_name: str | None = None,
        middle_name: str | None = None,
        last_name: str | None = None,
        suffix: str | None = None,
    ) -> CanonicalName | None:
        """Normalize structured fields, including conservative family-first routing.

        The normalized assignment may follow high-confidence East Asian order
        evidence, while ``source`` retains the caller-provided roles and order.
        """
        normalized = self._person_name_normalizer.normalize_components(
            first_name=first_name,
            middle_name=middle_name,
            last_name=last_name,
            suffix=suffix,
        )
        if normalized.outcome is not PersonNameOutcome.PERSON or normalized.canonical_name is None:
            return None
        baseline = normalized.canonical_name
        selected = self._canonical_person_name_from_baseline(baseline.source_text, baseline)
        return replace(selected, source=baseline.source)

    def _canonical_name_from_iteration_mark(
        self,
        raw_name: str,
    ) -> CanonicalName | None:
        """Return a source-script canonical name for one proven marked surname."""
        if self._ethnicity_service is None:
            return None
        try:
            decision = self._east_asian_name_order.infer_iteration_mark(
                raw_name,
                japanese_probability=self._ethnicity_service.japanese_probability,
            )
        except EvidenceFailure as error:
            LOGGER.warning(
                "Japanese iteration-mark routing abstained after classifier failure for %r: %s",
                raw_name,
                error,
            )
            return None
        if decision is None:
            return None
        if self._non_person_input_service is not None and self._non_person_input_service.failure_reason(raw_name) is not None:
            return None
        normalized = self._person_name_normalizer.normalize_text(raw_name)
        if normalized.outcome is not PersonNameOutcome.PERSON or normalized.canonical_name is None:
            return None
        return self._canonical_name_from_order_decision(
            normalized.canonical_name,
            decision,
        )

    def _attach_canonical_name(self, raw_name: str, result: ParseResult) -> ParseResult:
        """Attach canonical metadata without changing legacy recognition fields."""
        if result.canonical_name is not None:
            return result
        canonical_name = self._canonical_name_for_result(raw_name, result)
        if canonical_name is None:
            return result
        return replace(result, canonical_name=canonical_name)

    def _canonical_name_for_result(
        self,
        raw_name: str,
        result: ParseResult,
    ) -> CanonicalName | None:
        """Return the canonical sidecar used by scalar normalization."""
        if result.canonical_name is not None:
            return result.canonical_name
        canonical_name = self._canonical_name_from_iteration_mark(raw_name)
        if canonical_name is None:
            canonical_name = self._canonical_name_from_chinese_result(raw_name, result)
        if canonical_name is None:
            canonical_name = self._normalize_person_name_with_chinese_result(raw_name, result)
        return canonical_name

    def normalize_name(self, raw_name: str) -> ParseResult:
        """Detect/normalize Chinese names and surface canonical data for people.

        Legacy ``success``, ``result``, ``parsed``, and error semantics remain
        Chinese-specific. ``canonical_name`` is the all-person representation.
        """
        result = self._normalize_chinese_name(raw_name)
        return self._attach_canonical_name(raw_name, result)

    # Backwards compatibility alias
    def is_chinese_name(self, raw_name: str) -> ParseResult:  # pragma: no cover - thin wrapper
        """Deprecated: use normalize_name(). Maintained for compatibility."""
        return self.normalize_name(raw_name)

    def analyze_name_batch(
        self,
        names: list[str],
        format_threshold: float = 0.55,
        minimum_batch_size: int = 2,
    ) -> BatchParseResult:
        """
        Analyze a batch of names with format pattern detection.

        This method processes multiple names together, detects the dominant
        formatting pattern (surname-first vs given-first), and applies it
        consistently to improve accuracy for ambiguous cases.

        Args:
            names: List of raw name strings to analyze
            format_threshold: Minimum percentage (0.0-1.0) required for format detection
            minimum_batch_size: Minimum number of names required for batch processing

        Returns:
            BatchParseResult containing individual results, format pattern, and improvements

        Example:
            # Academic author list (surname-first pattern)
            names = ["Zhang Wei", "Li Ming", "Bei Yu", "Wang Xiaoli"]
            result = detector.analyze_name_batch(names)
            # "Bei Yu" will be correctly parsed as "Bei Yu" due to batch context
        """
        _validate_batch_policy(format_threshold, minimum_batch_size)
        self._ensure_initialized()
        if self._batch_analysis_service is None:
            message = "batch analysis service is not initialized"
            raise RuntimeError(message)
        batch_result = self._batch_analysis_service.analyze_name_batch(
            names,
            self._normalizer,
            self._formatting_service,
            BatchAnalysisOptions(
                minimum_batch_size=minimum_batch_size,
                format_threshold=format_threshold,
            ),
        )
        return self._attach_batch_canonical_names(batch_result)

    def _attach_batch_canonical_names(self, batch_result: BatchParseResult) -> BatchParseResult:
        """Attach canonical sidecars to one already-materialized focal batch."""
        return replace(
            batch_result,
            results=[
                self._attach_canonical_name(name, result)
                for name, result in zip(batch_result.names, batch_result.results, strict=True)
            ],
        )

    def _analyze_related_name_batches(
        self,
        pp_names: list[str],
        vys_pool_names: list[str] | None,
        format_threshold: float = 0.55,
        minimum_batch_size: int = 2,
        prepared_cache=None,
    ) -> RelatedBatchParseResult:
        """Analyze one lean PP/VYS work item without duplicating focal preparation."""
        _validate_batch_policy(format_threshold, minimum_batch_size)
        self._ensure_initialized()
        if self._batch_analysis_service is None:
            message = "batch analysis service is not initialized"
            raise RuntimeError(message)
        return self._batch_analysis_service.analyze_related_name_batches(
            pp_names,
            vys_pool_names,
            self._normalizer,
            self._formatting_service,
            BatchAnalysisOptions(
                minimum_batch_size=minimum_batch_size,
                format_threshold=format_threshold,
            ),
            prepared_cache,
        )

    def detect_batch_format(
        self,
        names: list[str],
        format_threshold: float = 0.55,
    ) -> BatchFormatPattern:
        """
        Detect the format pattern of a batch without full processing.

        This is useful for understanding the formatting consistency of a
        name list before deciding whether to apply batch processing.

        Args:
            names: List of raw name strings to analyze
            format_threshold: Minimum percentage (0.0-1.0) required for format detection

        Returns:
            BatchFormatPattern indicating the dominant format, count confidence,
            and decision confidence used for batch application

        Example:
            pattern = detector.detect_batch_format(["Zhang Wei", "Li Ming", "Wang Xiaoli"])
            if pattern.threshold_met:
                print(f"Detected {pattern.dominant_format} with {pattern.decision_confidence:.1%} confidence")
        """
        _validate_batch_policy(format_threshold)
        self._ensure_initialized()

        if self._batch_analysis_service is None:
            # Return a fallback pattern indicating mixed format
            return BatchFormatPattern(
                dominant_format=NameFormat.MIXED,
                confidence=0.0,
                surname_first_count=0,
                given_first_count=0,
                total_count=len(names),
                threshold_met=False,
            )

        return self._batch_analysis_service.detect_batch_format(
            names,
            self._normalizer,
            format_threshold=format_threshold,
        )

    def process_name_batch(
        self,
        names: list[str],
        format_threshold: float = 0.55,
        minimum_batch_size: int = 2,
    ) -> list[ParseResult]:
        """
        Process a batch of names and return just the parse results.

        This is a convenience method that returns only the ParseResult list
        from batch analysis, similar to calling normalize_name() on each name
        but with batch format detection applied.

        Args:
            names: List of raw name strings to process
            format_threshold: Minimum percentage (0.0-1.0) required for format detection
            minimum_batch_size: Minimum number of names required for batch processing

        Returns:
            List of ParseResult objects, one for each input name

        Example:
            names = ["Zhang Wei", "Li Ming", "Bei Yu"]
            results = detector.process_name_batch(names)
            for result in results:
                if result.success:
                    print(f"Formatted: {result.result}")
        """
        batch_result = self.analyze_name_batch(names, format_threshold, minimum_batch_size)
        return batch_result.results

    def normalize_names(
        self,
        names: list[str],
        *,
        parallel: ParallelMode = "auto",
        min_parallel_names: int | None = None,
        max_workers: int | None = None,
        chunk_size: int = 64,
        mp_start_method: str = "auto",
    ) -> list[ParseResult]:
        """
        Normalize independent names with automatic multiprocessing selection.

        This wrapper has per-name `normalize_name()` semantics. With
        `parallel="auto"`, it uses the local detector for small inputs and a
        persistent process pool for larger inputs. Use `parallel="always"` to
        force the process-pool path or `parallel="never"` for deterministic
        single-process execution.
        """
        self._ensure_initialized()
        if _should_use_multiprocessing(
            item_count=len(names),
            parallel=parallel,
            auto_threshold=min_parallel_names,
            default_auto_threshold=AUTO_MULTIPROCESS_MIN_NAMES,
            linux_auto_threshold=LINUX_AUTO_MULTIPROCESS_MIN_NAMES,
            max_workers=max_workers,
        ):
            with self.create_persistent_multiprocess_pool(
                max_workers=max_workers,
                chunk_size=chunk_size,
                mp_start_method=_auto_start_method(mp_start_method),
            ) as pool:
                return pool.normalize_names(names)
        return [self.normalize_name(name) for name in names]

    def analyze_name_batches(
        self,
        batches: list[list[str]],
        *,
        parallel: ParallelMode = "auto",
        min_parallel_batches: int | None = None,
        format_threshold: float = 0.55,
        minimum_batch_size: int = 2,
        max_workers: int | None = None,
        chunk_size: int = 64,
        mp_start_method: str = "auto",
    ) -> list[BatchParseResult]:
        """
        Analyze independent name batches with automatic multiprocessing selection.

        Each inner list is one batch-context boundary. With `parallel="auto"`,
        small batch lists run in-process and large batch lists use a persistent
        process pool. Output order matches the submitted batch order.
        """
        _validate_batch_policy(format_threshold, minimum_batch_size)
        self._ensure_initialized()
        if _should_use_multiprocessing(
            item_count=len(batches),
            parallel=parallel,
            auto_threshold=min_parallel_batches,
            default_auto_threshold=AUTO_MULTIPROCESS_MIN_BATCHES,
            linux_auto_threshold=LINUX_AUTO_MULTIPROCESS_MIN_BATCHES,
            max_workers=max_workers,
        ):
            with self.create_persistent_multiprocess_pool(
                max_workers=max_workers,
                chunk_size=chunk_size,
                mp_start_method=_auto_start_method(mp_start_method),
            ) as pool:
                return pool.analyze_name_batches(
                    batches,
                    format_threshold=format_threshold,
                    minimum_batch_size=minimum_batch_size,
                )
        return [
            self.analyze_name_batch(
                batch,
                format_threshold=format_threshold,
                minimum_batch_size=minimum_batch_size,
            )
            for batch in batches
        ]

    def _analyze_related_batch_requests(
        self,
        requests: list[tuple[list[str], list[str] | None]],
        *,
        parallel: ParallelMode = "auto",
        min_parallel_batches: int | None = None,
        format_threshold: float = 0.55,
        minimum_batch_size: int = 2,
        max_workers: int | None = None,
        chunk_size: int = 64,
        mp_start_method: str = "auto",
    ) -> list[RelatedBatchParseResult]:
        """Analyze PP/VYS pairs as related worker units while preserving request order."""
        _validate_batch_policy(format_threshold, minimum_batch_size)
        self._ensure_initialized()
        if _should_use_multiprocessing(
            item_count=len(requests),
            parallel=parallel,
            auto_threshold=min_parallel_batches,
            default_auto_threshold=AUTO_MULTIPROCESS_MIN_BATCHES,
            linux_auto_threshold=LINUX_AUTO_MULTIPROCESS_MIN_BATCHES,
            max_workers=max_workers,
        ):
            with self.create_persistent_multiprocess_pool(
                max_workers=max_workers,
                chunk_size=chunk_size,
                mp_start_method=_auto_start_method(mp_start_method),
            ) as pool:
                return pool._analyze_related_batch_requests(
                    requests,
                    format_threshold=format_threshold,
                    minimum_batch_size=minimum_batch_size,
                )
        prepared_cache = {}
        return [
            self._analyze_related_name_batches(
                pp_names,
                vys_pool_names,
                format_threshold=format_threshold,
                minimum_batch_size=minimum_batch_size,
                prepared_cache=prepared_cache,
            )
            for pp_names, vys_pool_names in requests
        ]

    def process_name_batches(
        self,
        batches: list[list[str]],
        *,
        parallel: ParallelMode = "auto",
        min_parallel_batches: int | None = None,
        format_threshold: float = 0.55,
        minimum_batch_size: int = 2,
        max_workers: int | None = None,
        chunk_size: int = 64,
        mp_start_method: str = "auto",
    ) -> list[list[ParseResult]]:
        """
        Process independent name batches with automatic multiprocessing selection.

        This is the high-level API for workloads such as many paper author
        lists. Each inner list gets normal `process_name_batch()` semantics.
        """
        batch_results = self.analyze_name_batches(
            batches,
            parallel=parallel,
            min_parallel_batches=min_parallel_batches,
            format_threshold=format_threshold,
            minimum_batch_size=minimum_batch_size,
            max_workers=max_workers,
            chunk_size=chunk_size,
            mp_start_method=mp_start_method,
        )
        return [batch_result.results for batch_result in batch_results]

    def create_persistent_multiprocess_pool(
        self,
        *,
        max_workers: int | None = None,
        chunk_size: int = 64,
        mp_start_method: str = "spawn",
    ) -> PersistentMultiprocessNormalizer:
        """
        Create a persistent multi-process pool for repeated processing calls.

        Notes:
        - Uses one detector instance per worker process.
        - Use `normalize_names()` for independent per-name parsing.
        - Use `process_name_batches()` for many independent author lists that
          each need batch-format correction.
        - For Windows/macOS scripts, call this behind an
          `if __name__ == "__main__":` guard.
        """
        self._ensure_initialized()
        return PersistentMultiprocessNormalizer(
            max_workers=max_workers,
            chunk_size=chunk_size,
            mp_start_method=mp_start_method,
            detector_config=self._config,
            detector_weights=self._weights,
        )

    def process_name_batch_multiprocess(
        self,
        names: list[str],
        *,
        max_workers: int | None = None,
        chunk_size: int = 64,
        mp_start_method: str = "spawn",
    ) -> list[ParseResult]:
        """
        Process one author list in a temporary process pool.

        This method has `process_name_batch()` semantics, including batch-format
        correction. Use `normalize_names()` for independent per-name parsing, or
        `create_persistent_multiprocess_pool()` to reuse workers across repeated
        calls.
        """
        self._ensure_initialized()
        return self.process_name_batches(
            [names],
            parallel="always",
            min_parallel_batches=1,
            max_workers=max_workers,
            chunk_size=chunk_size,
            mp_start_method=mp_start_method,
        )[0]
