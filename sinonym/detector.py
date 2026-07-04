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
import platform
import string
from typing import Literal

from sinonym.chinese_names_data import COMPOUND_VARIANTS
from sinonym.coretypes import (
    BatchFormatPattern,
    BatchParseResult,
    IndividualAnalysis,
    NameFormat,
    NameOrderEvidence,
)
from sinonym.coretypes.results import ParsedName
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
)
from sinonym.services.order_metadata import original_component_order
from sinonym.services.process_pool import PersistentMultiprocessNormalizer, normalize_names_multiprocess

ParallelMode = Literal["auto", "never", "always"]
AUTO_MULTIPROCESS_MIN_NAMES = 25000
AUTO_MULTIPROCESS_MIN_BATCHES = 4000
LINUX_AUTO_MULTIPROCESS_MIN_NAMES = 10000
LINUX_AUTO_MULTIPROCESS_MIN_BATCHES = 2000
BILINGUAL_SURNAME_STRENGTH_RATIO_MIN = 5.0
BILINGUAL_ROMAN_HAN_SOURCE_ORDER_RATIO_MAX = 12.0
BILINGUAL_ENDPOINT_PAIR_COUNT = 2
SPACED_HAN_PREFIX_SURNAME_RATIO_MIN = 5.0
CAMEL_CASE_LAST_SURNAME_RATIO_MIN = 5.0
CURATED_COMPOUND_SURNAME_FORMS = frozenset((*COMPOUND_VARIANTS.keys(), *COMPOUND_VARIANTS.values()))
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
    resolved_threshold = auto_threshold
    if resolved_threshold is None:
        resolved_threshold = linux_auto_threshold if platform.system() == "Linux" else default_auto_threshold
    if resolved_threshold < 1:
        message = "auto multiprocessing threshold must be >= 1"
        raise ValueError(message)
    if item_count == 0 or parallel == "never":
        return False
    if parallel == "always":
        return True
    return max_workers != 1 and item_count >= resolved_threshold


def _auto_start_method(mp_start_method: str) -> str:
    """Resolve the high-level auto start method."""
    if mp_start_method != "auto":
        return mp_start_method
    return "fork" if platform.system() == "Linux" else "spawn"


# ════════════════════════════════════════════════════════════════════════════════
# MAIN CHINESE NAME DETECTOR CLASS
# ════════════════════════════════════════════════════════════════════════════════


class ChineseNameDetector:
    """Main Chinese name detection and normalization service."""

    def __init__(self, config: ChineseNameConfig | None = None, weights: list[float] | None = None):
        self._config = config or ChineseNameConfig.create_default()
        self._cache_service = PinyinCacheService(self._config)
        self._normalizer = NormalizationService(self._config, self._cache_service)
        self._data_service = DataInitializationService(self._config, self._cache_service, self._normalizer)
        self._data: NameDataStructures | None = None
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
            self._data = self._data_service.initialize_data_structures()
            # Inject data context into normalizer after initialization
            self._normalizer.set_data_context(self._data)
            # Initialize services
            self._initialize_services()
        except Exception as e:
            logging.warning(f"Failed to initialize at construction: {e}. Will initialize lazily.")

    def _initialize_services(self) -> None:
        """Initialize service instances with data context."""
        if self._data is not None:
            # Create shared context to reduce dependency injection complexity
            context = ServiceContext(self._config, self._normalizer, self._data)

            self._ethnicity_service = EthnicityClassificationService(context)
            self._parsing_service = NameParsingService(context, weights=self._weights)
            self._formatting_service = NameFormattingService(context)
            self._non_person_input_service = NonPersonInputDetectionService(self._config, self._normalizer, self._data)
            self._batch_analysis_service = BatchAnalysisService(
                self._parsing_service,
                ethnicity_service=self._ethnicity_service,
                dependencies=BatchAnalysisDependencies(
                    min_tokens_required=self._config.min_tokens_required,
                    individual_parser=self.normalize_name,
                    input_failure=self._initial_input_failure,
                ),
            )

    def _ensure_initialized(self) -> None:
        """Ensure data is initialized (lazy initialization)."""
        if self._data is None:
            self._data = self._data_service.initialize_data_structures()
            # Inject data context into normalizer
            self._normalizer.set_data_context(self._data)
            # Initialize services
            self._initialize_services()

    def _is_surname_group(self, tokens: list[str], normalized_cache: dict[str, str]) -> bool:
        """Return whether a romanized Han token group is a surname component."""
        if len(tokens) == 1:
            token = tokens[0]
            normalized = self._normalizer.get_normalized(token, normalized_cache)
            return self._data.is_surname(token, normalized)

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

    def _surname_group_strength(self, pinyin_tokens: list[str] | tuple[str, ...], han_group: str = "") -> float:
        """Return the strongest surname-frequency signal for a Han/pinyin group."""
        if not pinyin_tokens:
            return 0.0

        data = self._data
        assert data is not None
        keys = [han_group] if han_group else []
        if len(pinyin_tokens) == 1 and not han_group:
            keys.append(pinyin_tokens[0])
        elif len(pinyin_tokens) > 1:
            keys.extend((" ".join(pinyin_tokens), "".join(pinyin_tokens)))
        return max((data.get_surname_freq(key) for key in keys if key), default=0.0)

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
    ) -> ParseResult:
        """Format parsed components and attach stable structured name fields."""
        formatted_name, given_final, surname_final, surname_str, given_str, middle_tokens = (
            self._formatting_service.format_name_output_with_tokens(
                surname_tokens,
                given_tokens,
                normalized_input.norm_map,
                normalized_input.compound_metadata,
                allow_surname_like_given_split=self._allows_surname_like_given_split(normalized_input),
            )
        )
        parsed = ParsedName(
            surname=surname_str,
            given_name=given_str,
            surname_tokens=surname_final,
            given_tokens=given_final,
            middle_name=" ".join(middle_tokens) if middle_tokens else "",
            middle_tokens=middle_tokens,
            order=["given", "middle", "surname"],
        )
        parsed_original_order = ParsedName(
            surname=surname_str,
            given_name=given_str,
            surname_tokens=surname_final,
            given_tokens=given_final,
            middle_name=" ".join(middle_tokens) if middle_tokens else "",
            middle_tokens=middle_tokens,
            order=original_component_order(
                NameFormat.GIVEN_FIRST if original_order and original_order[0] == "given" else NameFormat.SURNAME_FIRST,
                given_tokens,
                middle_tokens,
            ),
        )
        return ParseResult.success_with_name(
            formatted_name,
            parsed=parsed,
            parsed_original_order=parsed_original_order,
        )

    def _allows_surname_like_given_split(self, normalized_input: NormalizedInput) -> bool:
        """Return whether surname-like fused given tokens may be gold-split."""
        return not any(
            self._config.cjk_pattern.search(char) for token in normalized_input.tokens for char in token
        )

    def _normalize_camel_case_pair(self, normalized_input: NormalizedInput) -> ParseResult | None:
        """Parse a whole-input camelCase pair using surname-first provenance."""
        tokens = list(normalized_input.roman_tokens)
        if len(tokens) != 2 or self._is_compound_surname_group(tokens, normalized_input.norm_map):
            return None

        first, last = tokens
        first_key = self._normalizer.norm_light(first)
        last_key = self._normalizer.norm_light(last)
        first_is_surname = self._data.is_surname(first, first_key)
        last_is_surname = self._data.is_surname(last, last_key)
        if not first_is_surname and not last_is_surname:
            return None

        first_freq = self._data.get_surname_freq(first_key)
        last_freq = self._data.get_surname_freq(last_key)
        last_wins = last_is_surname and (
            not first_is_surname or last.isupper() or last_freq >= CAMEL_CASE_LAST_SURNAME_RATIO_MIN * first_freq
        )
        if last_wins:
            surname_tokens, given_tokens = [last], [first]
            original_order = ["given", "surname"]
        else:
            surname_tokens, given_tokens = [first], [last]
            original_order = ["surname", "given"]

        try:
            return self._format_parse_result(surname_tokens, given_tokens, normalized_input, original_order)
        except ValueError as e:
            return ParseResult.failure(str(e))

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

        try:
            return self._format_parse_result(surname_tokens, given_tokens, normalized_input, original_order)
        except ValueError as e:
            return ParseResult.failure(str(e))

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
        try:
            return self._format_parse_result(surname_tokens, given_tokens, normalized_input, ["given", "surname"])
        except ValueError as e:
            return ParseResult.failure(str(e))

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
        han_freq = self._data.get_surname_freq(pair.han_token)
        if len(pair.han_pinyin) == 1:
            return han_freq

        pinyin_key = " ".join(pair.han_pinyin)
        compact_key = "".join(pair.han_pinyin)
        return max(
            han_freq,
            self._data.get_surname_freq(pinyin_key),
            self._data.get_surname_freq(compact_key),
        )

    def _normalize_compact_han_roman_name(self, normalized_input: NormalizedInput) -> ParseResult | None:
        """Parse compact Han names followed by an exact Roman transliteration."""
        compact_components = self._compact_han_roman_components(normalized_input)
        if compact_components is None:
            return None
        surname_tokens, given_tokens, original_order = compact_components

        try:
            return self._format_parse_result(surname_tokens, given_tokens, normalized_input, original_order)
        except ValueError as e:
            return ParseResult.failure(str(e))

    def _compact_han_roman_components(
        self,
        normalized_input: NormalizedInput,
    ) -> tuple[list[str], list[str], list[str]] | None:
        """Return parsed components for exact compact Han/Roman transliterations."""
        han_groups = [token for token in normalized_input.tokens if self._is_source_han_token(token)]
        roman_tokens = [
            clean_token
            for token in normalized_input.tokens
            if self._is_source_roman_token(token)
            for clean_token in [self._clean_source_roman_token(token)]
            if clean_token
        ]

        source_token_count = len(han_groups) + len(roman_tokens)
        if len(han_groups) == 1 and source_token_count == len(normalized_input.tokens) and roman_tokens:
            han_group = han_groups[0]
            han_pinyin = tuple(self._cache_service.han_to_pinyin_fast(han_group))
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

    def _is_source_han_token(self, token: str) -> bool:
        """Return whether a source token is entirely CJK."""
        return bool(token) and all(self._config.cjk_pattern.search(char) for char in token)

    def _is_source_roman_token(self, token: str) -> bool:
        """Return whether a source token has Roman letters and no CJK characters."""
        return bool(
            token and self._config.ascii_alpha_pattern.search(token) and not self._config.cjk_pattern.search(token),
        )

    def _clean_source_roman_token(self, token: str) -> str:
        """Clean a source Roman token while preserving source capitalization."""
        return self._config.clean_roman_pattern.sub("", token)

    def _roman_tokens_match_han_pinyin(self, roman_tokens: list[str], han_pinyin: tuple[str, ...]) -> bool:
        """Return whether Roman source tokens exactly transliterate the Han pinyin."""
        roman_joined = "".join(self._normalizer.norm(token) for token in roman_tokens)
        han_joined = "".join(self._normalizer.norm(token) for token in han_pinyin)
        return roman_joined == han_joined

    def _han_surname_prefix_length(self, han_group: str, han_pinyin: tuple[str, ...]) -> int:
        """Return the Han surname prefix length in pinyin tokens."""
        if len(han_pinyin) >= self._config.min_tokens_required:
            first_two_pinyin = list(han_pinyin[:2])
            first_two_han = han_group[:2]
            if self._data.get_surname_freq(first_two_han) > 0 or self._is_compound_surname_group(first_two_pinyin, {}):
                return 2

        first_han = han_group[0]
        if self._data.get_surname_freq(first_han) > 0:
            return 1
        return 0

    def _han_surname_suffix_length(self, han_group: str, han_pinyin: tuple[str, ...]) -> int:
        """Return the Han surname suffix length in pinyin tokens."""
        if len(han_pinyin) >= self._config.min_tokens_required:
            last_two_pinyin = list(han_pinyin[-2:])
            last_two_han = han_group[-2:]
            if self._data.get_surname_freq(last_two_han) > 0 or self._is_compound_surname_group(last_two_pinyin, {}):
                return 2

        last_han = han_group[-1]
        if self._data.get_surname_freq(last_han) > 0:
            return 1
        return 0

    def _split_roman_tokens_for_han_prefix(
        self,
        roman_tokens: list[str],
        han_pinyin: tuple[str, ...],
        prefix_length: int,
    ) -> tuple[list[str], list[str]] | None:
        """Split Roman tokens at the boundary matching a Han pinyin prefix."""
        prefix_target = "".join(self._normalizer.norm(token) for token in han_pinyin[:prefix_length])
        current = ""
        for index, token in enumerate(roman_tokens, start=1):
            current += self._normalizer.norm(token)
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
        suffix_target = "".join(self._normalizer.norm(token) for token in han_pinyin[-suffix_length:])
        current = ""
        for index in range(len(roman_tokens) - 1, -1, -1):
            current = self._normalizer.norm(roman_tokens[index]) + current
            if current == suffix_target:
                return roman_tokens[:index], roman_tokens[index:]
            if not suffix_target.endswith(current):
                return None
        return None

    def _normalize_spaced_all_chinese_name(self, normalized_input: NormalizedInput) -> ParseResult | None:
        """Parse all-Han names whose whitespace already separates name components."""
        if (
            len(normalized_input.tokens) != self._config.min_tokens_required
            or len(normalized_input.roman_tokens) <= self._config.min_tokens_required
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
            last_compound_surname_wins = last_is_compound_surname and (
                self._is_curated_compound_surname_group(last_group, normalized_input.norm_map)
                or self._surname_group_strength(
                    last_group,
                    normalized_input.tokens[1],
                )
                >= self._surname_group_strength(
                    first_group,
                    normalized_input.tokens[0],
                )
                * BILINGUAL_SURNAME_STRENGTH_RATIO_MIN
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

        try:
            return self._format_parse_result(surname_tokens, given_tokens, normalized_input, original_order)
        except ValueError as e:
            return ParseResult.failure(str(e))

    def _spaced_han_prefers_prefix_surname(self, first_group: list[str], last_group: list[str]) -> bool:
        """Return whether noisy spacing likely split a surname-first Han name's given name."""
        if not first_group or not last_group:
            return False

        first_freq = self._data.get_surname_freq(first_group[0])
        last_freq = self._data.get_surname_freq(last_group[0])
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

    def normalize_name(self, raw_name: str) -> ParseResult:
        """
        Main API method: Detect if a name is Chinese and normalize it.

        Returns ParseResult with:
        - success=True, result=formatted_name if Chinese name detected
        - success=False, error_message=reason if not Chinese name
        """
        initial_failure = self._initial_input_failure(raw_name)
        if initial_failure is not None:
            return initial_failure

        # Use new normalization service for cleaner pipeline
        normalized_input = self._normalizer.apply(raw_name)

        if len(normalized_input.roman_tokens) < self._config.min_tokens_required:
            return ParseResult.failure(f"needs at least {self._config.min_tokens_required} Roman tokens")

        # Check if this is an all-Chinese input first
        is_all_chinese = self._normalizer._text_preprocessor.is_all_chinese_input(raw_name)

        # Check for non-Chinese ethnicity using normalized tokens (consistent for all inputs)
        non_chinese_result = self._ethnicity_service.classify_ethnicity(
            normalized_input.roman_tokens,
            normalized_input.norm_map,
            raw_name,
        )

        if non_chinese_result.success is False:
            return non_chinese_result

        aligned_bilingual_result = self._normalize_aligned_bilingual_name(normalized_input)
        if aligned_bilingual_result is not None:
            return aligned_bilingual_result

        compact_han_roman_result = self._normalize_compact_han_roman_name(normalized_input)
        if compact_han_roman_result is not None:
            return compact_han_roman_result

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
            token1_norm = normalized_input.norm_map.get(token1, self._normalizer.norm(token1))
            token1_is_surname = self._data.is_surname(token1, token1_norm)

            # For 2-character all-Chinese names, use surname-first if token1 is a valid surname
            if token1_is_surname:
                best_result = ([token1], [token2])
            else:
                # Fallback: if token1 is not a surname, try token2 as surname (less common but possible)
                token2_norm = normalized_input.norm_map.get(token2, self._normalizer.norm(token2))
                token2_is_surname = self._data.is_surname(token2, token2_norm)
                if token2_is_surname:
                    best_result = ([token2], [token1])
                else:
                    best_result = None

            if best_result:
                surname_tokens, given_tokens = best_result
                try:
                    formatted_name, given_final, surname_final, surname_str, given_str, middle_tokens = (
                        self._formatting_service.format_name_output_with_tokens(
                            surname_tokens,
                            given_tokens,
                            normalized_input.norm_map,
                            normalized_input.compound_metadata,
                            allow_surname_like_given_split=self._allows_surname_like_given_split(normalized_input),
                        )
                    )
                    parsed = ParsedName(
                        surname=surname_str,
                        given_name=given_str,
                        surname_tokens=surname_final,
                        given_tokens=given_final,
                        middle_name=" ".join(middle_tokens) if middle_tokens else "",
                        middle_tokens=middle_tokens,
                        order=["given", "middle", "surname"],
                    )
                    # Determine original input order and assign components accordingly
                    if token1_is_surname:
                        # Original: surname-first → preserve component labels and annotate order
                        order_list = ["surname", "given"]
                        parsed_original_order = ParsedName(
                            surname=surname_str,
                            given_name=given_str,
                            surname_tokens=surname_final,
                            given_tokens=given_final,
                            middle_name=" ".join(middle_tokens) if middle_tokens else "",
                            middle_tokens=middle_tokens,
                            order=order_list,
                        )
                    else:
                        # Original: given-first (or token2 surname) → keep labels
                        order_list = ["given", "surname"]
                        parsed_original_order = ParsedName(
                            surname=surname_str,
                            given_name=given_str,
                            surname_tokens=surname_final,
                            given_tokens=given_final,
                            middle_name=" ".join(middle_tokens) if middle_tokens else "",
                            middle_tokens=middle_tokens,
                            order=order_list,
                        )
                    return ParseResult.success_with_name(
                        formatted_name,
                        parsed=parsed,
                        parsed_original_order=parsed_original_order,
                    )
                except ValueError as e:
                    return ParseResult.failure(str(e))
        elif is_all_chinese and len(normalized_input.roman_tokens) == 3:
            # For 3-character all-Chinese names: check compound surname vs single surname
            tokens = list(normalized_input.roman_tokens)

            # Try both possibilities and see which one the parsing service accepts
            # Option 1: First two tokens as compound surname + third as given
            compound_parse = self._parsing_service.parse_name_order_tokens(
                tokens,
                normalized_input.norm_map,
                normalized_input.compound_metadata,
            )

            if compound_parse is not None and len(compound_parse[0]) == 2 and len(compound_parse[1]) == 1:
                # Parsing service recognized first two as compound surname
                best_result = (compound_parse[0], compound_parse[1])
            else:
                # Option 2: First token as single surname + last two as given name
                best_result = ([tokens[0]], tokens[1:])

            if best_result:
                surname_tokens, given_tokens = best_result
                try:
                    formatted_name, given_final, surname_final, surname_str, given_str, middle_tokens = (
                        self._formatting_service.format_name_output_with_tokens(
                            surname_tokens,
                            given_tokens,
                            normalized_input.norm_map,
                            normalized_input.compound_metadata,
                            allow_surname_like_given_split=self._allows_surname_like_given_split(normalized_input),
                        )
                    )
                    parsed = ParsedName(
                        surname=surname_str,
                        given_name=given_str,
                        surname_tokens=surname_final,
                        given_tokens=given_final,
                        middle_name=" ".join(middle_tokens) if middle_tokens else "",
                        middle_tokens=middle_tokens,
                        order=["given", "middle", "surname"],
                    )
                    # For 3-character all-Chinese, original order is surname-first
                    # Preserve component labels and annotate only the order.
                    parsed_original_order = ParsedName(
                        surname=surname_str,
                        given_name=given_str,
                        surname_tokens=surname_final,
                        given_tokens=given_final,
                        middle_name=" ".join(middle_tokens) if middle_tokens else "",
                        middle_tokens=middle_tokens,
                        order=["surname", "given"],
                    )
                    return ParseResult.success_with_name(
                        formatted_name,
                        parsed=parsed,
                        parsed_original_order=parsed_original_order,
                    )
                except ValueError as e:
                    return ParseResult.failure(str(e))
        else:
            if normalized_input.from_camel_case_pair:
                camel_result = self._normalize_camel_case_pair(normalized_input)
                if camel_result is not None:
                    return camel_result

            # Evaluate both order hypotheses and pick the best-scoring parse
            original_tokens = list(normalized_input.roman_tokens)
            best_candidate = None

            for order in (normalized_input.roman_tokens, normalized_input.roman_tokens[::-1]):
                order_tokens = list(order)
                parse_result = self._parsing_service.parse_name_order_tokens(
                    order_tokens,
                    normalized_input.norm_map,
                    normalized_input.compound_metadata,
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
                try:
                    formatted_name, given_final, surname_final, surname_str, given_str, middle_tokens = (
                        self._formatting_service.format_name_output_with_tokens(
                            surname_tokens,
                            given_tokens,
                            normalized_input.norm_map,
                            normalized_input.compound_metadata,
                            allow_surname_like_given_split=self._allows_surname_like_given_split(normalized_input),
                        )
                    )
                    parsed = ParsedName(
                        surname=surname_str,
                        given_name=given_str,
                        surname_tokens=surname_final,
                        given_tokens=given_final,
                        middle_name=" ".join(middle_tokens) if middle_tokens else "",
                        middle_tokens=middle_tokens,
                        order=["given", "middle", "surname"],
                    )
                    # Determine original input order relative to detected parse
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

                    original_format = NameFormat.GIVEN_FIRST if original_is_given_first else NameFormat.SURNAME_FIRST
                    order_list = original_component_order(original_format, given_tokens, middle_tokens)
                    parsed_original_order = ParsedName(
                        surname=surname_str,
                        given_name=given_str,
                        surname_tokens=surname_final,
                        given_tokens=given_final,
                        middle_name=" ".join(middle_tokens) if middle_tokens else "",
                        middle_tokens=middle_tokens,
                        order=order_list,
                    )

                    return ParseResult.success_with_name(
                        formatted_name,
                        parsed=parsed,
                        parsed_original_order=parsed_original_order,
                    )
                except ValueError as e:
                    return ParseResult.failure(str(e))

        return ParseResult.failure("name not recognised as Chinese")

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
        self._ensure_initialized()

        if self._batch_analysis_service is None:
            # Fallback to individual processing if batch service not available
            individual_results = [self.normalize_name(name) for name in names]
            return self._create_fallback_batch_result(names, individual_results)

        return self._batch_analysis_service.analyze_name_batch(
            names,
            self._normalizer,
            self._data,
            self._formatting_service,
            BatchAnalysisOptions(
                minimum_batch_size=minimum_batch_size,
                format_threshold=format_threshold,
            ),
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
        self._ensure_initialized()

        if self._batch_analysis_service is None:
            # Return a fallback pattern indicating mixed format
            from sinonym.coretypes import NameFormat

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
            self._data,
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
        Normalize a list of names in a temporary process pool.

        This method has per-name `normalize_name()` semantics and does not apply
        batch-format correction. Use `process_name_batch()` when batch-context
        parsing is required, or `create_persistent_multiprocess_pool()` to reuse
        workers across repeated per-name normalization calls.
        """
        self._ensure_initialized()
        return normalize_names_multiprocess(
            names,
            max_workers=max_workers,
            chunk_size=chunk_size,
            mp_start_method=mp_start_method,
            detector_config=self._config,
            detector_weights=self._weights,
        )

    def _create_fallback_batch_result(
        self,
        names: list[str],
        individual_results: list[ParseResult],
    ) -> BatchParseResult:
        """Create a fallback BatchParseResult when batch analysis is not available."""
        # Create dummy format pattern
        format_pattern = BatchFormatPattern(
            dominant_format=NameFormat.MIXED,
            confidence=0.0,
            surname_first_count=0,
            given_first_count=0,
            total_count=len(names),
            threshold_met=False,
        )

        # Create dummy individual analyses
        individual_analyses = []
        for name in names:
            individual_analyses.append(
                IndividualAnalysis(
                    raw_name=name,
                    candidates=[],
                    best_candidate=None,
                    confidence=0.0,
                ),
            )
        name_order_evidence = [NameOrderEvidence(raw_name=name) for name in names]

        return BatchParseResult(
            names=names,
            results=individual_results,
            format_pattern=format_pattern,
            individual_analyses=individual_analyses,
            improvements=[],
            name_order_evidence=name_order_evidence,
        )
