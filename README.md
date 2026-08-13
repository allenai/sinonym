# Sinonym

*A Chinese name detection and normalization library.*

Sinonym detects and normalizes Chinese names across several romanization
systems. Its legacy result fields distinguish Chinese from non-Chinese names;
the canonical APIs can also normalize recoverable person names from other
cultures, and TIMO provides final fields for writer integrations.

This was mostly written with Claude Code with extensive oversight from me... Sorry if the actual code is too AI-ish. It's fast, well-tested, and works pretty well.

Not all the tests pass, and the test suite is intentionally skewed towards failing tests, so I know what to try to work on next. It's more-or-less impossible to guess with 100% accuracy whether a Romanized Chinese name is in the `Given-Name Surname` or `Surname Given-Name` format, and the best approach is to try to guess the most likely format from a batch of names that should all have the same format (like all the authors of an academic paper or all the names in a specific dataset). This kind of batch processing is described below.

## Data Flow Pipeline

```
Raw Input
    ↓
TextPreprocessor (structural cleaning)
    ↓
NormalizationService (creates NormalizedInput with compound_metadata)
    ↓
CompoundDetector (generates metadata) → compound_metadata
    ↓
NameParsingService (uses compound_metadata)
    ↓
NameFormattingService (uses compound_metadata)
    ↓
Formatted Output
```

## What to Expect: Behavior and Output

### 1. Output Formatting & Standardization

*   **Name Order is `Given-Name Surname`**
    *   The library's primary function is to standardize names into a `Given-Name Surname` format, regardless of the input order.
    *   **Input:** `"Liu Dehua"` → **Output:** `"De-Hua Liu"`
    *   **Input:** `"Wei, Yu-Zhong"` → **Output:** `"Yu-Zhong Wei"`

*   **Capitalization is `Title Case`**
    *   The output is consistently formatted in Title Case, with the first letter of the surname and each part of the given name capitalized.
    *   **Input:** `"DAN CHEN"` → **Output:** `"Dan Chen"`

*   **Chinese Given-Name Boundaries are Preserved**
    *   Recognized Chinese given names composed of multiple fully written syllables are joined by a hyphen when the boundary is inferred or authored as a hyphen. An authored apostrophe remains an ASCII apostrophe while the parsed component lineage stays split. An all-initial Chinese given span remains one compound first name, with every true initial rendered with a period. In a mixed span, fully written syllables form the given name and standalone initials occupy the middle-name field.
    *   **Input (Standard):** `"Wang Li Ming"` → **Output:** `"Li-Ming Wang"`
    *   **Input (Apostrophe):** `"Zheng Cui’e"` → **Output:** `"Cui'e Zheng"`
    *   **Input (Initials):** `"Y. Z. Wei"` → **Output:** `"Y.-Z. Wei"`
    *   **Input (Mixed):** `"Wei M. Wang"` → **Output:** `"Wei M. Wang"` (`given_name="Wei"`, `middle_name="M."`)
    *   **Input (Reduplicated):** `"Chen Linlin"` → **Output:** `"Lin-Lin Chen"`

### 2. Name Component Handling

*   **Compound Surname Formatting is Strictly Preserved**
    *   The library identifies compound (two-character) surnames and preserves their original formatting (compact, spaced, hyphenated, or CamelCase).
    *   **Input (Compact):** `"Duanmu Wenjie"` → **Output:** `"Wen-Jie Duanmu"`
    *   **Input (Spaced):** `"Au Yeung Chun"` → **Output:** `"Chun Au Yeung"`
    *   **Input (Hyphenated):** `"Au-Yeung Chun"` → **Output:** `"Chun Au-Yeung"`
    *   **Input (CamelCase):** `"AuYeung Ka Ming"` → **Output:** `"Ka-Ming AuYeung"`

*   **Unspaced Compound Given Names are Split and Hyphenated**
    *   If a multi-syllable given name is provided as a single unspaced string, the library identifies the syllables and inserts hyphens.
    *   **Input:** `"Wang Xueyin"` → **Output:** `"Xue-Yin Wang"`

### 3. Input Flexibility & Error Correction

*   **Handles All-Chinese Character Names**
    *   It correctly processes names written entirely in Chinese characters, applying surname-first convention with frequency-based disambiguation.
    *   **Input:** `"巩俐"` → **Output:** `"Li Gong"` (李 is more frequent surname than 巩)
    *   **Input:** `"李伟"` → **Output:** `"Wei Li"` (李 recognized as surname in first position)

*   **Handles Mixed Chinese (Hanzi) and Roman Characters**
    *   It correctly parses names containing both Chinese characters and Pinyin, using the Roman parts for the output.
    *   **Input:** `"Xiaohong Li 张小红"` → **Output:** `"Xiao-Hong Li"`

*   **Normalizes Diacritics, Accents, and Special Characters**
    *   It converts pinyin with tone marks and special characters like `ü` into their basic Roman alphabet equivalents.
    *   **Input:** `"Dèng Yǎjuān"` → **Output:** `"Ya-Juan Deng"`

*   **Normalizes Full-Width Characters**
    *   It processes full-width Latin characters (often from PDFs) into standard characters.
    *   **Input:** `"Ｌｉ　Ｘｉａｏｍｉｎｇ"` → **Output:** `"Xiao-Ming Li"`

*   **Handles Messy Formatting (Commas, Dots, Spacing)**
    *   The library correctly parses names despite common data entry or OCR errors.
    *   **Input (Bad Comma):** `"Chen,Mei Ling"` → **Output:** `"Mei-Ling Chen"`
    *   **Input (Dot Separators):** `"Li.Wei.Zhang"` → **Output:** `"Li-Wei Zhang"`

*   **Splits Concatenated Names**
    *   It can split names that have been concatenated without spaces, using CamelCase or mixed-case cues.
    *   **Input:** `"ZhangWei"` → **Output:** `"Wei Zhang"`

*   **Strips Parenthetical Western Names**
    *   If a Western name is included in parentheses, it is stripped out, and the remaining Chinese name is parsed correctly.
    *   **Input:** `"李（Peter）Chen"` → **Output:** `"Li Chen"`

### 4. Cultural & Regional Specificity

*   **Keeps Chinese Recognition Conservative**
    *   The legacy `success`, `result`, and `parsed` fields reject names that lack Chinese evidence, avoiding false positives. Recoverable people can still have a normalized `canonical_name`.
    *   **Western:** Rejects `"John Smith"` and even `"Christian Wong"`.
    *   **Korean:** Rejects `"Kim Min-jun"`.
    *   **Vietnamese:** Rejects `"Nguyen Van Anh"`.
    *   **Japanese:** Rejects `"Sato Taro"` and **Japanese names in Chinese characters** like `"山田太郎"` (Yamada Taro) using ML classification.

*   **Supports Regional Romanizations (Cantonese, Wade-Giles)**
    *   The library recognizes and preserves different English romanization systems.
    *   **Cantonese:** Input `"Chan Tai Man"` becomes `"Tai-Man Chan"` (not `"Chen"`).
    *   **Wade-Giles:** Input `"Ts'ao Ming"` becomes `"Ming Ts'ao"` (preserves apostrophe).

### 5. Performance

*   **High-Performance with Caching**
    *   The library is benchmarked to be very fast, capable of processing over 3,000 diverse names per second, and uses caching to significantly speed up the processing of repeated names.

## How It Works

Sinonym processes names through a multi-stage pipeline designed for high accuracy and performance:

1.  **Input Preprocessing**: The input string is cleaned and normalized. This includes handling mixed scripts (e.g., "张 Wei") and standardizing different romanization variants.
2.  **All-Chinese Detection**: The system detects inputs written entirely in Chinese characters and applies Han-to-Pinyin conversion with surname-first ordering preferences.
3.  **Chinese Recognition**: Linguistic patterns and machine learning keep Western, Korean, Vietnamese, and Japanese names out of the legacy Chinese result fields. For all-Chinese character inputs, a trained ML classifier (99.5% accuracy) distinguishes names such as "山田太郎" from Chinese names. The all-person canonical path can still normalize recoverable non-Chinese people.
4.  **Probabilistic Parsing**: The system identifies potential surname and given name boundaries by leveraging frequency data, which helps in accurately distinguishing between a surname and a given name. For all-Chinese inputs, it applies a surname-first bonus while still considering frequency data.
5.  **Compound Name Splitting**: For names with fused given names (e.g., "Weiming"), a tiered confidence system is used to correctly split them into their constituent parts (e.g., "Wei-Ming").
6.  **Output Formatting**: The final output is standardized to a "Given-Name Surname" format (e.g., "Wei Zhang").

## Installation

To get started with Sinonym, clone the repository and install the necessary dependencies using `uv`:

```bash
git clone https://github.com/allenai/sinonym.git
cd sinonym
```

1. From repo root:

```bash
# create the project venv (uv defaults to .venv if you don't give a name)
uv venv --python 3.11
```

2. Activate the venv (choose one):

```bash
# macOS / Linux (bash / zsh)
source .venv/bin/activate

# Windows PowerShell
. .venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat
```

3. Install project dependencies (dev extras):

```bash
uv sync --active --all-extras --dev
```

### Machine Learning Dependencies

Sinonym includes a ML-based Japanese vs Chinese name classifier for enhanced accuracy with all-Chinese character names.

## Quick Start

The same detector supports legacy Chinese recognition, all-person canonical
normalization, and batch context:

```python
from sinonym.detector import ChineseNameDetector

detector = ChineseNameDetector()

# Legacy Chinese recognition and formatting.
chinese = detector.normalize_name("Li Wei")
assert chinese.success
print(chinese.result)  # Wei Li

compound = detector.normalize_name("Wang Weiming")
print(compound.result)  # Wei-Ming Wang

# A non-Chinese person fails the legacy Chinese check but still gets a
# canonical all-person representation.
western = detector.normalize_name("John Smith")
assert not western.success
print(western.canonical_name.text)  # John Smith

# Related names can vote on one shared input convention.
authors = ["Zhang Wei", "Li Ming", "Wang Xiaoli"]
batch = detector.analyze_name_batch(authors)
print(batch.format_pattern.dominant_format.value)  # surname_first
print([result.result for result in batch.results])
# ['Wei Zhang', 'Ming Li', 'Xiao-Li Wang']
```

## Parse Results

When you call `normalize_name`, you get a `ParseResult` with helpful structured fields:

- `success`: True/False indicating recognition as Chinese
- `result`: Final formatted string in `Given-Name Surname` order
- `parsed`: A `ParsedName` with normalized components in output order
  - `surname`, `given_name`: component strings as in `result`
  - `surname_tokens`, `given_tokens`: normalized, capitalized tokens used to form components
  - `middle_tokens`: normalized middle components, including standalone initials collected from any source position
  - `order`: component order descriptor, typically `["given", "middle", "surname"]`
- `parsed_original_order`: A `ParsedName` with the same semantic `surname` and
  `given_name` labels as `parsed`, plus an `order` list that records how those
  components appeared in the input.
- `canonical_name`: an all-person canonical representation. This is populated
  for Chinese and non-Chinese people, while the legacy `success`, `result`, and
  `parsed` fields remain Chinese-recognition fields.

### Canonical names for all people

`canonical_name.text` is the fully normalized display form. Its `normalized`
components expose `given_name`, `middle_name`, `surname`, and `suffix`, plus
immutable token tuples and their display order. Name dashes and apostrophes are
standardized to ASCII `-` and `'`; obvious titles and credentials are removed;
and true generational suffixes are kept in the suffix field.

Periods are treated by role and shape rather than removed globally. Known
leading titles and trailing credentials are consumed; generational suffixes are
canonicalized; every true initial is rendered as an uppercase letter followed
by one period; transliteration abbreviations such as ``M.Yu.`` retain their
mixed casing; and a terminal full stop on an ordinary word is removed as
sentence punctuation. Dotted, fused-dotted, and spaced initial sequences are
semantically equivalent. An undelimited token such as ``AD`` is not split
without separate compact-initial evidence.

The culture-specific field policy is:

- On the non-Chinese path, the first unbound initial is the given name and all
  later initials are middle names. A fully written given name remains given, and
  following initials are middle names when a full surname remains. If comma-free
  input contains only `Full I I`, capitalization or periods alone do not justify
  reordering: the source order is retained, with the final initial serving as the
  required surname floor. An explicit comma, or a structured record whose only
  populated name component is `last_name`, supplies stronger surname-first evidence.
- On the Chinese path, an all-initial given span is one hyphenated compound
  given name. In a mixed span, fully written Chinese syllables form the
  hyphenated given name and every standalone initial is placed in the middle
  field, regardless of its source position.
- Explicit hyphens bind their parts and remain hyphens; explicit apostrophes
  remain ASCII apostrophes while preserving split-token lineage. Native-script
  alignment and reviewed identity evidence override shape-based initial inference,
  so a proven one-letter syllable remains undotted.

```python
western = detector.normalize_name("Dr. Ana–Maria O’Neill PhD")
assert not western.success  # unchanged: not recognized as Chinese
assert western.parsed is None
assert western.canonical_name.text == "Ana-Maria O'Neill"
assert western.canonical_name.normalized.given_name == "Ana-Maria"
assert western.canonical_name.normalized.surname == "O'Neill"

suffixed = detector.normalize_person_name("Steve Blando IV")
assert suffixed.text == "Steve Blando IV"
assert suffixed.normalized.suffix == "IV"

repaired = detector.normalize_person_name_components(
    first_name="dr steve",
    middle_name="marsh",
    last_name="phd",
)
assert repaired.text == "Steve Marsh"
assert repaired.normalized.given_name == "Steve"
assert repaired.normalized.middle_name == ""
assert repaired.normalized.surname == "Marsh"
```

### Canonical output contract

`normalize_name` keeps its legacy Chinese-recognition fields (`success`,
`result`, `parsed`, and `error_message`) and attaches `canonical_name` for any
recoverable person. Consumers that write normalized author fields should use
`canonical_name.normalized`; `canonical_name.source` records source spelling
and supplied component order for lineage, not an alternative normalized
answer.

`normalize_person_name_components` accepts structured first/middle/last input,
preserves that input in `canonical_name.source`, and may update
`canonical_name.normalized` through the same culture-specific initial and
East Asian routing policies used for raw names. Unsupported inputs keep the
generic canonical assignment; invalid and non-person inputs have no canonical
name. Reviewed source-only metadata, credential rows, and last-field-only Han
institutions are rejected at this structured boundary. Raw Hangul surfaces
whose every token has a reviewed organization suffix are likewise non-person
inputs; ordinary Hangul person names are unaffected.

Optional detector parsing weights must contain eight or nine finite real
numbers (booleans are not coefficients). Eight-element vectors receive the
documented ninth default. Invalid vectors raise during construction, before
service initialization. A custom `ChineseNameConfig.min_tokens_required` is a
lower bound and must be an integer of at least two.

Initials do not themselves establish that a name is Chinese. In particular, an
initials-only name with a cross-cultural surname spelling such as `Lee`, `Lim`,
`Tan`, or `Yi` stays on the non-Chinese fallback unless native script, identity
data, or other affirmative evidence establishes the Chinese path.

### Routed writer integration

New writer integrations should use the `sinonym` TIMO model. It accepts aligned
structured paper authors and returns one directly writable field object for
each author. Writers copy those fields as-is unless `resolution_action` is
`suppress`; no downstream fallback or suffix merge is needed.

Source fields ordinarily preserve lineage while scalar and batch inference use
their combined text. A small set of reviewed structured shapes can also supply
direct evidence. See the [TIMO writer contract](docs/timo.md) for a
runnable example, wire shapes, missing-value rules, decisions, and failure
semantics.

That reviewed tier includes guarded two- and three-component CJK
transliterations packed into a last field with an authored middle dot. The
delimiter boundary is preserved without treating every middle dot, or every
source-field label, as semantic role evidence.

TIMO clients select `sinonym` for terminal writer-ready fields.

### Conservative East Asian routing

Raw parsing preserves visible order by default. A separate conservative router
assigns semantic family-first components only for evidence combinations that
held the existing non-Chinese benchmark constant: three-syllable compact
Hangul; Japanese native text supported by the Chinese/Japanese classifier and
component dictionaries; strict Korean romanized shapes; diacritic-bearing
Vietnamese names and guarded, reviewed bare-ASCII Vietnamese family-first
forms; and two-token Japanese romanizations whose surname/given dictionaries
support only the family-first direction. Ambiguous or unsupported names retain
the generic input-order normalization.

For spaced native Japanese names, mutually exclusive dictionary evidence is a
hard writer decision: strict family-first evidence assigns the exchanged
endpoints, while strict given-first evidence preserves them. TIMO applies those
decisions before PP/VYS candidates, and the public canonical APIs apply every
terminal mapped decision before a competing Chinese interpretation. One-sided
or conflicting dictionary shapes remain soft and retain the existing
conservative arbitration.

The East Asian assets are primarily component lexicons. The Roman asset also
contains a small, provenance-backed exact full-name tier for reviewed routing
decisions. Sources, hashes, and licenses are documented in
[`sinonym/data/EAST_ASIAN_NAME_LEXICONS.md`](sinonym/data/EAST_ASIAN_NAME_LEXICONS.md);
the rebuild tool is
[`scripts/build_east_asian_name_lexicons.py`](scripts/build_east_asian_name_lexicons.py).

Notes:
- The tokens in `parsed` and `parsed_original_order` are the same normalized tokens; only the conceptual ordering differs via the `order` list.
- `middle_tokens` are preserved in both structures and included between given and surname when present.

Examples:

```python
res = detector.normalize_name("Li Wei")
# res.result == "Wei Li"
# res.parsed.order == ["given", "middle", "surname"]
# res.parsed_original_order.order == ["surname", "given"]
# res.parsed_original_order.given_name == "Wei"
# res.parsed_original_order.surname == "Li"

res = detector.normalize_name("Chi-Ying F. Huang")
# res.result == "Chi-Ying F. Huang"
# res.parsed.given_tokens == ["Chi", "Ying"]
# res.parsed.middle_tokens == ["F."]
# res.parsed.order == ["given", "middle", "surname"]
# res.parsed_original_order.order == ["given", "middle", "surname"]
# res.parsed_original_order.given_name == "Chi-Ying"
# res.parsed_original_order.surname == "Huang"
```

## Batch Processing for Consistent Formatting

Use batch processing when related names, such as one paper's author list, are
likely to share an ordering convention. Sinonym uses eligible names to infer a
surname-first or given-first pattern, then applies a clear pattern to ambiguous
members without forcing non-voting or unambiguous names into it.

```python
from sinonym.detector import ChineseNameDetector

detector = ChineseNameDetector()
batch = detector.analyze_name_batch(
    ["Zhang Wei", "Li Ming", "Wang Xiaoli", "Liu Jiaming"],
)
print(batch.format_pattern.dominant_format.value)
for raw_name, result in zip(batch.names, batch.results, strict=True):
    print(raw_name, "->", result.result or result.error_message)
```

`analyze_name_batch()` keeps the decision evidence; use
`process_name_batch()` when you only need the aligned `ParseResult` list. The
defaults are `format_threshold=0.55` and `minimum_batch_size=2`; correction
still requires at least two eligible votes.

See [Batch processing](docs/batch_processing.md) for API selection, evidence
fields, validation and failure behavior, mixed inputs, and multiprocessing.

## Development

If you'd like to contribute to Sinonym, here’s how to set up your development environment.

### Setup

First, clone the repository:

```bash
git clone https://github.com/allenai/sinonym.git
cd sinonym
```

Then, install the development dependencies:

```bash
uv sync --active --all-extras --dev
```

### Running Tests

To run the test suite, use the following command:

```bash
uv run pytest
```

### Code Quality

We use `ruff` for linting and formatting:

```bash
# Run linting and formatting
uv run ruff check . --fix
uv run ruff format .
```

### Benchmarking & Profiling

See [scripts/README.md](scripts/README.md) for benchmark, profiling, and test status scripts.

## License

Sinonym is licensed under the Apache 2.0 License. See the `LICENSE` file for more details.

## Contributing

We welcome contributions! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new feature branch.
3.  Make your changes and ensure all tests and quality checks pass.
4.  Submit a pull request.

## Data Sources

The accuracy of Sinonym is enhanced by data derived from ORCID records, which provides valuable frequency information for Chinese surnames and given names.
