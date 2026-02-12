# Sinonym

*A Chinese name detection and normalization library.*

Sinonym is a Python library designed to accurately detect and normalize Chinese names across various romanization systems. It filters out non-Chinese names (such as Western, Korean, Vietnamese, and Japanese names).

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

*   **Given Names are Hyphenated**
    *   Given names composed of multiple syllables are joined by a hyphen. This applies to standard names, names with initials, and reduplicated (repeated) names.
    *   **Input (Standard):** `"Wang Li Ming"` → **Output:** `"Li-Ming Wang"`
    *   **Input (Initials):** `"Y. Z. Wei"` → **Output:** `"Y-Z Wei"`
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

*   **Rejects Non-Chinese Names**
    *   The library uses advanced heuristics and machine learning to reject names from other cultures to avoid false positives.
    *   **Western:** Rejects `"John Smith"` and even `"Christian Wong"`.
    *   **Korean:** Rejects `"Kim Min-jun"`.
    *   **Vietnamese:** Rejects `"Nguyen Van Anh"`.
    *   **Japanese:** Rejects `"Sato Taro"` and **Japanese names in Chinese characters** like `"山田太郎"` (Yamada Taro) using ML classification.

*   **Supports Regional Romanizations (Cantonese, Wade-Giles)**
    *   The library recognizes and preserves different English romanization systems.
    *   **Cantonese:** Input `"Chan Tai Man"` becomes `"Tai-Man Chan"` (not `"Chen"`).
    *   **Wade-Giles:** Input `"Ts'ao Ming"` becomes `"Ming Ts'ao"` (preserves apostrophe).

*   **Corrects for Pinyin Library Inconsistencies**
    *   It contains an internal mapping to fix cases where the underlying `pypinyin` library's output doesn't match the most common romanization for a surname.
    *   *Example:* The character `曾` is converted by `pypinyin` to `Zeng`, but this library corrects it to the expected `Zeng`.

### 5. Performance

*   **High-Performance with Caching**
    *   The library is benchmarked to be very fast, capable of processing over 3,000 diverse names per second, and uses caching to significantly speed up the processing of repeated names.

## How It Works

Sinonym processes names through a multi-stage pipeline designed for high accuracy and performance:

1.  **Input Preprocessing**: The input string is cleaned and normalized. This includes handling mixed scripts (e.g., "张 Wei") and standardizing different romanization variants.
2.  **All-Chinese Detection**: The system detects inputs written entirely in Chinese characters and applies Han-to-Pinyin conversion with surname-first ordering preferences.
3.  **Ethnicity Classification**: The name is analyzed to filter out non-Chinese names. This stage uses linguistic patterns and machine learning to identify and reject Western, Korean, Vietnamese, and Japanese names. For all-Chinese character inputs, a trained ML classifier (99.5% accuracy) determines if names like "山田太郎" are Japanese vs Chinese.
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

Here's a simple example of how to use Sinonym to detect and normalize a Chinese name:

```python
from sinonym.detector import ChineseNameDetector

# Initialize the detector
detector = ChineseNameDetector()

# --- Example 1: A simple Chinese name ---
result = detector.normalize_name("Li Wei")
if result.success:
    print(f"Normalized Name: {result.result}")
    # Expected Output: Normalized Name: Wei Li

# --- Example 2: A compound given name ---
result = detector.normalize_name("Wang Weiming")
if result.success:
    print(f"Normalized Name: {result.result}")
    # Expected Output: Normalized Name: Wei-Ming Wang

# --- Example 3: An all-Chinese character name ---
result = detector.normalize_name("巩俐")
if result.success:
    print(f"Normalized Name: {result.result}")
    # Expected Output: Normalized Name: Li Gong

# --- Example 4: A non-Chinese name ---
result = detector.normalize_name("John Smith")
if not result.success:
    print(f"Error: {result.error_message}")
    # Expected Output: Error: name not recognised as Chinese

# --- Example 5: Japanese name in Chinese characters (ML-enhanced detection) ---
result = detector.normalize_name("山田太郎")
if not result.success:
    print(f"Error: {result.error_message}")
    # Expected Output: Error: Japanese name detected by ML classifier

# --- Example 6: Batch processing of academic author list ---
author_list = ["Zhang Wei", "Li Ming", "Wang Xiaoli", "Liu Jiaming", "Feng Cha"]
batch_result = detector.analyze_name_batch(author_list)
print(f"Format detected: {batch_result.format_pattern.dominant_format}")
print(f"Confidence: {batch_result.format_pattern.confidence:.1%}")
# Expected Output: Format detected: NameFormat.SURNAME_FIRST, Confidence: 94%

for i, result in enumerate(batch_result.results):
    if result.success:
        print(f"{author_list[i]} → {result.result}")
# Expected Output: Zhang Wei → Wei Zhang, Li Ming → Ming Li, etc.

# --- Example 7: Quick format detection for data validation ---
unknown_format_list = ["Wei Zhang", "Ming Li", "Xiaoli Wang"]
pattern = detector.detect_batch_format(unknown_format_list)
if pattern.threshold_met:
    print(f"Consistent {pattern.dominant_format} formatting detected")
    print(f"Safe to process as batch with {pattern.confidence:.1%} confidence")
else:
    print("Mixed formatting detected - process individually")

  # --- Example 8: Simple batch processing for data cleanup ---
  messy_names = ["Li, Wei", "Zhang.Ming", "Wang Xiaoli"]
  clean_results = detector.process_name_batch(messy_names)
  for original, clean in zip(messy_names, clean_results):
      if clean.success:
          print(f"Cleaned: '{original}' → '{clean.result}'")
      # Expected Output: Li, Wei → Wei Li, Zhang.Ming → Ming Zhang, etc.
```

## Parse Results

When you call `normalize_name`, you get a `ParseResult` with helpful structured fields:

- `success`: True/False indicating recognition as Chinese
- `result`: Final formatted string in `Given-Name Surname` order
- `parsed`: A `ParsedName` with normalized components in output order
  - `surname`, `given_name`: component strings as in `result`
  - `surname_tokens`, `given_tokens`: normalized, capitalized tokens used to form components
  - `middle_tokens`: trailing single-letter initials extracted from given name, if present
  - `order`: component order descriptor, typically `["given", "middle", "surname"]`
- `parsed_original_order`: A `ParsedName` aligned to the input’s original order. In this view, the labels are positional: `given` corresponds to the first component(s) in the original input, and `surname` to the last component(s), regardless of the semantic roles used for normalization.

Notes:
- The tokens in `parsed` and `parsed_original_order` are the same normalized tokens; only the conceptual ordering differs via the `order` list.
- `middle_tokens` are preserved in both structures and included between given and surname when present.

Examples:

```python
res = detector.normalize_name("Li Wei")
# res.result == "Wei Li"
# res.parsed.order == ["given", "middle", "surname"]
# res.parsed_original_order.order == ["surname", "given"]
# res.parsed_original_order.given_name == "Li"   # first in input
# res.parsed_original_order.surname == "Wei"     # last in input

res = detector.normalize_name("Chi-Ying F. Huang")
# res.result == "Chi-Ying F Huang"
# res.parsed.given_tokens == ["Chi", "Ying"]
# res.parsed.middle_tokens == ["F"]
# res.parsed.order == ["given", "middle", "surname"]
# res.parsed_original_order.order == ["given", "middle", "surname"]
# res.parsed_original_order.given_name == "Chi-Ying"
# res.parsed_original_order.surname == "Huang"
```

## Batch Processing for Consistent Formatting

Sinonym includes advanced batch processing capabilities that significantly improve accuracy when processing lists of names that share consistent formatting patterns. This is particularly valuable for real-world datasets like academic author lists, company directories, or database migrations.

### How Batch Processing Works

When processing multiple names together, Sinonym:

1.  **Detects Format Patterns**: Analyzes the entire batch to identify whether names follow a surname-first (e.g., "Zhang Wei") or given-first (e.g., "Wei Zhang") pattern
2.  **Aggregates Evidence**: Uses frequency statistics across all names to build confidence in the detected pattern
3.  **Applies Consistent Formatting**: When confidence exceeds 67%, applies the detected pattern to improve parsing of ambiguous individual names
4.  **Tracks Improvements**: Identifies which names benefit from batch context vs. individual processing

### Key Benefits

*   **Fixes Ambiguous Cases**: Names like "Feng Cha" that are difficult to parse individually become clear in batch context
*   **Maintains Consistency**: Ensures all names in a list follow the same formatting pattern
*   **High Accuracy**: Achieves 90%+ success rate on previously problematic cases when proper format context is available
*   **Intelligent Fallback**: Automatically falls back to individual processing when batch patterns are unclear

### Batch Processing Methods

```python
from sinonym.detector import ChineseNameDetector

detector = ChineseNameDetector()

# Full batch analysis with detailed results
result = detector.analyze_name_batch([
    "Zhang Wei", "Li Ming", "Wang Xiaoli", "Liu Jiaming"
])
print(f"Format detected: {result.format_pattern.dominant_format}")
print(f"Confidence: {result.format_pattern.confidence:.1%}")
print(f"Improved names: {len(result.improvements)}")

# Quick format detection without full processing
pattern = detector.detect_batch_format([
    "Zhang Wei", "Li Ming", "Wang Xiaoli"
])
if pattern.threshold_met:
    print(f"Strong {pattern.dominant_format} pattern detected")

# Simple batch processing (returns list of results)
results = detector.process_name_batch([
    "Zhang Wei", "Li Ming", "Wang Xiaoli"
])
for result in results:
    print(f"Processed: {result.result}")
```

### When to Use Batch Processing

*   **Academic Papers**: Author lists typically follow consistent formatting
*   **Company Directories**: Employee lists often use uniform formatting conventions  
*   **Large Datasets**: Processing 100+ names where format consistency is expected

Batch processing requires a minimum of 2 names and works best with 5+ names for reliable pattern detection.

### Batch Processing Behavior

**Unambiguous Names**: Some names have only one possible parsing format (e.g., compound given names like "Wei‑Qi Wang"). Batch processing never forces such names into the detected pattern and never raises. These names keep their best individual parse while other Chinese names benefit from the jointly detected order.

**Confidence (Advisory Only)**: Batch detection computes a dominant format and a confidence value, but there is no confidence threshold gating. Results are always returned. The confidence is informational (e.g., for logging/UX) and is not used to raise errors.

### Batch Processing with Mixed Name Types

Batch processing works seamlessly with mixed datasets containing both Chinese and non-Chinese names. Non-Chinese names are rejected during individual analysis but still appear in the batch output as failed results.

```python
# Mixed dataset: 2 Western names + 8 Chinese names
mixed_names = [
    "John Smith",     # Western - will be rejected
    "Mary Johnson",   # Western - will be rejected  
    "Xin Liu",        # Chinese - GIVEN_FIRST preference
    "Yang Li",        # Chinese - GIVEN_FIRST preference
    "Wei Zhang",      # Chinese - GIVEN_FIRST preference
    "Ming Wang",      # Chinese - GIVEN_FIRST preference
    "Li Chen",        # Chinese - GIVEN_FIRST preference
    "Hui Zhou",       # Chinese - GIVEN_FIRST preference
    "Feng Zhao",      # Chinese - GIVEN_FIRST preference
    "Tong Zhang",     # Chinese - might prefer SURNAME_FIRST (ambiguous)
]

result = detector.analyze_name_batch(mixed_names)

# Format detection uses only the 8 Chinese names
# If 7 prefer GIVEN_FIRST vs 1 SURNAME_FIRST = 87.5% confidence
# GIVEN_FIRST pattern is applied to Chinese names; non‑Chinese names return clear failures

print(f"Total results: {len(result.results)}")  # 10 (same as input)
print(f"Format detected: {result.format_pattern.dominant_format}")  # GIVEN_FIRST
print(f"Confidence: {result.format_pattern.confidence:.1%}")  # 87.5%

# Check results by type
for i, (name, result_obj) in enumerate(zip(mixed_names, result.results)):
    if result_obj.success:
        print(f"✅ {name} → {result_obj.result}")
    else:
        print(f"❌ {name} → {result_obj.error_message}")

# Output:
# ❌ John Smith → name not recognised as Chinese
# ❌ Mary Johnson → name not recognised as Chinese  
# ✅ Xin Liu → Xin Liu
# ✅ Yang Li → Yang Li
# ✅ Wei Zhang → Wei Zhang
# ... (all Chinese names processed successfully with consistent formatting)
```

**Key Benefits:**
- **Maintains input-output correspondence**: Results array matches input array length and order
- **Robust format detection**: Only valid Chinese names contribute to pattern detection
- **Consistent formatting**: All Chinese names get the same detected format applied
- **Clear failure reporting**: Non-Chinese names are clearly marked as failed with error messages

## Development

If you'd like to contribute to Sinonym, here’s how to set up your development environment.

### Setup

First, clone the repository:

```bash
git clone https://github.com/yourusername/sinonym.git
cd sinonym
```

Then, install the development dependencies:

```bash
uv sync --extra dev
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
