# Sinonym

*A Chinese name detection and normalization library.*

Sinonym is a Python library designed to accurately detect and normalize Chinese names across various romanization systems. It filters out non-Chinese names (such as Western, Korean, Vietnamese, and Japanese names) to prevent false positives.

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

Based on a comprehensive analysis of the library's test suites, here is a detailed breakdown of its behavior.

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
    *   *Example:* The character `曾` is converted by `pypinyin` to `ceng`, but this library corrects it to the expected `Zeng`.

### 5. Performance

*   **High-Performance with Caching**
    *   The library is benchmarked to be very fast, capable of processing over 10,000 diverse names per second, and uses caching to significantly speed up the processing of repeated names.

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
git clone https://github.com/yourusername/sinonym.git
cd sinonym
uv sync
```

### Machine Learning Dependencies

Sinonym includes an optional ML-based Japanese name classifier for enhanced accuracy with all-Chinese character names. The core dependencies (scikit-learn, numpy, scipy) are automatically installed. If these are not available, the system gracefully falls back to rule-based classification without the ML enhancement.

## Quick Start

Here's a simple example of how to use Sinonym to detect and normalize a Chinese name:

```python
from sinonym.detector import ChineseNameDetector

# Initialize the detector
detector = ChineseNameDetector()

# --- Example 1: A simple Chinese name ---
result = detector.is_chinese_name("Li Wei")
if result.success:
    print(f"Normalized Name: {result.result}")
    # Expected Output: Normalized Name: Wei Li

# --- Example 2: A compound given name ---
result = detector.is_chinese_name("Wang Weiming")
if result.success:
    print(f"Normalized Name: {result.result}")
    # Expected Output: Normalized Name: Wei-Ming Wang

# --- Example 3: An all-Chinese character name ---
result = detector.is_chinese_name("巩俐")
if result.success:
    print(f"Normalized Name: {result.result}")
    # Expected Output: Normalized Name: Li Gong

# --- Example 4: A non-Chinese name ---
result = detector.is_chinese_name("John Smith")
if not result.success:
    print(f"Error: {result.error_message}")
    # Expected Output: Error: name not recognised as Chinese

# --- Example 5: Japanese name in Chinese characters (ML-enhanced detection) ---
result = detector.is_chinese_name("山田太郎")
if not result.success:
    print(f"Error: {result.error_message}")
    # Expected Output: Error: Japanese name detected by ML classifier
```

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

We use `ruff` for linting and formatting, and `mypy` for type checking. To ensure your code meets our quality standards, run the following commands:

```bash
# Run linting and formatting
uv run ruff check . --fix
uv run ruff format .

# Run type checking
uv run mypy sinonym/
```

## License

Sinonym is licensed under the MIT License. See the `LICENSE` file for more details.

## Contributing

We welcome contributions! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new feature branch.
3.  Make your changes and ensure all tests and quality checks pass.
4.  Submit a pull request.

## Data Sources

The accuracy of Sinonym is enhanced by data derived from ORCID records, which provides valuable frequency information for Chinese surnames and given names.