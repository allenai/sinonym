# Sinonym

A sophisticated Chinese name detection and normalization library that handles various romanization systems with robust filtering to prevent false positives from Western, Korean, Vietnamese, and Japanese names.

## Features

- **Comprehensive Romanization Support**: Handles Pinyin, Wade-Giles, Cantonese, and mixed scripts
- **Advanced Name Splitting**: Uses tiered confidence system for splitting compound given names
- **Robust Filtering**: Prevents false positives from non-Chinese names
- **Performance Optimized**: Lazy normalization, early exit patterns, and persistent caching
- **Thread Safe**: Immutable data structures suitable for concurrent use

## Installation

Install from source using [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/yourusername/sinonym.git
cd sinonym
uv sync
```

## Quick Start

```python
from sinonym.chinese_names import ChineseNameDetector

# Initialize the detector
detector = ChineseNameDetector()

# Detect and normalize Chinese names
result = detector.parse("Li Wei")
if result.success:
    print(f"Normalized: {result.normalized_name}")
    print(f"Surname: {result.surname}")
    print(f"Given name: {result.given_name}")
```

## Architecture

The library uses a clean service-oriented architecture:

- **NormalizationService**: Centralized normalization with lazy computation
- **PinyinCacheService**: Persistent cache management for Han→Pinyin mappings
- **DataInitializationService**: Immutable data structure initialization
- **ChineseNameDetector**: Main detection engine with dependency injection

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/sinonym.git
cd sinonym

# Install with development dependencies
uv sync --extra dev
```

### Running Tests

```bash
uv run pytest
```

### Code Quality

```bash
# Run linting and formatting
uv run ruff check . --fix
uv run ruff format .

# Type checking
uv run mypy sinonym/
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## Data

The library includes Chinese name frequency data derived from ORCID records for improved accuracy in surname/given name boundary detection.