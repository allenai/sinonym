#!/usr/bin/env python3
"""
Generate ML Training Data for Parsing System

This script generates training data for the machine learning-based parsing system by:
1. Downloading Chinese names from the corpus
2. Romanizing them using the existing sinonym services
3. Generating all possible parses for each name
4. Creating ground truth labels (surname-first is correct for Chinese names)
5. Extracting features for each parse candidate
6. Saving structured training dataset

The output will be used to train a ranking model that can replace the rule-based
parsing system in sinonym/services/parsing.py.

This is historical - I tried to make a ML model, but it didn't work well.
"""

import json
import random
import unicodedata
from pathlib import Path
from typing import Any

import requests

# Import existing components
from sinonym.detector import ChineseNameDetector
from sinonym.services.parsing import NameParsingService

# Data sources (reuse from train_ml_classifier.py)
CN_URL = (
    "https://github.com/wainshine/Chinese-Names-Corpus/"
    "raw/refs/heads/master/Chinese_Names_Corpus/"
    "Chinese_Names_Corpus%EF%BC%88120W%EF%BC%89.txt"
)


def download_names(url: str, sample_size: int | None) -> list[str]:
    """Download Chinese names from corpus (reused from train_ml_classifier.py)."""
    print(f"Downloading {url.split('/')[-1]} ...", end=" ", flush=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    # Handle the Chinese corpus which has 3 header lines
    lines = resp.content.decode("utf-8", errors="ignore").splitlines()
    if "Chinese_Names_Corpus" in url:
        lines = lines[3:]  # Skip header lines for Chinese corpus

    lines = [ln.strip() for ln in lines if ln.strip()]
    if sample_size and len(lines) > sample_size:
        lines = random.sample(lines, sample_size)
    print(f"{len(lines):,} names.")
    return lines


def is_all_chinese_characters(name: str) -> bool:
    """Check if name contains only Chinese/Japanese characters."""
    return all(unicodedata.name(ch, "").startswith(("CJK UNIFIED", "CJK COMPATIBILITY")) for ch in name)


def romanize_chinese_name(name: str, detector: ChineseNameDetector) -> str | None:
    """
    Convert Chinese characters to romanized form using existing services.
    Returns None if conversion fails or result is not valid.
    """
    try:
        # Use the cache service to convert characters to pinyin
        romanized_parts = []
        for char in name:
            pinyin_result = detector._cache_service.han_to_pinyin_fast(char)
            if pinyin_result:
                # Take first pinyin option and remove tone numbers
                pinyin = pinyin_result[0].lower()
                # Remove tone numbers (1-4)
                pinyin_clean = "".join(c for c in pinyin if not c.isdigit())
                romanized_parts.append(pinyin_clean.title())

        if romanized_parts:
            return " ".join(romanized_parts)
        return None
    except Exception:
        return None


def generate_parse_candidates(
    tokens: list[str],
    parsing_service: NameParsingService,
    normalized_cache: dict[str, str],
    compound_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Generate all possible parse candidates for a tokenized name.
    Returns list of parse dictionaries with surname/given tokens.
    """
    try:
        # Use the existing _generate_all_parses_with_format method
        parses_with_format = parsing_service._generate_all_parses_with_format(
            tokens,
            normalized_cache,
            compound_metadata,
        )

        parse_candidates = []
        for surname_tokens, given_tokens, original_compound_format in parses_with_format:

            parse_candidates.append(
                {
                    "surname_tokens": surname_tokens,
                    "given_tokens": given_tokens,
                    "original_compound_format": original_compound_format,
                },
            )

        return parse_candidates
    except Exception as e:
        print(f"Error generating parses for {tokens}: {e}")
        return []


def extract_basic_features(
    surname_tokens: list[str],
    given_tokens: list[str],
    tokens: list[str],
    parsing_service: NameParsingService,
    normalized_cache: dict[str, str],
) -> dict[str, float]:
    """
    Extract the top 7 features from a parse candidate.
    Original 4 features plus 3 new statistical ratio features for surname/given disambiguation.
    """
    features = {}

    # Feature 1: surname_log_prob (Gain: 51.9)
    if surname_tokens:
        surname_key = parsing_service._surname_key(surname_tokens, normalized_cache)
        surname_logp = parsing_service._data.surname_log_probabilities.get(
            surname_key,
            parsing_service._config.default_surname_logp,
        )
        features["surname_log_prob"] = surname_logp / 10.0  # Scale down large negative values
    else:
        features["surname_log_prob"] = parsing_service._config.default_surname_logp / 10.0

    # Feature 2: given_logp_rank_avg (Gain: 14.8)
    if given_tokens:
        given_ranks = [
            _get_precomputed_given_rank(g_token, parsing_service, normalized_cache) for g_token in given_tokens
        ]
        features["given_logp_rank_avg"] = sum(given_ranks) / len(given_ranks)
    else:
        features["given_logp_rank_avg"] = 1.0  # Bottom percentile

    # Feature 3: surname_log_prob_median (Gain: 14.0)
    if surname_tokens:
        surname_median_logp = _get_precomputed_surname_median(surname_tokens, parsing_service, normalized_cache)
        features["surname_log_prob_median"] = surname_median_logp / 10.0
    else:
        features["surname_log_prob_median"] = parsing_service._config.default_surname_logp / 10.0

    # Feature 4: given_log_prob_sum (Gain: 13.8)
    if given_tokens:
        given_logp_sum = sum(
            parsing_service._data.given_log_probabilities.get(
                parsing_service._given_name_key(g_token, normalized_cache),
                parsing_service._config.default_given_logp,
            )
            for g_token in given_tokens
        )
        features["given_log_prob_sum"] = given_logp_sum / 10.0  # Scale down
    else:
        features["given_log_prob_sum"] = parsing_service._config.default_given_logp / 10.0

    # Feature 5: surname_given_rank_ratio - Key discriminator for Ao vs Xiang cases
    # if surname_tokens:
    #     features["surname_given_rank_ratio"] = _surname_given_rank_ratio(
    #         surname_tokens, parsing_service, normalized_cache
    #     )
    # else:
    #     features["surname_given_rank_ratio"] = 1.0

    # Feature 6: surname_logp_ratio - Ratio of surname to given log probabilities
    # if surname_tokens:
    #     features["surname_logp_ratio"] = _surname_logp_ratio(
    #         surname_tokens, parsing_service, normalized_cache
    #     )
    # else:
    #     features["surname_logp_ratio"] = 1.0

    # Feature 7: median_ratio_feature - Ratio using median probabilities
    # if surname_tokens:
    #     features["median_ratio_feature"] = _median_ratio_feature(
    #         surname_tokens, parsing_service, normalized_cache
    #     )
    # else:
    #     features["median_ratio_feature"] = 1.0

    return features


def _get_precomputed_surname_median(
    surname_tokens: list[str], parsing_service: NameParsingService, normalized_cache: dict[str, str],
) -> float:
    """Get pre-computed median-based surname log probability."""
    if len(surname_tokens) == 1:
        # Single surname - use pre-computed median
        token = surname_tokens[0]
        normalized = normalized_cache.get(token, parsing_service._normalizer.norm(token))

        # Look up pre-computed median
        return parsing_service._data.surname_median_logprobs.get(
            normalized, parsing_service._config.default_surname_logp,
        )
    # Compound surname - return the max-based value (no median for compounds)
    surname_key = parsing_service._surname_key(surname_tokens, normalized_cache)
    return parsing_service._data.surname_log_probabilities.get(
        surname_key, parsing_service._config.default_surname_logp,
    )


def _get_precomputed_given_median(
    token: str, parsing_service: NameParsingService, normalized_cache: dict[str, str],
) -> float:
    """Get pre-computed median-based given name log probability for a single token."""
    normalized = normalized_cache.get(token, parsing_service._normalizer.norm(token))

    # Look up pre-computed median
    return parsing_service._data.given_median_logprobs.get(normalized, parsing_service._config.default_given_logp)


def _get_precomputed_surname_rank(
    surname_tokens: list[str], parsing_service: NameParsingService, normalized_cache: dict[str, str],
) -> float:
    """Get pre-computed percentile rank of this surname in the frequency database (0-1 scale)."""
    surname_key = parsing_service._surname_key(surname_tokens, normalized_cache)

    # Look up pre-computed percentile rank
    return parsing_service._data.surname_percentile_ranks.get(
        surname_key, 1.0,
    )  # Bottom percentile for unknown surnames


def _get_precomputed_given_rank(
    token: str, parsing_service: NameParsingService, normalized_cache: dict[str, str],
) -> float:
    """Get pre-computed percentile rank of this given name character in the frequency database (0-1 scale)."""
    given_key = parsing_service._given_name_key(token, normalized_cache)

    # Look up pre-computed percentile rank
    return parsing_service._data.given_percentile_ranks.get(given_key, 1.0)  # Bottom percentile for unknown characters


def _surname_given_rank_ratio(
    surname_tokens: list[str],
    parsing_service: NameParsingService,
    normalized_cache: dict[str, str],
) -> float:
    """
    Mean of (P_surname / P_given) for the surname tokens, using
    percentile ranks as a proxy. Small epsilon avoids zero-division.

    This is the key discriminator for cases like Ao (high surname ratio) vs Xiang (low surname ratio).
    """
    eps = 1e-6
    ratios = []
    for token in surname_tokens:
        s_rank = _get_precomputed_surname_rank([token], parsing_service, normalized_cache)
        g_rank = _get_precomputed_given_rank(token, parsing_service, normalized_cache)
        # Convert ranks to probabilities: lower rank = higher probability
        surname_prob = 1.0 - s_rank + eps
        given_prob = 1.0 - g_rank + eps
        ratios.append(surname_prob / given_prob)
    return sum(ratios) / len(ratios) if ratios else 1.0


def _surname_logp_ratio(
    surname_tokens: list[str],
    parsing_service: NameParsingService,
    normalized_cache: dict[str, str],
) -> float:
    """
    Ratio of surname log-probability to given log-probability for surname tokens.
    Higher values indicate tokens that are much more likely to be surnames than given names.
    """
    eps = 1e-6
    ratios = []
    for token in surname_tokens:
        surname_key = parsing_service._surname_key([token], normalized_cache)
        given_key = parsing_service._given_name_key(token, normalized_cache)

        surname_logp = parsing_service._data.surname_log_probabilities.get(
            surname_key,
            parsing_service._config.default_surname_logp,
        )
        given_logp = parsing_service._data.given_log_probabilities.get(
            given_key,
            parsing_service._config.default_given_logp,
        )

        # Convert log probabilities to probabilities, then ratio
        surname_prob = max(eps, abs(surname_logp))  # Use absolute value since log probs are negative
        given_prob = max(eps, abs(given_logp))
        ratios.append(given_prob / surname_prob)  # Higher ratio = more likely to be surname

    return sum(ratios) / len(ratios) if ratios else 1.0


def _median_ratio_feature(
    surname_tokens: list[str],
    parsing_service: NameParsingService,
    normalized_cache: dict[str, str],
) -> float:
    """
    Ratio of median surname probability to median given probability for surname tokens.
    Uses pre-computed median values for efficient computation.
    """
    eps = 1e-6
    ratios = []
    for token in surname_tokens:
        surname_median = _get_precomputed_surname_median([token], parsing_service, normalized_cache)
        given_median = _get_precomputed_given_median(token, parsing_service, normalized_cache)

        # Convert log probabilities to probabilities, then ratio
        surname_prob = max(eps, abs(surname_median))
        given_prob = max(eps, abs(given_median))
        ratios.append(given_prob / surname_prob)  # Higher ratio = more likely to be surname

    return sum(ratios) / len(ratios) if ratios else 1.0


def determine_correct_surname_length(original_chinese_name: str, detector: ChineseNameDetector) -> int | None:
    """
    Determine the correct surname length using the surname database.
    Returns 1 or 2 for surname length, or None if name should be discarded.

    Logic:
    - If 2 characters: first is surname (return 1)
    - If 3-4 characters:
      - Check if first 2 chars are in familyname_orcid.csv, if yes return 2
      - Otherwise check if first char is in familyname_orcid.csv, if yes return 1
      - If neither, return None (discard)
    """
    name_length = len(original_chinese_name)

    if name_length == 2:
        return 1
    if name_length in [3, 4]:
        # Check if first 2 characters form a compound surname
        # Use surname_frequencies which contains the Chinese characters as keys
        first_two_chars = original_chinese_name[:2]
        if detector._data.surname_frequencies.get(first_two_chars, 0) > 0:
            return 2

        # Check if first character is a single surname
        first_char = original_chinese_name[0]
        if detector._data.surname_frequencies.get(first_char, 0) > 0:
            return 1

        # Neither case - discard this name
        return None
    # Names with 1 char or >4 chars are filtered out before this function
    return None


def create_ground_truth_label(
    surname_tokens: list[str],
    given_tokens: list[str],
    tokens: list[str],
    original_chinese_name: str,
    detector: ChineseNameDetector,
) -> bool | None:
    """
    Create ground truth label for a parse candidate by comparing with the Chinese name structure.
    Returns True if this parse is correct, False if incorrect, None if name should be discarded.

    The correct parse is determined by comparing the parse tokens with the romanized
    Chinese surname and given name parts.
    """
    # Determine the correct surname length from the original Chinese name
    correct_surname_length = determine_correct_surname_length(original_chinese_name, detector)
    if correct_surname_length is None:
        return None  # Signal to discard this entire example

    # Extract the Chinese surname and given name parts
    chinese_surname = original_chinese_name[:correct_surname_length]
    chinese_given = original_chinese_name[correct_surname_length:]

    # Romanize the Chinese parts to get expected tokens
    try:
        # Romanize surname
        expected_surname_tokens = []
        for char in chinese_surname:
            pinyin_result = detector._cache_service.han_to_pinyin_fast(char)
            if pinyin_result:
                pinyin = pinyin_result[0].lower()
                pinyin_clean = "".join(c for c in pinyin if not c.isdigit())
                expected_surname_tokens.append(pinyin_clean)

        # Romanize given name
        expected_given_tokens = []
        for char in chinese_given:
            pinyin_result = detector._cache_service.han_to_pinyin_fast(char)
            if pinyin_result:
                pinyin = pinyin_result[0].lower()
                pinyin_clean = "".join(c for c in pinyin if not c.isdigit())
                expected_given_tokens.append(pinyin_clean)

        if not expected_surname_tokens or not expected_given_tokens:
            return False

        # Compare the parse with the expected tokens
        # The parse is correct if it matches the ground truth exactly
        return surname_tokens == expected_surname_tokens and given_tokens == expected_given_tokens

    except Exception:
        # If romanization fails, return False
        return False


def process_names_batch(
    chinese_names: list[str],
    detector: ChineseNameDetector,
    batch_size: int = 1000,
) -> list[dict[str, Any]]:
    """Process a batch of Chinese names and generate training examples."""
    training_examples = []
    parsing_service = detector._parsing_service
    rejected_examples = []

    print(f"Processing {len(chinese_names)} names...")

    for i, chinese_name in enumerate(chinese_names):
        if i % 100 == 0:
            print(f"Processed {i}/{len(chinese_names)} names")

        # Romanize the Chinese name
        romanized_name = romanize_chinese_name(chinese_name, detector)
        if not romanized_name:
            continue

        # Tokenize the romanized name
        try:
            # First check if this name should be kept based on surname analysis
            correct_surname_length = determine_correct_surname_length(chinese_name, detector)
            if correct_surname_length is None:
                # Collect rejected examples for analysis
                if len(rejected_examples) < 20:  # Only collect first 20 for printing
                    first_char = chinese_name[0]
                    first_two_chars = chinese_name[:2] if len(chinese_name) >= 2 else chinese_name
                    rejected_examples.append(
                        {
                            "name": chinese_name,
                            "length": len(chinese_name),
                            "first_char": first_char,
                            "first_two_chars": first_two_chars,
                            "first_char_in_surnames": detector._data.surname_frequencies.get(first_char, 0) > 0,
                            "first_two_in_surnames": detector._data.surname_frequencies.get(first_two_chars, 0) > 0,
                        },
                    )
                continue  # Skip names that don't have valid surnames

            # Use the detector's preprocessing to get proper tokens
            result = detector.normalize_name(romanized_name)
            if not result.success:
                continue

            # Extract tokens from the preprocessing pipeline
            tokens = romanized_name.lower().split()
            if len(tokens) < 2:
                continue

            # Create normalized cache and compound metadata
            normalized_cache = {}
            compound_metadata = {}
            for token in tokens:
                normalized_cache[token] = detector._normalizer.norm(token)
                # Add basic compound metadata (simplified for now)
                compound_metadata[token] = type(
                    "CompoundMetadata",
                    (),
                    {"is_compound": False, "compound_target": None, "format_type": None},
                )()

            # Generate all possible parse candidates
            parse_candidates = generate_parse_candidates(tokens, parsing_service, normalized_cache, compound_metadata)

            if not parse_candidates:
                continue

            # Create training example
            example = {
                "original_chinese_name": chinese_name,
                "romanized_name": romanized_name,
                "tokens": tokens,
                "parses": [],
            }

            # Process each parse candidate
            for parse_candidate in parse_candidates:
                surname_tokens = parse_candidate["surname_tokens"]
                given_tokens = parse_candidate["given_tokens"]

                # Extract features
                features = extract_basic_features(
                    surname_tokens,
                    given_tokens,
                    tokens,
                    parsing_service,
                    normalized_cache,
                )

                # Create ground truth label
                is_correct = create_ground_truth_label(surname_tokens, given_tokens, tokens, chinese_name, detector)
                if is_correct is None:
                    continue  # Skip this parse if the name should be discarded

                # Add parse to example
                example["parses"].append(
                    {
                        "surname_tokens": surname_tokens,
                        "given_tokens": given_tokens,
                        "features": features,
                        "is_correct": is_correct,
                    },
                )

            # Add all examples with parses (don't filter by correctness)
            if example["parses"]:
                training_examples.append(example)

            # Continue processing all names (removed batch size limit for full dataset)

        except Exception as e:
            print(f"Error processing {chinese_name}: {e}")
            continue

    # Print rejected examples for analysis
    if rejected_examples:
        print(f"\nRejected {len(rejected_examples)} examples (showing first 20):")
        print("Name\tLen\tFirst\tFirst2\t1stInDB\t2ndInDB")
        print("-" * 50)
        for ex in rejected_examples:
            print(
                f"{ex['name']}\t{ex['length']}\t{ex['first_char']}\t{ex['first_two_chars']}\t{ex['first_char_in_surnames']}\t{ex['first_two_in_surnames']}",
            )

    return training_examples


def main():
    """Generate training data for ML parsing system."""
    print("Generating ML Parsing Training Data")
    print("=" * 50)

    # Set random seed for reproducibility
    random.seed(42)

    # Double the sample size for better training
    SAMPLE_SIZE = 200000  # Doubled from 100k to 200k

    # Download Chinese names
    print("Downloading Chinese names...")
    chinese_names = download_names(CN_URL, SAMPLE_SIZE)

    # Filter to Chinese character names only, and remove names with 1 char or >4 chars
    chinese_names = [name for name in chinese_names if is_all_chinese_characters(name) and 2 <= len(name) <= 4]
    print(f"Filtered to {len(chinese_names)} Chinese character names (2-4 chars only)")

    # Initialize the detector (this loads all the services we need)
    print("Initializing Chinese name detector...")
    detector = ChineseNameDetector()

    # Process names and generate training data
    training_examples = process_names_batch(chinese_names, detector, batch_size=1000)

    print(f"\nGenerated {len(training_examples)} training examples")

    # Calculate statistics
    total_parses = sum(len(example["parses"]) for example in training_examples)
    correct_parses = sum(sum(1 for parse in example["parses"] if parse["is_correct"]) for example in training_examples)

    print(f"Total parse candidates: {total_parses}")
    print(f"Correct parses: {correct_parses}")
    print(f"Accuracy baseline: {correct_parses/total_parses:.3f}")

    # Save training data
    try:
        import sinonym as _sin
        data_dir = Path(_sin.__file__).resolve().parent / "data"
    except Exception:
        data_dir = Path.cwd()
    data_dir.mkdir(parents=True, exist_ok=True)

    output_path = data_dir / "ml_parsing_training_data.json"
    print(f"\nSaving training data to {output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(training_examples, f, ensure_ascii=False, indent=2)

    print("✓ Training data generation completed!")

    # Save metadata
    metadata = {
        "total_examples": len(training_examples),
        "total_parses": total_parses,
        "correct_parses": correct_parses,
        "accuracy_baseline": correct_parses / total_parses if total_parses > 0 else 0,
        "sample_size": SAMPLE_SIZE,
        "chinese_names_processed": len(chinese_names),
    }

    metadata_path = data_dir / "ml_parsing_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
