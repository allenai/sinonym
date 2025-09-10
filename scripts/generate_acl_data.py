#!/usr/bin/env python3
"""
Process ACL 2025 authors and add Chinese names to ML training data.

The ACL authors come in "Given Surname" format (romanized), which is different
from our original Chinese dataset. We need to:
1. Identify which names are Chinese using our detector
2. Convert them to the training data format
3. Add them to our existing training split

This is historical - I tried to make a ML model, but it didn't work well.
"""

import json
import re
import sys
from pathlib import Path

sys.path.append(".")

from importlib.resources import files

from sinonym import ChineseNameDetector


def load_acl_authors():
    """Load ACL 2025 authors from text file."""
    authors = []
    acl_res = files("sinonym.data") / "acl_2025_authors.txt"
    # Read directly via importlib.resources
    for line in acl_res.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):  # Skip comments and empty lines
            authors.append(line)

    print(f"Loaded {len(authors)} ACL authors")
    return authors


def filter_chinese_names(authors):
    """Use our detector to identify Chinese names from ACL authors."""
    detector = ChineseNameDetector()

    chinese_names = []
    non_chinese_count = 0

    print("Filtering Chinese names...")
    for i, author in enumerate(authors):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(authors)} authors...")

        # Skip obviously non-Chinese patterns
        if re.search(r"\b(Jr|Sr|III|II|PhD|Dr|Prof)\b", author, re.IGNORECASE):
            non_chinese_count += 1
            continue

        # Skip names with too many parts (likely Western)
        parts = author.split()
        if len(parts) > 4:
            non_chinese_count += 1
            continue

        # Test with our detector
        result = detector.normalize_name(author)
        if result.success:
            # Parse the result to extract surname and given parts
            formatted_name = result.result  # e.g., "Wei-Ming Zhang"

            chinese_names.append(
                {
                    "original": author,  # "Wei Ming Zhang" (given surname format)
                    "formatted": formatted_name,  # "Wei-Ming Zhang" (given-surname format)
                    "tokens": author.split(),  # ["Wei", "Ming", "Zhang"]
                },
            )
        else:
            non_chinese_count += 1

    print(f"\nFound {len(chinese_names)} Chinese names out of {len(authors)} total")
    print(f"Filtered out {non_chinese_count} non-Chinese names")

    return chinese_names


def convert_to_training_format(chinese_names, detector):
    """Convert ACL Chinese names to our training data format."""

    training_examples = []

    print(f"\nConverting {len(chinese_names)} Chinese names to training format...")

    for i, name_data in enumerate(chinese_names):
        if i % 500 == 0:
            print(f"Processed {i}/{len(chinese_names)} names...")

        original = name_data["original"]  # "Wei Ming Zhang"
        tokens = name_data["tokens"]  # ["Wei", "Ming", "Zhang"]

        # For ACL data, the format is "Given Surname" (romanized)
        # So the last token is typically the surname, earlier tokens are given names

        if len(tokens) < 2:
            continue  # Skip single-token names

        # Generate parse candidates using our detector's logic
        # We'll use the detector to generate all possible parses, then identify the correct one

        # Use detector's normalization
        normalized_input = detector._normalizer.apply(original)

        # Generate all possible parses
        parses_with_format = detector._parsing_service._generate_all_parses_with_format(
            list(normalized_input.roman_tokens),
            normalized_input.norm_map,
            normalized_input.compound_metadata,
        )

        if not parses_with_format:
            continue

        # For ACL format (given surname), the correct parse should be:
        # - Surname: last token(s)
        # - Given: first token(s)

        # Find the parse that matches this pattern
        correct_parse = None
        all_parses = []

        for surname_tokens, given_tokens, original_compound_format in parses_with_format:
            parse_entry = {
                "surname_tokens": surname_tokens,
                "given_tokens": given_tokens,
            }
            all_parses.append(parse_entry)

            # Check if this matches ACL format (given surname)
            # In ACL format, given names come first, surname comes last
            original_tokens = list(normalized_input.roman_tokens)

            # For 2-token names: "Given Surname" -> given=first, surname=last
            if len(original_tokens) == 2:
                if (
                    len(given_tokens) == 1
                    and len(surname_tokens) == 1
                    and given_tokens[0] == original_tokens[0]
                    and surname_tokens[0] == original_tokens[1]
                ):
                    correct_parse = parse_entry
                    break

            # For 3+ token names, surname is typically the last token
            elif len(original_tokens) >= 3:
                # Check if surname is the last token and given names are the rest
                if (
                    len(surname_tokens) == 1
                    and surname_tokens[0] == original_tokens[-1]
                    and given_tokens == original_tokens[:-1]
                ):
                    correct_parse = parse_entry
                    break

        # If we couldn't identify the correct parse based on position,
        # use the detector's actual output as the correct parse
        if not correct_parse:
            result = detector.normalize_name(original)
            if result.success:
                # Parse the formatted result to extract correct parse
                formatted = result.result  # "Wei-Ming Zhang"
                parts = formatted.split()
                if len(parts) == 2:
                    given_part, surname_part = parts
                    # Handle hyphenated given names
                    if "-" in given_part:
                        given_tokens_correct = given_part.split("-")
                    else:
                        given_tokens_correct = [given_part]

                    # Handle compound surnames
                    if "-" in surname_part or " " in surname_part:
                        surname_tokens_correct = surname_part.replace("-", " ").split()
                    else:
                        surname_tokens_correct = [surname_part]

                    correct_parse = {
                        "surname_tokens": surname_tokens_correct,
                        "given_tokens": given_tokens_correct,
                    }

        if not correct_parse or not all_parses:
            continue

        # Create training example
        training_example = {
            "original_chinese_name": f"ACL_{original}",  # Mark as ACL-derived
            "romanized_name": original,
            "tokens": list(normalized_input.roman_tokens),
            "parses": all_parses,
        }

        # Ensure the correct parse is first in the list
        if correct_parse in all_parses:
            all_parses.remove(correct_parse)
            all_parses.insert(0, correct_parse)
        else:
            all_parses.insert(0, correct_parse)

        training_examples.append(training_example)

    print(f"Created {len(training_examples)} training examples from ACL data")
    return training_examples


def load_existing_training_data():
    """Load existing training data."""
    train_data = []
    test_data = []

    data_pkg = files("sinonym.data")
    train_res = data_pkg / "ml_parsing_train_split.json"
    test_res = data_pkg / "ml_parsing_test_split.json"

    if train_res.is_file():
        train_data = json.loads(train_res.read_text(encoding="utf-8"))

    if test_res.is_file():
        test_data = json.loads(test_res.read_text(encoding="utf-8"))

    print(f"Existing training data: {len(train_data)} train, {len(test_data)} test")
    return train_data, test_data


def main():
    print("Processing ACL 2025 Authors for ML Training")
    print("=" * 60)

    # Load ACL authors
    authors = load_acl_authors()

    # Filter for Chinese names
    chinese_names = filter_chinese_names(authors)

    if not chinese_names:
        print("No Chinese names found in ACL data!")
        return

    # Convert to training format
    detector = ChineseNameDetector()
    acl_training_examples = convert_to_training_format(chinese_names, detector)

    if not acl_training_examples:
        print("No valid training examples created from ACL data!")
        return

    # Load existing training data
    existing_train, existing_test = load_existing_training_data()

    # Add ACL examples to training data (not test data)
    # We want to train on ACL data but test on our original curated test set
    combined_train = existing_train + acl_training_examples

    print(f"\nCombined training data: {len(combined_train)} examples")
    print(f"  - Original: {len(existing_train)}")
    print(f"  - ACL added: {len(acl_training_examples)}")
    print(f"Test data unchanged: {len(existing_test)} examples")

    # Write outputs to a writable data directory in the package source
    try:
        import sinonym as _sin
        data_dir = Path(_sin.__file__).resolve().parent / "data"
    except Exception:
        data_dir = Path.cwd()
    data_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "ml_parsing_train_split.json"
    test_path = data_dir / "ml_parsing_test_split.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(combined_train, f, indent=2, ensure_ascii=False)

    print(f"\nSaved updated training data to: {train_path}")
    print(f"Training examples: {len(combined_train)}")

    # Also create a backup of ACL-only data for analysis
    acl_only_path = data_dir / "acl_training_examples.json"
    with open(acl_only_path, "w", encoding="utf-8") as f:
        json.dump(acl_training_examples, f, indent=2, ensure_ascii=False)

    print(f"Saved ACL-only training data to: {acl_only_path}")

    # Show some examples
    print("\nSample ACL training examples:")
    for i, example in enumerate(acl_training_examples[:5]):
        print(f"{i+1}. {example['romanized_name']}")
        correct_parse = example["parses"][0]
        print(f"   Correct parse: surname={correct_parse['surname_tokens']}, given={correct_parse['given_tokens']}")
        print(f"   Total parses: {len(example['parses'])}")


if __name__ == "__main__":
    main()
