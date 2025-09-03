#!/usr/bin/env python3
"""
Train Chinese vs Japanese Name Classifier

This script trains and saves the ML model with the correct package structure
so it can be loaded properly in the sinonym package.

The training is already done. This is just here for reference and to ensure
the model can be retrained if needed.
"""

import random
import unicodedata
from pathlib import Path

import joblib
from skops.io import dump as skops_dump
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from sinonym.ml_model_components import EnhancedHeuristicFlags
from importlib.resources import files

# Data sources - Apache 2.0
CN_URL = (
    "https://github.com/wainshine/Chinese-Names-Corpus/"
    "raw/refs/heads/master/Chinese_Names_Corpus/"
    "Chinese_Names_Corpus%EF%BC%88120W%EF%BC%89.txt"
)
JP_URL = (
    "https://github.com/wainshine/Chinese-Names-Corpus/"
    "raw/refs/heads/master/Japanese_Names_Corpus/"
    "Japanese_Names_Corpus%EF%BC%8818W%EF%BC%89.txt"
)


def download_names(url: str, sample_size: int | None) -> list[str]:
    """Download a UTF-8 text file and return a list of names."""
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


def is_all_kanji(name: str) -> bool:
    """Check if name contains only Chinese/Japanese characters."""
    return all(unicodedata.name(ch, "").startswith(("CJK UNIFIED", "CJK COMPATIBILITY")) for ch in name)


def build_pipeline() -> Pipeline:
    """Build the ML classification pipeline."""
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        max_features=5000,
    )

    # Use the EnhancedHeuristicFlags from the package
    flags = EnhancedHeuristicFlags()

    combined = FeatureUnion(
        [
            ("chars", char_vectorizer),
            ("flags", flags),
        ],
    )

    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        class_weight="balanced",
        C=0.1,
        random_state=42,
    )

    return Pipeline([("features", combined), ("clf", clf)])


def main():
    """Train and save the ML model."""
    random.seed(42)

    print("Training Chinese vs Japanese Name Classifier for Sinonym")
    print("=" * 60)

    # Use a smaller sample for quick training in development
    SAMPLE_SIZE = None  # Put in a small number for faster training

    # Download data
    cn_names = download_names(CN_URL, SAMPLE_SIZE)
    jp_names = download_names(JP_URL, SAMPLE_SIZE)

    # Filter to kanji-only names
    cn_names = [n for n in cn_names if is_all_kanji(n)]
    jp_names = [n for n in jp_names if is_all_kanji(n)]

    print("\nFiltered dataset:")
    print(f"Chinese names: {len(cn_names):,}")
    print(f"Japanese names: {len(jp_names):,}")

    # Prepare training data
    X = cn_names + jp_names
    y = ["cn"] * len(cn_names) + ["jp"] * len(jp_names)

    print(f"\nTotal dataset: {len(X):,} samples")

    # Build and train model
    print("\nTraining model...")
    pipeline = build_pipeline()
    pipeline.fit(X, y)

    # Test a few examples
    demo_names = ["山田太郎", "陈锦壬", "田中花子", "李建国"]
    predictions = pipeline.predict(demo_names)
    probabilities = pipeline.predict_proba(demo_names)

    print("\nDemo predictions:")
    for name, pred, prob in zip(demo_names, predictions, probabilities, strict=False):
        confidence = max(prob)
        print(f"{name:10} → {pred.upper()} (confidence: {confidence:.3f})")

    # Choose a writable data directory within the package source tree
    import sinonym as _sin

    data_dir = Path(_sin.__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    skops_path = data_dir / "chinese_japanese_classifier.skops"
    joblib_path = data_dir / "chinese_japanese_classifier.joblib"

    print(f"\nSaving SKOPS model to {skops_path}...")
    skops_dump(pipeline, skops_path)

    print(f"Saving legacy joblib model to {joblib_path}...")
    joblib.dump(pipeline, joblib_path)

    print("✓ Model training completed and saved successfully!")


if __name__ == "__main__":
    main()
