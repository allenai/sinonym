# Sinonym Scripts Directory

This directory contains utility scripts for data generation, model training, and testing of the Sinonym library.

## Scripts Overview

### 1. `train_ml_classifier_for_chinese_vs_japanese.py` ✅ ACTIVE
**Purpose**: Train the machine learning classifier that distinguishes Chinese names from Japanese names when written in Chinese characters.

**Status**: ✅ **Successfully implemented and integrated**

**What it does**:
- Downloads Chinese (1.2M) and Japanese (180K) name corpora from GitHub
- Filters names to keep only those written in Chinese/Japanese characters (kanji)
- Trains a scikit-learn Pipeline with:
  - TF-IDF character n-gram features (1-3 grams, max 5000 features)
  - 20 linguistic heuristic features (Japanese markers, character patterns, etc.)
  - Logistic Regression classifier with balanced class weights
- Saves the trained model as `data/chinese_japanese_classifier.joblib`
- Achieves 99.5% accuracy on test data

**Dependencies**:
- scikit-learn, numpy, scipy, joblib
- `sinonym.ml_model_components.EnhancedHeuristicFlags` (custom feature extractor)

**Output**: 
- `data/chinese_japanese_classifier.skops` - The trained model used in production
- `data/model_features.json` - Feature vocabulary metadata

**Usage**:
```bash
python scripts/train_ml_classifier_for_chinese_vs_japanese.py
```

---

### 2. `generate_chinese_name_corpus_data.py` ❌ ABANDONED
**Purpose**: Generate training data for an ML-based name parsing disambiguation model.

**Status**: ❌ **Historical - Abandoned effort**

**What it was supposed to do**:
- Download 200K Chinese names from the Chinese Names Corpus
- Romanize Chinese names to pinyin (without tones)
- Generate all possible surname/given name parse candidates
- Create ground truth labels based on Chinese name structure rules
- Extract features for each parse (log probabilities, ranks, ratios)
- Save training data for an ML model to choose the best parse

**Why it was abandoned**:
- The ML parsing model "didn't work well" (as noted in code comments)
- The rule-based parsing system in `sinonym.services.parsing` works sufficiently well
- The complexity of training data generation and feature engineering didn't justify the marginal improvements

**Output files (still present but unused)**:
- `data/ml_parsing_training_data.json` - 199K training examples with parse candidates
- `data/ml_parsing_metadata.json` - Statistics about the training data

---

### 3. `generate_acl_data.py` ❌ ABANDONED
**Purpose**: Process ACL 2025 conference authors to create additional training examples for the parsing model.

**Status**: ❌ **Historical - Part of abandoned ML parsing effort**

**What it does**:
- Loads author names from `data/acl_2025_authors.txt`
- Uses the ChineseNameDetector to identify Chinese names
- Converts ACL format names (Given Surname) to training examples
- Generates parse candidates with features for ML training

**Why it exists**:
- Attempted to augment the ML parsing training data with real academic names
- ACL authors represent a different distribution (romanized, Western ordering)
- Was meant to improve the never-implemented parsing model

**Output**:
- `data/acl_training_examples.json` - Training examples from ACL authors
- Would have updated `ml_parsing_train_split.json` (file doesn't exist)

---

## Summary

### Active Scripts
- **`train_ml_classifier_for_chinese_vs_japanese.py`** - The only actively used script that trains the Chinese vs Japanese classifier

### Historical/Abandoned Scripts  
- **`generate_chinese_name_corpus_data.py`** - Abandoned ML parsing model data generation
- **`generate_acl_data.py`** - Abandoned ACL author data processing for ML parsing

## Data Flow

```
Chinese/Japanese Corpora (GitHub)
           ↓
train_ml_classifier_for_chinese_vs_japanese.py
           ↓
chinese_japanese_classifier.joblib ← [ACTIVELY USED BY LIBRARY]
           +
model_features.json


Chinese Names Corpus (GitHub)
           ↓
generate_chinese_name_corpus_data.py
           ↓
ml_parsing_training_data.json ← [ABANDONED, NOT USED]
           +
ml_parsing_metadata.json


ACL 2025 Authors
           ↓
generate_acl_data.py
           ↓
acl_training_examples.json ← [ABANDONED, NOT USED]
```

## Notes

The scripts demonstrate two different ML efforts:
1. **Successful**: Chinese vs Japanese classification for names written in Chinese characters
2. **Abandoned**: ML-based parsing disambiguation to choose between multiple valid name parses

The abandoned parsing model efforts remain in the codebase for historical reference but are not integrated into the library. The rule-based parsing in `sinonym.services.parsing.NameParsingService` handles name parsing instead.