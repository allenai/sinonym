# Sinonym Data Directory

This directory contains the data files and trained models used by the Sinonym library.

## Data Files

### Name Frequency Data
- **`familyname_orcid.csv`**: Chinese surnames with frequency data (parts per million) derived from ORCID records. Contains the most common Chinese surnames like 王, 李, 张, etc., with their frequency statistics.
- **`givenname_orcid.csv`**: Chinese given name characters with usage statistics from ORCID records. Includes character, pinyin romanization, and frequency (ppm).
ss
### Chinese vs Japanese Name Classifier 
- **`chinese_japanese_classifier.joblib`**: Pre-trained scikit-learn model for distinguishing Chinese from Japanese names when written in Chinese characters. 
- **`model_features.json`**: Feature vocabulary for the Chinese vs Japanese classifier, containing character n-grams and linguistic patterns used by the model.

**Model Details:**
- **Purpose**: Classify all-Chinese character names (like "山田太郎") as Chinese vs Japanese
- **Accuracy**: 99.53% on test set of 263,409 names  
- **Training Data**: 1.3M names (1.14M Chinese, 172K Japanese from open corpora)
- **Features**: 5,000 character n-grams + 20 linguistic heuristic features
- **Algorithm**: Logistic Regression with TF-IDF character features
- **Use Case**: Fills gap where Japanese names in Chinese characters bypass romanization-based detection
- **Status**: **ACTIVE - Integrated in `sinonym.services.ml_japanese_classifier.MLJapaneseClassifier`**

**Model Features:**
- Character-level 1-3 gram TF-IDF features (5,000 features)
- Linguistic binary flags (20 features):
  - Japanese iteration mark (々)
  - Japanese/Chinese surname character patterns
  - Japanese/Chinese name ending patterns
  - Unique character indicators
  - Name length patterns
  - Character frequency patterns
s
**Training Date**: 2025-07-30

## Model Integration

The Chinese vs Japanese classifier is integrated into `sinonym.services.ethnicity.EthnicityClassificationService` and automatically activates for all-Chinese character inputs that pass initial ethnicity screening.

If ML dependencies are not available, the system gracefully falls back to the existing rule-based ethnicity classification without the enhanced Japanese detection capability.

## Data Generation Scripts
s
Scripts in the `scripts/` directory are used to generate and process these data files:
- `train_ml_classifier_for_chinese_vs_japanese.py`: Trains the Chinese vs Japanese classifier model

## Summary of ML Efforts

1. **Chinese vs Japanese Classifier** (✅ IMPLEMENTED & ACTIVE):
   - Files: `chinese_japanese_classifier.skops`, `model_features.json`
   - Achieves 99.5% accuracy