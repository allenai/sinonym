# Sinonym Data Directory

This directory contains the data files and trained models used by the Sinonym library.

## Data Files

### Name Frequency Data
- **`familyname_orcid.csv`**: Chinese surnames with frequency data derived from ORCID records
- **`givenname_orcid.csv`**: Chinese given names with usage statistics from ORCID records

### Machine Learning Models

#### Chinese vs Japanese Name Classifier
- **`chinese_japanese_classifier.joblib`**: Pre-trained scikit-learn model for distinguishing Chinese from Japanese names using Chinese characters
- **`model_features.json`**: Feature information and metadata for the classifier

**Model Details:**
- **Purpose**: Classify all-Chinese character names (like "山田太郎") as Chinese vs Japanese
- **Accuracy**: 99.53% on test set of 263,409 names  
- **Training Data**: 1.3M names (1.14M Chinese, 172K Japanese from open corpora)
- **Features**: 5,000 character n-grams + 20 linguistic heuristic features
- **Algorithm**: Logistic Regression with TF-IDF character features
- **Use Case**: Fills gap where Japanese names in Chinese characters bypass romanization-based detection

**Model Features:**
- Character-level 1-3 gram TF-IDF features (5,000 features)
- Linguistic binary flags (20 features):
  - Japanese iteration mark (々)
  - Japanese/Chinese surname character patterns
  - Japanese/Chinese name ending patterns
  - Unique character indicators
  - Name length patterns
  - Character frequency patterns
  - Stroke complexity estimation

**Dependencies:**
- scikit-learn >= 1.7.1
- numpy >= 2.2.6  
- scipy >= 1.15.3
- joblib >= 1.5.1

**Training Date**: 2025-07-30

## Model Integration

The Chinese vs Japanese classifier is integrated into `sinonym.services.ethnicity.EthnicityClassificationService` and automatically activates for all-Chinese character inputs that pass initial ethnicity screening.

If ML dependencies are not available, the system gracefully falls back to the existing rule-based ethnicity classification without the enhanced Japanese detection capability.