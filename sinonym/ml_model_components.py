"""
ML Model Components for Chinese vs Japanese Name Classification

This module contains the custom transformer classes needed to load the
pre-trained ML model. These components must be importable when loading
the model with skops (preferred) or joblib (legacy fallback).
"""


import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

# Replicate the linguistic resources from the training script
CN_SURNAME_CHARS = {
    "王", "李", "张", "刘", "陈", "杨", "黄", "赵", "周", "吴",
    "徐", "孙", "朱", "马", "胡", "郭", "林", "何", "高", "梁",
    "郑", "罗", "宋", "谢", "唐", "韩", "曹", "许", "邓", "萧",
}

JP_SURNAME_CHARS = {
    "田", "中", "山", "本", "木", "村", "井", "川", "原", "藤",
    "野", "池", "石", "松", "竹", "林", "森", "東", "西", "北",
    "南", "上", "下", "大", "小", "高", "長", "新", "古", "佐",
}

JP_NAME_ENDINGS = {
    "子", "美", "也", "郎", "男", "之", "哉", "奈", "菜", "里",
    "佳", "香", "恵", "愛", "花", "夏", "春", "秋", "冬", "雪",
    "月", "星", "海", "空", "光", "希", "未", "真", "純", "清",
}

CN_NAME_ENDINGS = {
    "华", "明", "伟", "强", "军", "平", "勇", "杰", "涛", "波",
    "磊", "鹏", "辉", "刚", "超", "飞", "龙", "凤", "霞", "红",
    "玲", "丽", "娟", "芳", "燕", "静", "敏", "慧", "兰", "梅",
}

ITERATION_MARK = "々"

JP_UNIQUE_CHARS = {
    "辺", "沢", "浜", "栄", "竜", "礼", "稲", "彦", "蔵", "衛",
    "介", "助", "郎", "丸", "丞", "斎", "斉", "桜", "櫻",
}

CN_SIMPLIFIED_CHARS = {
    "赵", "刘", "陈", "邓", "关", "兰", "乔", "许", "闫", "贾",
    "钱", "孔", "白", "崔", "康", "史", "顾", "侯", "邵", "孟",
}

CN_FREQUENT_CHARS = {
    "国", "民", "建", "文", "志", "忠", "义", "礼", "智", "信",
    "仁", "勇", "才", "德", "宝", "福", "寿", "康", "安", "宁",
}

JP_FREQUENT_CHARS = {
    "雄", "雅", "正", "直", "克", "修", "治", "和", "昭", "博",
    "弘", "宏", "広", "寛", "豊", "富", "貴", "尊", "敬", "慶",
}


class EnhancedHeuristicFlags(BaseEstimator, TransformerMixin):
    """Enhanced transformer with improved linguistic features for Chinese vs Japanese classification."""

    def __init__(self):
        self.flag_names = [
            # Original features
            "jp_iter_mark",
            "jp_surname_chars",
            "cn_surname_chars",
            "jp_name_endings",
            "cn_name_endings",
            "jp_unique_chars",
            "cn_simplified_chars",
            "len_eq2",
            "len_eq3",
            "len_ge4",

            # Enhanced features
            "jp_frequent_chars",
            "cn_frequent_chars",
            "surname_jp_pattern",
            "surname_cn_pattern",
            "given_jp_pattern",
            "given_cn_pattern",
            "jp_ending_ratio",
            "cn_ending_ratio",
            "char_diversity",
            "avg_char_strokes",
        ]

    def fit(self, X, y=None):
        """Fit method (no-op for this transformer)."""
        return self

    def transform(self, X):
        """Transform names into heuristic feature vectors."""
        rows, cols, data = [], [], []

        for i, name in enumerate(X):
            if len(name) < 2:
                continue

            # Basic character analysis
            chars = list(name)
            first_char = chars[0]
            last_char = chars[-1]

            # Calculate features
            features = [
                # Original features
                ITERATION_MARK in name,
                any(c in JP_SURNAME_CHARS for c in chars[:2]),  # First 2 chars
                any(c in CN_SURNAME_CHARS for c in chars[:2]),
                last_char in JP_NAME_ENDINGS,
                last_char in CN_NAME_ENDINGS,
                any(c in JP_UNIQUE_CHARS for c in chars),
                any(c in CN_SIMPLIFIED_CHARS for c in chars),
                len(name) == 2,
                len(name) == 3,
                len(name) >= 4,

                # Enhanced features
                sum(1 for c in chars if c in JP_FREQUENT_CHARS) > 0,
                sum(1 for c in chars if c in CN_FREQUENT_CHARS) > 0,
                first_char in JP_SURNAME_CHARS,
                first_char in CN_SURNAME_CHARS,
                any(c in JP_NAME_ENDINGS for c in chars[1:]),  # Given name area
                any(c in CN_NAME_ENDINGS for c in chars[1:]),
                sum(1 for c in chars if c in JP_NAME_ENDINGS) / len(chars),  # Ratio
                sum(1 for c in chars if c in CN_NAME_ENDINGS) / len(chars),
                len(set(chars)) / len(chars),  # Character diversity
                self._estimate_stroke_complexity(chars),
            ]

            for j, val in enumerate(features):
                if isinstance(val, bool) and val:
                    rows.append(i)
                    cols.append(j)
                    data.append(1)
                elif isinstance(val, (int, float)) and val > 0:
                    rows.append(i)
                    cols.append(j)
                    data.append(float(val))

        n_samples = len(X)
        n_features = len(self.flag_names)
        return sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_features))

    def _estimate_stroke_complexity(self, chars):
        """Rough estimate of average stroke complexity."""
        complexity_scores = []
        for char in chars:
            char_code = ord(char)
            if 0x4E00 <= char_code <= 0x9FFF:  # CJK Unified Ideographs
                # Simple heuristic: higher unicode values tend to be more complex
                complexity = (char_code - 0x4E00) / (0x9FFF - 0x4E00)
                complexity_scores.append(complexity)

        return np.mean(complexity_scores) if complexity_scores else 0.0
