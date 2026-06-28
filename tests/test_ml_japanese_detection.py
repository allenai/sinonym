"""
ML Japanese Detection Test Suite

This module tests the ML-based Japanese name detection for all-Chinese character names.
These tests verify that Japanese names written in Chinese characters (kanji) are correctly
identified and rejected, filling the gap in the original rule-based system.
"""

import pytest

from tests._case_assertions import assert_normalized_name, assert_rejected

# Japanese names in Chinese characters that should be rejected by ML classifier
# These would slip through the original rule-based system because they get converted
# to pinyin (e.g., "山田太郎" → ["shan", "tian", "tai", "lang"]) which doesn't match
# Japanese surname patterns like "yamada"
ML_JAPANESE_TEST_CASES = [
    # Classic Japanese names from our training demo
    ("山田太郎", (False, "should_be_rejected")),  # Yamada Taro - classic Japanese name
    ("田中花子", (False, "should_be_rejected")),  # Tanaka Hanako - classic Japanese female name
    ("佐藤花子", (False, "should_be_rejected")),  # Sato Hanako - most common Japanese surname
    ("藤原愛子", (False, "should_be_rejected")),  # Fujiwara Aiko - noble Japanese surname
    ("松下幸之助", (False, "should_be_rejected")),  # Matsushita Konosuke - famous businessman
    ("安倍晋三", (False, "should_be_rejected")),  # Abe Shinzo - former Prime Minister
    ("小泉纯一郎", (False, "should_be_rejected")),  # Koizumi Junichiro - former Prime Minister
    # Additional Japanese names from training data error analysis
    ("井普源", (False, "should_be_rejected")),  # Training data example that was challenging
    ("山展源", (False, "should_be_rejected")),  # Training data example that was challenging
    ("田相吉", (False, "should_be_rejected")),  # Training data example that was challenging
    ("田弘光", (False, "should_be_rejected")),  # Training data example that was challenging
    ("山一鼎", (False, "should_be_rejected")),  # Training data example that was challenging
    ("田一博", (False, "should_be_rejected")),  # Training data example that was challenging
    # Common Japanese surname patterns
    ("田中健一", (False, "should_be_rejected")),  # Tanaka Kenichi
    ("山本太郎", (False, "should_be_rejected")),  # Yamamoto Taro
    ("鈴木一郎", (False, "should_be_rejected")),  # Suzuki Ichiro
    ("高橋美咲", (False, "should_be_rejected")),  # Takahashi Misaki
    ("伊藤博文", (False, "should_be_rejected")),  # Ito Hirobumi
    ("渡邊愛子", (False, "should_be_rejected")),  # Watanabe Aiko
    ("加藤浩一", (False, "should_be_rejected")),  # Kato Koichi
    ("本田勝", (False, "should_be_rejected")),  # Honda Masaru
    ("藤原哲也", (False, "should_be_rejected")),  # Fujiwara Tetsuya
    # Japanese names with characteristic endings
    ("中村美香", (False, "should_be_rejected")),  # Nakamura Mika - "香" common in Japanese female names
    ("小林花子", (False, "should_be_rejected")),  # Kobayashi Hanako - "子" classic Japanese female ending
    ("大野一郎", (False, "should_be_rejected")),  # Ono Ichiro - "一郎" classic Japanese male ending
    ("橋本太郎", (False, "should_be_rejected")),  # Hashimoto Taro - classic Japanese name
    ("木村拓哉", (False, "should_be_rejected")),  # Kimura Takuya - famous actor
    ("福田康夫", (False, "should_be_rejected")),  # Fukuda Yasuo - former Prime Minister
    ("野田佳彦", (False, "should_be_rejected")),  # Noda Yoshihiko - former Prime Minister
    ("黒田東彦", (False, "should_be_rejected")),  # Kuroda Haruhiko - Bank of Japan governor
    ("岸田文雄", (False, "should_be_rejected")),  # Kishida Fumio - current Prime Minister
    ("菅義偉", (False, "should_be_rejected")),  # Suga Yoshihide - former Prime Minister
    ("原田 泰夫", (False, "should_be_rejected")),  # Harada Yasuo - spaced Kanji should still hit ML classifier
]

# Chinese control cases that should be accepted (for comparison)
ML_CHINESE_CONTROL_CASES = [
    ("张三", (True, "San Zhang")),  # Common Chinese test name
    ("李四", (True, "Si Li")),  # Common Chinese test name
    ("王五", (True, "Wu Wang")),  # Common Chinese test name
    ("赵六", (True, "Liu Zhao")),  # Common Chinese test name
    ("陈七", (True, "Qi Chen")),  # Common Chinese test name
    ("刘八", (True, "Ba Liu")),  # Common Chinese test name
    ("黄九", (True, "Jiu Huang")),  # Common Chinese test name
    ("周十", (True, "Shi Zhou")),  # Common Chinese test name
    ("黄 嘉平", (True, "Jia-Ping Huang")),  # Spaced Han Chinese name should remain accepted
    ("功华 张", (True, "Gong-Hua Zhang")),  # Spaced Han given-surname should use compact text for JP classification
]


@pytest.mark.parametrize(("input_name", "_expected_result"), ML_JAPANESE_TEST_CASES)
def test_ml_japanese_detection(detector, input_name, _expected_result):
    """Test that Japanese names in Chinese characters are correctly rejected by ML classifier."""
    assert_rejected(detector, input_name)


@pytest.mark.parametrize(("input_name", "expected_result"), ML_CHINESE_CONTROL_CASES)
def test_ml_chinese_control(detector, input_name, expected_result):
    """Test that Chinese control names are correctly accepted."""
    assert_normalized_name(detector, input_name, expected_result)
