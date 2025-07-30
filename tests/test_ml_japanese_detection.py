"""
ML Japanese Detection Test Suite

This module tests the ML-based Japanese name detection for all-Chinese character names.
These tests verify that Japanese names written in Chinese characters (kanji) are correctly
identified and rejected, filling the gap in the original rule-based system.
"""

import sys
from pathlib import Path

# Add the parent directory to path to import sinonym
sys.path.insert(0, str(Path(__file__).parent.parent))

from sinonym import ChineseNameDetector

# Japanese names in Chinese characters that should be rejected by ML classifier
# These would slip through the original rule-based system because they get converted
# to pinyin (e.g., "山田太郎" → ["shan", "tian", "tai", "lang"]) which doesn't match
# Japanese surname patterns like "yamada"
ML_JAPANESE_TEST_CASES = [
    # Classic Japanese names from our training demo
    "山田太郎",  # Yamada Taro - classic Japanese name
    "田中花子",  # Tanaka Hanako - classic Japanese female name
    "佐藤花子",  # Sato Hanako - most common Japanese surname
    "藤原愛子",  # Fujiwara Aiko - noble Japanese surname
    "松下幸之助",  # Matsushita Konosuke - famous businessman
    "安倍晋三",  # Abe Shinzo - former Prime Minister
    "小泉纯一郎",  # Koizumi Junichiro - former Prime Minister

    # Additional Japanese names from training data error analysis
    "井普源",  # Training data example that was challenging
    "山展源",  # Training data example that was challenging
    "田相吉",  # Training data example that was challenging
    "田弘光",  # Training data example that was challenging
    "山一鼎",  # Training data example that was challenging
    "田一博",  # Training data example that was challenging

    # Common Japanese surname patterns
    "田中健一",  # Tanaka Kenichi
    "山本太郎",  # Yamamoto Taro
    "鈴木一郎",  # Suzuki Ichiro
    "高橋美咲",  # Takahashi Misaki
    "伊藤博文",  # Ito Hirobumi
    "渡邊愛子",  # Watanabe Aiko
    "加藤浩一",  # Kato Koichi
    "本田勝",  # Honda Masaru
    "藤原哲也",  # Fujiwara Tetsuya

    # Japanese names with characteristic endings
    "中村美香",  # Nakamura Mika - "香" common in Japanese female names
    "小林花子",  # Kobayashi Hanako - "子" classic Japanese female ending
    "大野一郎",  # Ono Ichiro - "一郎" classic Japanese male ending
    "森田直子",  # Morita Naoko - "直子" common Japanese name
    "上田雅子",  # Ueda Masako - "雅子" common Japanese name

    # Japanese compound surnames
    "松井秀喜",  # Matsui Hideki - famous baseball player
    "長谷川潤",  # Hasegawa Jun - model/actress
    "竹内結子",  # Takeuchi Yuko - actress
]

# Chinese names that should NOT be rejected (control group)
# These should pass through the ML classifier as Chinese
ML_CHINESE_CONTROL_CASES = [
    # Classic Chinese names from our training demo
    "陈锦壬",  # Chen Jinren - training demo Chinese name
    "赵丽颖",  # Zhao Liying - actress
    "林森",  # Lin Sen - simple Chinese name
    "王小明",  # Wang Xiaoming - common Chinese name
    "李建国",  # Li Jianguo - patriotic Chinese name
    "张伟明",  # Zhang Weiming - common Chinese name
    "刘德华",  # Liu Dehua - famous actor
    "毛泽东",  # Mao Zedong - historical figure

    # Chinese names that might be challenging for ML classifier
    "王田",  # Wang Tian - could look Japanese due to "田"
    "李山",  # Li Shan - could look Japanese due to "山"
    "陈中",  # Chen Zhong - common Chinese pattern
    "张华",  # Zhang Hua - very common Chinese name
    "刘伟",  # Liu Wei - very common Chinese name

    # Chinese names with characters that appear in Japanese names
    "林美",  # Lin Mei - "美" appears in Japanese names but this is Chinese
    "王子",  # Wang Zi - "子" appears in Japanese names but this is Chinese
    "李一",  # Li Yi - "一" appears in Japanese names but this is Chinese
]


def test_ml_japanese_detection():
    """Test that ML classifier correctly identifies Japanese names in Chinese characters."""
    detector = ChineseNameDetector()

    passed = 0
    failed = 0

    print("Testing ML Japanese detection...")

    # Test Japanese names (should be rejected)
    for japanese_name in ML_JAPANESE_TEST_CASES:
        result = detector.is_chinese_name(japanese_name)
        if result.success is False:
            passed += 1
            print(f"✓ PASS: '{japanese_name}' correctly rejected as Japanese")
        else:
            failed += 1
            print(f"✗ FAIL: '{japanese_name}' incorrectly accepted as Chinese: {result.result}")

    print(f"\nJapanese name detection: {passed} passed, {failed} failed out of {len(ML_JAPANESE_TEST_CASES)} tests")

    # Test Chinese names (should be accepted - control group)
    chinese_passed = 0
    chinese_failed = 0

    for chinese_name in ML_CHINESE_CONTROL_CASES:
        result = detector.is_chinese_name(chinese_name)
        if result.success is True:
            chinese_passed += 1
            print(f"✓ PASS: '{chinese_name}' correctly accepted as Chinese: {result.result}")
        else:
            chinese_failed += 1
            print(f"✗ FAIL: '{chinese_name}' incorrectly rejected: {result.error_message}")

    print(f"\nChinese name control: {chinese_passed} passed, {chinese_failed} failed out of {len(ML_CHINESE_CONTROL_CASES)} tests")

    # Overall assessment
    total_tests = len(ML_JAPANESE_TEST_CASES) + len(ML_CHINESE_CONTROL_CASES)
    total_passed = passed + chinese_passed
    total_failed = failed + chinese_failed

    accuracy = total_passed / total_tests if total_tests > 0 else 0
    print(f"\nOverall ML classifier integration: {total_passed}/{total_tests} ({accuracy:.1%} accuracy)")

    # We expect high accuracy but allow for some edge cases
    # The original 99.5% model accuracy may be slightly lower in integration due to confidence thresholds
    min_acceptable_accuracy = 0.90  # 90% minimum acceptable

    if accuracy >= min_acceptable_accuracy:
        print(f"🎉 ML integration test PASSED! Accuracy {accuracy:.1%} meets {min_acceptable_accuracy:.1%} threshold")
    else:
        print(f"❌ ML integration test FAILED! Accuracy {accuracy:.1%} below {min_acceptable_accuracy:.1%} threshold")

    # For automated testing, we'll be less strict since we use high confidence threshold
    assert failed <= len(ML_JAPANESE_TEST_CASES) * 0.2, f"Too many Japanese names missed: {failed}/{len(ML_JAPANESE_TEST_CASES)}"
    assert chinese_failed <= len(ML_CHINESE_CONTROL_CASES) * 0.1, f"Too many Chinese names rejected: {chinese_failed}/{len(ML_CHINESE_CONTROL_CASES)}"

    return {
        "japanese_detection_rate": passed / len(ML_JAPANESE_TEST_CASES),
        "chinese_preservation_rate": chinese_passed / len(ML_CHINESE_CONTROL_CASES),
        "overall_accuracy": accuracy,
    }


def test_ml_classifier_availability():
    """Test that ML classifier is available and working."""
    detector = ChineseNameDetector()

    # Check if ML classifier is available
    ml_classifier = detector._ethnicity_service._ml_classifier

    if ml_classifier.is_available():
        print("✓ ML classifier is available and loaded successfully")

        # Test a simple classification
        details = ml_classifier.get_classification_details("山田太郎")
        print(f"✓ ML classifier working: {details}")

        assert details.get("available", False), "ML classifier reports not available"
        assert "prediction" in details, "ML classifier not returning predictions"

    else:
        print("⚠️  ML classifier not available - possibly missing dependencies")
        print("   This is acceptable for environments without scikit-learn")

        # Test graceful fallback
        result = detector.is_chinese_name("山田太郎")
        print(f"   Fallback behavior: {result}")


if __name__ == "__main__":
    print("="*60)
    print("ML JAPANESE DETECTION TEST SUITE")
    print("="*60)

    # Test ML classifier availability first
    test_ml_classifier_availability()

    print("\n" + "-"*60)

    # Test ML detection performance
    results = test_ml_japanese_detection()

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Japanese Detection Rate: {results['japanese_detection_rate']:.1%}")
    print(f"Chinese Preservation Rate: {results['chinese_preservation_rate']:.1%}")
    print(f"Overall Accuracy: {results['overall_accuracy']:.1%}")
    print("\nML Japanese detection test completed!")
