"""
Performance Tests for Chinese Name Detector

This module contains performance benchmarks to ensure the ChineseNameDetector
maintains fast processing speeds suitable for production use.

Key Performance Requirements:
- Average processing time < 1ms per name (sub-millisecond requirement)
- Processing rate > 1000 names/second for diverse data
- Consistent performance across different name types

The test includes both diverse data (minimal cache benefit) and cached data
to measure real-world vs optimal performance scenarios.
"""

import random
import sys
import time
from pathlib import Path

import pytest

# Add the parent directory to path to import sinonym
sys.path.insert(0, str(Path(__file__).parent.parent))

from sinonym import ChineseNameDetector

# Performance thresholds as constants
MAX_MICROSECONDS_PER_NAME = 1000  # 1ms requirement
MIN_DIVERSE_NAMES_PER_SECOND = 9000  # 9,000 names/second for diverse data
MAX_PERFORMANCE_VARIANCE = 0.7  # 70% threshold


class TestChineseNameDetectorPerformance:
    """Performance test suite for ChineseNameDetector."""

    @pytest.fixture(scope="class")
    def detector(self):
        """Create a detector instance for performance testing."""
        return ChineseNameDetector()

    def generate_test_names(self, detector: ChineseNameDetector, count: int) -> list[str]:
        """Generate diverse Chinese and non-Chinese names for realistic testing."""

        # Use surnames from our data structures
        chinese_surnames = list(detector._data.surnames)[:200]  # Top 200 surnames
        chinese_givens = list(detector._data.given_names)[:500]  # Top 500 given names

        # Common non-Chinese patterns
        western_first = [
            "John",
            "Mary",
            "David",
            "Sarah",
            "Michael",
            "Lisa",
            "James",
            "Jennifer",
            "Robert",
            "Jessica",
            "William",
            "Ashley",
            "Christopher",
            "Amanda",
        ]
        western_last = [
            "Smith",
            "Johnson",
            "Williams",
            "Brown",
            "Jones",
            "Garcia",
            "Miller",
            "Davis",
            "Rodriguez",
            "Martinez",
            "Hernandez",
            "Lopez",
            "Gonzalez",
        ]

        korean_names = ["Kim Min Soo", "Park Ji Hoon", "Lee Soo Jin", "Choi Young Hee", "Jung Hye Won"]
        japanese_names = ["Tanaka Hiroshi", "Suzuki Yuki", "Yamamoto Akira", "Sato Kenji"]

        names = []

        for _ in range(count):
            choice = random.random()

            if choice < 0.6:  # 60% Chinese names
                surname = random.choice(chinese_surnames).title()
                if random.random() < 0.7:  # 70% single given name
                    given = random.choice(chinese_givens).title()
                else:  # 30% compound given name
                    given1 = random.choice(chinese_givens).title()
                    given2 = random.choice(chinese_givens).title()
                    given = f"{given1}-{given2}" if random.random() < 0.5 else f"{given1}{given2}"

                # Mix surname-first and surname-last orders
                if random.random() < 0.6:
                    names.append(f"{given} {surname}")
                else:
                    names.append(f"{surname} {given}")

            elif choice < 0.8:  # 20% Western names
                first = random.choice(western_first)
                last = random.choice(western_last)
                names.append(f"{first} {last}")

            elif choice < 0.9:  # 10% Korean names
                names.append(random.choice(korean_names))

            else:  # 10% Japanese names
                names.append(random.choice(japanese_names))

        return names

    def test_performance_diverse_data(self, detector):
        """
        Test processing performance with diverse data (realistic scenario).

        This test ensures the detector maintains sub-millisecond average processing
        time and can handle at least 1000 names per second with diverse input.
        """
        # Generate diverse test data
        test_names = self.generate_test_names(detector, 1000)

        print(f"\nTesting performance with {len(test_names)} diverse names...")

        # Measure processing time
        start_time = time.perf_counter()
        results = []
        for name in test_names:
            result = detector.is_chinese_name(name)
            results.append(result)
        end_time = time.perf_counter()

        # Calculate performance metrics
        total_time = end_time - start_time
        names_per_second = len(test_names) / total_time
        microseconds_per_name = (total_time / len(test_names)) * 1_000_000

        print(f"Diverse data: {len(test_names)} names in {total_time:.3f}s")
        print(f"Rate: {names_per_second:.0f} names/second")
        print(f"Time per name: {microseconds_per_name:.1f} microseconds")

        # Performance assertions - ensure it's "really fast"
        assert (
            microseconds_per_name < MAX_MICROSECONDS_PER_NAME
        ), f"Average processing time {microseconds_per_name:.1f} μs exceeds 1ms requirement"
        assert names_per_second > MIN_DIVERSE_NAMES_PER_SECOND, (
            f"Processing rate {names_per_second:.0f} names/sec is below "
            f"{MIN_DIVERSE_NAMES_PER_SECOND} names/sec requirement"
        )

        # Verify all names were processed
        assert len(results) == len(test_names), "Not all names were processed"

    def test_performance_consistency(self, detector):
        """
        Test that performance is consistent across multiple runs.

        This test ensures there are no significant performance regressions
        or inconsistencies in processing speed.
        """
        test_names = ["Zhang Wei", "John Smith", "Kim Min Soo", "Liu Dehua"] * 50  # 200 names
        run_times = []

        # Run the test multiple times
        for _run in range(5):
            start_time = time.perf_counter()
            for name in test_names:
                detector.is_chinese_name(name)
            end_time = time.perf_counter()
            run_times.append(end_time - start_time)

        # Calculate statistics
        avg_time = sum(run_times) / len(run_times)
        min_time = min(run_times)
        max_time = max(run_times)
        avg_rate = len(test_names) / avg_time
        avg_time_per_name = (avg_time / len(test_names)) * 1_000_000

        print(f"\nConsistency Test (5 runs of {len(test_names)} names):")
        print(f"Average: {avg_rate:.0f} names/sec ({avg_time_per_name:.1f} μs/name)")
        print(f"Range: {len(test_names)/max_time:.0f} - {len(test_names)/min_time:.0f} names/sec")

        # Performance should be consistent (variance < 50%)
        variance_ratio = (max_time - min_time) / avg_time
        assert (
            variance_ratio < MAX_PERFORMANCE_VARIANCE
        ), f"Performance variance {variance_ratio:.2f} exceeds {MAX_PERFORMANCE_VARIANCE:.0%} threshold"
        assert (
            avg_rate > MIN_DIVERSE_NAMES_PER_SECOND
        ), f"Average rate {avg_rate:.0f} below minimum {MIN_DIVERSE_NAMES_PER_SECOND} names/sec"
