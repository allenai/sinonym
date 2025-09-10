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
MIN_DIVERSE_NAMES_PER_SECOND = 7500  # 7,500 names/second for diverse data


class TestChineseNameDetectorPerformance:
    """Performance test suite for ChineseNameDetector."""

    @pytest.fixture(scope="class")
    def detector(self):
        """Create a detector instance for performance testing."""
        return ChineseNameDetector()

    def generate_test_names(self, detector: ChineseNameDetector, count: int) -> list[str]:
        """Generate diverse Chinese and non-Chinese names for realistic testing.

        Returns exactly 'count' unique names in a deterministic way by setting random seed.
        """
        # Set deterministic seed for reproducible results
        random.seed(42)

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
            "Thomas",
            "Elizabeth",
            "Daniel",
            "Patricia",
            "Matthew",
            "Linda",
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
            "Wilson",
            "Anderson",
            "Thomas",
            "Taylor",
            "Moore",
            "Jackson",
            "Martin",
        ]

        korean_names = [
            "Kim Min Soo",
            "Park Ji Hoon",
            "Lee Soo Jin",
            "Choi Young Hee",
            "Jung Hye Won",
            "Lim Da Hye",
            "Song Ji Ho",
            "Kang Min Jung",
            "Yoon Seok Jin",
            "Han So Young",
        ]
        japanese_names = [
            "Tanaka Hiroshi",
            "Suzuki Yuki",
            "Yamamoto Akira",
            "Sato Kenji",
            "Watanabe Miki",
            "Ito Takeshi",
            "Nakamura Yuki",
            "Kobayashi Rei",
        ]

        names = set()  # Use set to ensure uniqueness
        attempts = 0
        max_attempts = count * 10  # Prevent infinite loops

        while len(names) < count and attempts < max_attempts:
            attempts += 1
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
                    name = f"{given} {surname}"
                else:
                    name = f"{surname} {given}"
                names.add(name)

            elif choice < 0.8:  # 20% Western names
                first = random.choice(western_first)
                last = random.choice(western_last)
                names.add(f"{first} {last}")

            elif choice < 0.9:  # 10% Korean names
                names.add(random.choice(korean_names))

            else:  # 10% Japanese names
                names.add(random.choice(japanese_names))

        # Convert to list and ensure deterministic ordering
        result = sorted(list(names))

        # If we couldn't generate enough unique names, pad with numbered variations
        if len(result) < count:
            base_names = result[:]
            while len(result) < count:
                base_name = base_names[len(result) % len(base_names)]
                suffix_num = (len(result) // len(base_names)) + 1
                result.append(f"{base_name} {suffix_num}")

        return result[:count]  # Ensure exactly 'count' names

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
