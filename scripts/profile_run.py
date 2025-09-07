import cProfile
import pstats
import time

from sinonym.detector import ChineseNameDetector
from tests.test_performance import TestChineseNameDetectorPerformance


def main():
    detector = ChineseNameDetector()
    helper = TestChineseNameDetectorPerformance()
    names = helper.generate_test_names(detector, 3000)

    pr = cProfile.Profile()
    pr.enable()
    start = time.perf_counter()
    for n in names:
        detector.is_chinese_name(n)
    end = time.perf_counter()
    pr.disable()

    print(f"Total time: {end-start:.3f}s for {len(names)} names")
    ps = pstats.Stats(pr).strip_dirs().sort_stats("tottime")
    ps.print_stats(25)


if __name__ == "__main__":
    main()

