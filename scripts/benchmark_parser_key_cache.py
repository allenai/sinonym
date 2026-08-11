#!/usr/bin/env python3
"""Benchmark normalization with and without the parser-key cache.

The uncached variant bypasses only ``SurnameResolver``'s bounded cache, so both
measurements exercise the same checkout, data, normalization, and scoring code.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import statistics
import time
from typing import TYPE_CHECKING
from unittest.mock import patch

from sinonym import ChineseNameDetector
from sinonym.services.name_lookup import SurnameResolver

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sinonym.coretypes import ParseResult

_SURNAMES = "Zhang Wang Li Chen Liu Yang Huang Zhao Wu Zhou Xu Sun Ma Zhu Hu Guo He Gao Lin Luo".split()  # noqa: SIM905
_GIVEN_NAMES = "Wei Ming Hua Fang Lei Jun Jing Qiang Yan Tao Jie Hui Yong Xin Ping Hong Yu Lin Feng Bo".split()  # noqa: SIM905
_MAX_NAMES = len(_SURNAMES) * len(_GIVEN_NAMES) ** 2


def _parse_args() -> argparse.Namespace:
    """Parse and validate benchmark options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", required=True, type=int, help=f"Names to score (1-{_MAX_NAMES}).")
    parser.add_argument("--repeats", default=7, type=int, help="Timed repetitions per variant.")
    args = parser.parse_args()
    if not 1 <= args.limit <= _MAX_NAMES:
        parser.error(f"--limit must be between 1 and {_MAX_NAMES}")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    return args


def _build_names(limit: int) -> list[str]:
    """Build a deterministic workload of common Chinese romanized names."""
    combinations = itertools.product(_SURNAMES, _GIVEN_NAMES, _GIVEN_NAMES)
    return [f"{surname} {given} {middle}" for surname, given, middle in itertools.islice(combinations, limit)]


def _uncached_parser_key(self: SurnameResolver, surname_tokens: Sequence[str]) -> str:
    """Reproduce parser-key derivation while bypassing only its new cache."""
    self._require_tokens(surname_tokens)
    return self._derive_parser_key(tuple(surname_tokens))


def _measure(names: list[str], repeats: int) -> tuple[list[float], list[ParseResult]]:
    """Measure repeated normalization and ensure every repetition is identical."""
    detector = ChineseNameDetector()
    timings: list[float] = []
    expected: list[ParseResult] | None = None
    for _ in range(repeats):
        gc.collect()
        gc_enabled = gc.isenabled()
        if gc_enabled:
            gc.disable()
        try:
            start = time.perf_counter()
            observed = [detector.normalize_name(name) for name in names]
            timings.append(time.perf_counter() - start)
        finally:
            if gc_enabled:
                gc.enable()
        if expected is None:
            expected = observed
        elif observed != expected:
            message = "normalization output changed between benchmark repetitions"
            raise RuntimeError(message)
    assert expected is not None
    return timings, expected


def main() -> None:
    """Run the cache ablation and report median elapsed time and speedup."""
    args = _parse_args()
    names = _build_names(args.limit)

    with patch.object(SurnameResolver, "_parser_key", _uncached_parser_key):
        before_timings, before_outputs = _measure(names, args.repeats)
    after_timings, after_outputs = _measure(names, args.repeats)

    if before_outputs != after_outputs:
        message = "cached and uncached normalization outputs differ"
        raise RuntimeError(message)

    before_median = statistics.median(before_timings)
    after_median = statistics.median(after_timings)
    print(f"names={len(names)} repeats={args.repeats} output_parity=true")
    print("before_seconds=" + ",".join(f"{timing:.6f}" for timing in before_timings))
    print("after_seconds=" + ",".join(f"{timing:.6f}" for timing in after_timings))
    print(f"before_median_seconds={before_median:.6f}")
    print(f"after_median_seconds={after_median:.6f}")
    print(f"throughput_speedup={before_median / after_median:.3f}x")


if __name__ == "__main__":
    main()
