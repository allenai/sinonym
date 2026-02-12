#!/usr/bin/env python3
"""
Hotspot share profiler for sinonym.

Runs one cProfile pass on deterministic warmed workload and reports
top functions/modules by internal time share.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import os
from collections import defaultdict

from sinonym.detector import ChineseNameDetector
from tests.test_performance import TestChineseNameDetectorPerformance


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile hotspot shares for sinonym.")
    parser.add_argument("--names", type=int, default=3000, help="Number of deterministic test names.")
    parser.add_argument("--warmup", type=int, default=3000, help="Number of warm-up names before profiling.")
    parser.add_argument("--top-functions", type=int, default=20, help="Top function rows to print.")
    parser.add_argument("--top-modules", type=int, default=12, help="Top module rows to print.")
    parser.add_argument(
        "--sinonym-only",
        action="store_true",
        help="Restrict function table to files containing '/sinonym/' or '\\sinonym\\'.",
    )
    return parser.parse_args()


def _is_sinonym_filename(filename: str) -> bool:
    path = filename.replace("\\", "/").lower()
    return "/sinonym/" in path


def _short_label(filename: str, function_name: str) -> str:
    path = filename.replace("\\", "/")
    parts = path.split("/")
    if "sinonym" in parts:
        idx = parts.index("sinonym")
        short_path = "/".join(parts[idx:])
    else:
        short_path = parts[-1]
    return f"{short_path}:{function_name}"


def main() -> int:  # noqa: PLR0915
    args = _parse_args()

    # Keep process-level settings explicit for reproducibility.
    os.environ.setdefault("PYTHONHASHSEED", "42")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    detector = ChineseNameDetector()
    helper = TestChineseNameDetectorPerformance()
    names = helper.generate_test_names(detector, args.names)

    warmup = min(args.warmup, len(names))
    for name in names[:warmup]:
        detector.normalize_name(name)

    gc.collect()
    profiler = cProfile.Profile()
    profiler.enable()
    for name in names:
        detector.normalize_name(name)
    profiler.disable()

    stats_dict = profiler.getstats()

    function_rows: list[tuple[float, float, int, str, str]] = []
    module_tottime: dict[str, float] = defaultdict(float)

    total_tottime = 0.0
    for entry in stats_dict:
        code = entry.code
        if not hasattr(code, "co_filename"):
            continue

        filename = code.co_filename
        function_name = code.co_name
        callcount = entry.callcount
        tottime = float(entry.inlinetime)
        cumtime = float(entry.totaltime)

        if args.sinonym_only and not _is_sinonym_filename(filename):
            continue

        total_tottime += tottime
        label = _short_label(filename, function_name)
        function_rows.append((tottime, cumtime, callcount, filename, label))

        module_key = label.split(":", 1)[0]
        module_tottime[module_key] += tottime

    if total_tottime <= 0:
        print("No profiler samples captured.")
        return 1

    function_rows.sort(key=lambda row: row[0], reverse=True)
    module_rows = sorted(module_tottime.items(), key=lambda item: item[1], reverse=True)

    print("=" * 92)
    print("SINONYM HOTSPOT SHARE PROFILE")
    print("=" * 92)
    print(f"names={args.names} warmup={warmup} sinonym_only={args.sinonym_only}")
    print(f"total_tottime_seconds={total_tottime:.6f}")
    print()

    print("Top Functions by tottime")
    print("-" * 92)
    print(f"{'share%':>7} {'tottime':>10} {'cumtime':>10} {'calls':>10}  function")
    for tottime, cumtime, callcount, _filename, label in function_rows[: args.top_functions]:
        share = (tottime / total_tottime) * 100.0
        print(f"{share:7.2f} {tottime:10.4f} {cumtime:10.4f} {callcount:10d}  {label}")

    print()
    print("Top Modules by tottime")
    print("-" * 92)
    print(f"{'share%':>7} {'tottime':>10}  module")
    for module, tottime in module_rows[: args.top_modules]:
        share = (tottime / total_tottime) * 100.0
        print(f"{share:7.2f} {tottime:10.4f}  {module}")

    if args.sinonym_only:
        print()
        print("Note: shares are relative to sinonym-only tottime.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
