# Sinonym Scripts

Utility scripts for benchmarking, profiling, testing, and model training.

## Active Scripts

### `check_test_status.py`
Runs the full test suite and reports individual test case failures with detailed diagnostics. Runs performance tests separately, then exits 0/1 based on whether the failure count and failure signatures exactly match the expected baseline configured in `scripts/check_test_status.py`. Regressions, unexpected failure signatures, or missing expected baseline signatures fail until the configured baseline is intentionally updated.

```bash
uv run python scripts/check_test_status.py
```

### `benchmark_stable.py`
Median-based performance benchmark gate. Spawns isolated worker subprocesses (fresh process per run) with controlled `PYTHONHASHSEED` and thread environment variables. Reports mean/median/stddev/CV of throughput and supports a `--min-median-names-per-sec` gate that exits non-zero on failure.

```bash
uv run python scripts/benchmark_stable.py --runs 5 --names 3000 --warmup 3000
uv run python scripts/benchmark_stable.py --runs 7 --min-median-names-per-sec 5000
```

### `profile_hotspots.py`
Hotspot time-share profiler. Warms caches on deterministic test names, runs one `cProfile` pass, then reports top functions and modules ranked by internal time (`tottime`) share. Use `--sinonym-only` to filter out third-party/stdlib noise.

```bash
uv run python scripts/profile_hotspots.py --names 3000 --warmup 3000 --sinonym-only
```

### `profile_run.py`
Quick single-process profiling script. Generates deterministic test names, warms caches, takes 5 pure timing measurements (no profiling overhead) for accurate throughput stats, then runs one `cProfile` pass for a top-25 function breakdown. Good for a fast sanity check during development.

```bash
uv run python scripts/profile_run.py
```

### `profile_threaded.py`
Multi-threaded performance and thread-safety validation. Tests `normalize_name` throughput across 1/2/4/8 threads using a shared `ChineseNameDetector` instance, verifies that multi-threaded results are identical to single-threaded results, and reports speedup and CV per thread count.

```bash
uv run python scripts/profile_threaded.py
```

### `profile_multiprocess.py`
Persistent multi-process throughput and parity check. Compares single-process throughput to a spawn-based persistent process pool, verifies that outputs are identical for a deterministic workload, and reports median speedup.

```bash
uv run python scripts/profile_multiprocess.py --names 12000 --warmup 3000 --runs 3 --workers 6 --chunk-size 64
```

### `verify_multiprocess.py`
Cross-platform multiprocessing verifier. Checks correctness parity for local,
one-shot, auto-wrapper, and persistent-pool paths, then reports throughput for
flat independent names and independent author-list batches.

```bash
uv run python scripts/verify_multiprocess.py --json-output scratch/mp_verify_windows.json
UV_PROJECT_ENVIRONMENT=/tmp/sinonym-wsl-venv uv run python scripts/verify_multiprocess.py --json-output scratch/mp_verify_wsl.json
```

### `train_ml_classifier_for_chinese_vs_japanese.py`
Trains the Chinese-vs-Japanese name classifier used in production. Downloads Chinese (~1.2M) and Japanese (~180K) name corpora, trains a scikit-learn pipeline (TF-IDF character n-grams + 20 linguistic heuristic features + logistic regression), and saves the model to `data/chinese_japanese_classifier.skops`.

```bash
uv run python scripts/train_ml_classifier_for_chinese_vs_japanese.py
```

### `name_order_routing_rules.py`
Applies the external routing rules for context runs that compare paper-level PP, VYS, input-order abstain, and terminal non-person outputs. The routing policy lives in `sinonym.pipeline.name_order_routing`; this script is the file-format CLI wrapper for already-expanded routing rows and adds `router_prediction` plus `router_reason`.
`router_prediction` can be `pp`, `vys`, `abstain`, or `not_person`; `not_person` means no person parse should be emitted.
For PP/VYS/abstain, input rows contain parser evidence from aligned PP and VYS runs. The router derives `old_prediction`, `old_reason`, `new_prediction`, and `new_reason` as audit output; those fields are not policy inputs.
For direct PP/VYS `BatchParseResult` usage, call `build_pp_vys_abstain_rows` or `route_pp_vys_abstain_batches`; those functions use the same evidence-row policy path as the file CLI.
The PP/VYS/abstain row regime requires raw `name` so the narrow validated name-prior guards can run.
CSV and JSONL use only the standard library; Parquet inputs/outputs require pandas plus a Parquet engine in the runtime environment.

```bash
uv run python scripts/name_order_routing_rules.py pp-vys-abstain --input pp_vys_features.parquet --output routed.parquet
uv run python scripts/name_order_routing_rules.py pp-abstain --input pp_only_features.parquet --output routed.parquet
```

### `change_class_tally.py`
Reference detector behind the "change-class sizes" table in the PR review. Runs SQL slice-detectors (via DuckDB, memory-safe streaming) over a canonical parquet whose rows are one per distinct production `(first, middle, last)` split (a joined name string can recur under several splits, so row count > distinct name count). sinonym 0.4.0 was run on the joined `nm` only; the `db_*` fields are the untouched production reference and `norm_*` is sinonym's output — production vs sinonym side by side. Prints per-split count, occurrence count, and share of non-Chinese occ per change class. Counts are exact for each detector's signature, but a signature is a heuristic (can under-capture; the order-swap / compound-surname classes are ceilings that include correct cases) — not ground truth. Requires `duckdb`.

```bash
uv run python scripts/change_class_tally.py path/to/canonical.parquet
```

## Abandoned Scripts

These remain for historical reference but are not used by the library. The rule-based parser in `sinonym.services.parsing` replaced the ML approach.

### `generate_chinese_name_corpus_data.py`
Was intended to generate training data for an ML-based name parsing disambiguation model. Downloads 200K Chinese names, romanizes them, generates all possible surname/given-name parses, and creates labeled training examples. The ML parsing model did not outperform the rule-based system.

### `generate_acl_data.py`
Supplementary data generator for the abandoned ML parsing effort. Processes ACL 2025 conference author names to create additional training examples in a different distribution (romanized, Western ordering).
