# Batch processing

Batch processing helps when a related group of names is likely to follow one
ordering convention. A paper author list is a natural batch boundary; a whole
corpus assembled from unrelated sources usually is not.

Sinonym first parses each name independently. Eligible Latin-only Chinese names
then vote for surname-first or given-first order. If the batch signal clears the
application gate, that order is applied to eligible ambiguous names. Names that
do not participate keep their own parsing.

Authored structure is retained separately from parser token order. Explicit
`Last, First` rows and accepted compact CamelCase pairs cast their authoritative
source-order vote but are locked against peer reversal. When identical endpoint
spellings make both full role reconstructions possible (for example,
`Yang Yang`), the row is `mixed`, does not vote, and cannot receive a batch
override. With no applicable batch context, materialization is exactly the
scalar result.

## Choose an API

| Method | Input | Output | Use it when |
| --- | --- | --- | --- |
| `detect_batch_format()` | One name list | `BatchFormatPattern` | You need only the detected convention. |
| `process_name_batch()` | One related name list | `list[ParseResult]` | You need final parses and not the evidence. |
| `analyze_name_batch()` | One related name list | `BatchParseResult` | You need final parses plus decision and per-name evidence. |
| `process_name_batches()` | Many independent name lists | `list[list[ParseResult]]` | You have many batch boundaries, such as many papers. |
| `analyze_name_batches()` | Many independent name lists | `list[BatchParseResult]` | You need evidence for many independent batches. |
| `normalize_names()` | Independent names | `list[ParseResult]` | The names do not share a batch convention. |

`process_name_batch_multiprocess()` is a compatibility helper that processes
one batch through a temporary pool. Prefer `process_name_batches()` for many
batches or a persistent pool for repeated calls.

Every result sequence preserves input order. For the plural APIs, each inner
list remains an independent batch boundary; evidence never crosses between
inner lists.

## Inspect one batch

```python
from sinonym.detector import ChineseNameDetector

detector = ChineseNameDetector()
names = ["Zhang Wei", "Li Ming", "Wang Xiaoli", "Liu Jiaming"]
analysis = detector.analyze_name_batch(names)

pattern = analysis.format_pattern
print(pattern.dominant_format.value)
print(f"application confidence: {pattern.decision_confidence:.1%}")

for name, result, evidence in zip(
    analysis.names,
    analysis.results,
    analysis.name_order_evidence,
    strict=True,
):
    print(name, result.result, evidence.batch_changed_format)
```

Use `analysis.improvements` for the indices whose selected order changed under
batch context. `analysis.individual_analyses` contains the scalar candidates
and scores that preceded the batch decision.

## Decision gate and validation

The public defaults are:

- `format_threshold=0.55`
- `minimum_batch_size=2`

The detected direction is applied only when all of these conditions hold:

- the submitted list contains at least `minimum_batch_size` names;
- at least two eligible names cast a direction vote;
- the direction is unambiguous and its `decision_confidence` is greater than
  `0.5`; and
- `decision_confidence >= format_threshold`.

`minimum_batch_size` accepts integers of at least `1`. Setting it to `1` permits
analysis of a one-name list, but it does not remove the two-vote requirement for
applying a batch direction. If the submitted list is smaller than
`minimum_batch_size`, Sinonym still reports the detected pattern but sets
`threshold_met=False` and keeps individual results.

`format_threshold` must be finite and within `[0.0, 1.0]`. Invalid thresholds
and `minimum_batch_size < 1` raise `ValueError` before analysis starts.

`confidence` and `decision_confidence` answer different questions:

- `confidence` is the dominant direction's count share among detected
  participants.
- `decision_confidence` is the application score. It can use confidence-weighted
  tie-breaking and is the value compared with `format_threshold`.

## Result and evidence fields

A `BatchParseResult` contains aligned `names`, `results`, and
`name_order_evidence` lists, plus the detected `format_pattern`, the scalar
`individual_analyses`, and the indices in `improvements`.

### `BatchFormatPattern`

| Field | Meaning |
| --- | --- |
| `dominant_format` | `surname_first`, `given_first`, or `mixed`. |
| `confidence` | Count-based confidence for the dominant direction. |
| `decision_confidence` | Score used by the application threshold. |
| `threshold_met` | Whether the complete application gate passed. |
| `surname_first_count` | Surname-first direction votes. |
| `given_first_count` | Given-first direction votes. |
| `voting_count` | Sum of the two direction counts. |
| `total_count` | Candidate-bearing names counted by format detection, not necessarily the submitted-list length. |
| `vote_margin_count` | Absolute difference between the two direction counts. |
| `vote_margin` | `vote_margin_count / total_count`, or `0.0` when `total_count` is zero. |

### `NameOrderEvidence`

Each evidence record describes observable parser behavior. It deliberately does
not contain caller metadata or a PP/VYS routing decision.

| Fields | Meaning |
| --- | --- |
| `raw_name`, `raw_tokens`, `raw_token_count` | Submitted text and the tokens used for evidence. |
| `script_representation` | Parser cohort, such as `latin_only`, `han_only`, or `rejected_input`. |
| `batch_participant` | Whether the row contributed a candidate to the Latin batch vote. |
| `batch_applied` | Whether this successful row actually selected the detected convention. |
| `batch_changed_format` | Whether batch context changed the selected order. |
| `individual_format`, `selected_format` | Order before and after batch selection. |
| `selected_surname_position` | `first`, `last`, `internal`, or `unknown` in the evidence tokens. |
| `selected_surname_token_count` | Width of the selected surname endpoint span. |
| `first_token_surname_frequency`, `last_token_surname_frequency` | Surname frequency evidence at both endpoints. |
| `selected_surname_frequency` | Frequency evidence for the selected surname span. |
| `alternate_endpoint_surname_frequency` | Frequency evidence at the opposite endpoint. |
| `selected_over_alternate_surname_frequency_ratio` | Selected-to-alternate frequency ratio when defined. |
| `has_all_caps_token`, `all_caps_tokens` | Whether all-caps source tokens were observed and which tokens they were. |

Keep source, venue, year, and other caller-owned metadata beside the batch call.
External routing code can combine that metadata with `name_order_evidence`
without depending on Sinonym internals.

## Mixed inputs and script cohorts

Only vote-eligible Latin-only Chinese names vote on the Latin batch convention.
They receive that convention only when they have a candidate in the detected
format and source-order evidence has not locked the row. Han-only names,
explicitly aligned Han/Roman names, and other mixed-script inputs use their own
script evidence, so a Latin convention does not flip their order. Latin rows
with all-caps source-token cues expose those cues in `name_order_evidence` but
do not vote or receive Latin batch formatting.

Unambiguous names also retain their best individual parse. Recoverable
non-Chinese people remain unsuccessful in the legacy Chinese fields, while
their `canonical_name` sidecar remains available. All rows stay positionally
aligned:

```python
from sinonym.detector import ChineseNameDetector

detector = ChineseNameDetector()
names = ["John Smith", "Xin Liu", "Yang Li", "Wei Zhang", "Ming Wang"]
analysis = detector.analyze_name_batch(names)

assert len(analysis.results) == len(names)
for name, result in zip(names, analysis.results, strict=True):
    if result.success:
        print(name, "->", result.result)
    elif result.canonical_name is not None:
        print(name, "->", result.canonical_name.text)
```

## Failure behavior

`analyze_name_batch()`, `analyze_name_batches()`, `process_name_batch()`, and
`process_name_batches()` are fail-fast. Validation, service, invariant, and
unexpected implementation failures propagate to the caller instead of being
converted into fallback rows. Successful public analysis calls attach
`canonical_name` sidecars to their aligned parse results.

The persistent-pool equivalent is `pool.analyze_name_batches()`. It preserves
the original worker task error and closes the pool after a failed task. Broken
workers and initialization failures surface with process-pool context. Retry
and fallback policies belong to the caller so they can be bounded and observed.

## Many batches and multiprocessing

The high-level plural methods accept `parallel="auto"`, `"never"`, or
`"always"`. Auto mode avoids process-startup overhead for smaller calls. Its
current defaults start a pool at 2,000 batches on Linux and 4,000 batches on
other platforms; `normalize_names()` uses 10,000 and 25,000 names respectively.
Override `min_parallel_batches` or `min_parallel_names` when measurements on
your workload justify a different crossover.

For repeated high-throughput calls, keep one pool open so each worker can reuse
its detector:

```python
from sinonym.detector import ChineseNameDetector


def main() -> None:
    detector = ChineseNameDetector()
    paper_batches = [
        ["Wang An", "Yan Li", "Wu Gang", "Li Bao"],
        ["Li Wei", "Wang Weiming", "Zhang Ming"],
    ]

    with detector.create_persistent_multiprocess_pool(
        max_workers=6,
        chunk_size=64,
    ) as pool:
        parsed = pool.process_name_batches(paper_batches)
        analyzed = pool.analyze_name_batches(paper_batches)

    print(len(parsed), len(analyzed))


if __name__ == "__main__":
    main()
```

The pool uses `spawn` by default on Windows, macOS, and Linux. Put pool creation
behind `if __name__ == "__main__":` in executable scripts. Prefer the context
manager: exiting it closes the workers, and a closed pool cannot be reused.

Pool options are validated before workers start: `max_workers`, when supplied,
must be at least `1`; `chunk_size` must be at least `1`; and
`mp_start_method` must be supported by the platform. For per-name work,
`chunk_size` is the number of names sent in one worker task. For plural batch
work, it is the number of independent batches sent in one worker task.
