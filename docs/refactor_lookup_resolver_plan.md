# Lookup Resolver Refactor Plan

This plan coordinates a small, behavior-preserving refactor for the surname
lookup-key bug class. The objective is not to change scoring policy; it is to
make the existing surname lookup policies explicit, named, and hard to misuse.

Router hardening is intentionally out of scope for this document. It is a
separate boundary and should get its own plan after the lookup work is stable.

## The Contract

There are two legitimate surname-mass policies. Every migrated call site should
choose one explicitly.

**Parser policy**

- Purpose: parse scoring and parse comparison.
- Key rule: use the parser-strict surname key, preserving as-written spellings
  only where parser scoring already intends to do so.
- Accessors: surname frequency, log probability, rank, and surname membership
  under the parser policy.
- Important boundary: romanization share discounts remain parser scoring logic,
  not resolver logic.

**Evidence policy**

- Purpose: batch/routing evidence and audit fields.
- Key rule: use `surname_lookup_key(norm_light, norm)`.
- Accessors: surname frequency, surname membership, and dominant-surname
  threshold checks under the as-written evidence policy.
- Important boundary: evidence should report the as-written mass that the input
  spelling carries, including curated romanization rows such as `fai` and
  `chien`.

Public resolver methods should return answers, not raw keys. The only exception
is batch span assembly: callers may ask for a span key when they need a key as
text, never to pair with a data lookup. Private key helpers inside the resolver
are fine. Exposing general-purpose keys recreates the original failure mode: a
caller can pair the right key with the wrong accessor.

## Current Evidence

The inconsistent branch is here:

- [parsing.py:760](sinonym/services/parsing.py#L760) derives `first_norm` and
  `last_norm` with `get_normalized`.
- [parsing.py:768](sinonym/services/parsing.py#L768) passes those remapped
  strings into `get_surname_freq`.
- Nearby code already avoids this by deriving parser surname keys before
  surname checks and frequency reads.

This is enough to justify a refactor, but not enough to call it a confirmed
wrong-output bug. Track 0 did not reproduce a concrete wrong parse on the
795-case eval corpus: an in-memory parser-key variant of this branch produced
zero output diffs. Treat this as a reachable inconsistent branch, not a proven
live behavior bug, until a smaller oracle-backed case is found.

## Safety Rule

Before changing scoring-adjacent code, record the current eval baseline. After
each slice, rerun the same command and compare both counts and failing inputs.

Current observed baseline on this branch:

```powershell
uv run python scratch/baseline_failures/run_eval.py --repo "D:\Google Drive\code\ai2\sinonym"
```

Observed result:

```text
total: 795  pass: 775  fail: 20
```

Do not hard-code this forever. If the branch moves, rerun the command and update
the working baseline before editing. For behavior-preserving slices, expected
delta is zero. If failures move, stop and classify the movement as intended,
accidental, or outside the slice before continuing.

Before starting Track 1, reconcile the 795-case eval with the current pytest
status gate once. `scratch/baseline_failures/cases.json` is an extracted
normalization corpus, while `scripts/check_test_status.py` gates the full test
suite by expected failure signatures. Record the source modules/count for the
extracted corpus, run the status checker, and document how the two gates relate
so that "zero eval delta" is not mistaken for full-suite coverage.

Current reconciliation result: `cases.json` contains 795 normalization cases
from 11 test modules. `uv run python scripts/check_test_status.py` passes at the
configured full-suite baseline of 16 pytest failures with performance OK. The
795-case eval is a scoring-focused normalization gate; the status checker
remains the full-suite gate.

## Coordination Rules

- Work in the order below unless the dependency list says otherwise.
- If multiple agents are active, claim one unchecked track and record owner/date.
- Keep to the track's write scope. If another file is needed, add a note under
  the track instead of quietly expanding scope.
- Use `uv run ...` for Python commands.
- Every completed track must include the exact verification command and result.
- Do not run broad or expensive jobs until the focused tests and the 795-case
  eval are clean for that slice.

Status markers:

- `[ ]` not started
- `[~]` in progress
- `[x]` complete
- `[!]` blocked

## Non-Goals

- No scoring-weight changes.
- No routing-policy changes.
- No rewrite of `normalize_name`.
- No public API or serialization-format change without explicit approval.
- No full call-site migration in one commit.
- No privatizing raw `NameDataStructures` accessors until migration is proven.

## Target Shape

Suggested module: `sinonym/services/name_lookup.py`.

Initial public API should be surname-only. Lookup callers should use
answer-returning methods; batch span assembly gets one explicit key accessor:

```python
class SurnameResolver:
    def parser_frequency(self, surname_tokens: Sequence[str]) -> float: ...
    def parser_logp(self, surname_tokens: Sequence[str], default: float) -> float: ...
    def parser_rank(self, surname_tokens: Sequence[str], default: float = 0.0) -> float: ...
    def parser_is_surname(self, surname_tokens: Sequence[str]) -> bool: ...
    def parser_is_wade_giles_initial_remapped_surname(self, token: str) -> bool: ...

    def evidence_frequency(self, token: str) -> float: ...
    def evidence_is_surname(self, token: str) -> bool: ...
    def evidence_is_dominant_surname(self, token: str) -> bool: ...
    def evidence_span_key(self, token: str) -> str: ...
    def spelling(self, token: str) -> SurnameRomanization | None: ...
```

Design notes:

- Use `Sequence[str]` for parser methods. A single surname token is
  `("li",)`, and a compound surname is `("ou", "yang")`.
- Keep parser key derivation private to the resolver.
- Keep general evidence key derivation private to the resolver.
- `parser_is_surname` is single-token in practice. It uses `tokens[0]` as the
  raw token and derives the normalized lookup value internally before delegating
  to the data-structure membership check.
- For evidence, the resolver owns key derivation plus frequency lookup.
  Surname-span selection remains in `BatchAnalysisService` and calls
  `evidence_frequency` for the chosen token or token-equivalent span.
- `evidence_span_key` is the only raw-key escape hatch. It exists for
  `BatchAnalysisService` span-string assembly, not for pairing keys with data
  frequency/membership APIs at call sites.
- Dominant-surname detection uses the evidence policy via
  `evidence_is_dominant_surname`; detection checks should not compare parser
  remap target mass to the dominant-surname threshold.
- Prefer direct `norm` / `norm_light` calls inside the resolver. They already
  use LRU-cached normalization. If performance regresses, add resolver-local
  memoization rather than threading per-call normalized caches through the
  public API.
- Keep `_surname_spelling_share_logp` in `NameParsingService`; it is scoring
  policy. It may call resolver methods for lookup answers.

## Track 0: Verify Or Downgrade The Premise

Owner: Codex

Write scope:

- `tests/test_regression_proposals.py`
- scratch probes only if needed

TODO:

- [x] Try to reproduce a concrete wrong parse caused by the branch around
  [parsing.py:760](sinonym/services/parsing.py#L760).
- [x] No wrong parse reproduced, so no failing behavioral regression test was
  added.
- [x] Add a green characterization test or note,
  and update wording from "live bad branch" to "inconsistent branch".
- [x] Record the probe evidence. There is no exact wrong input/expected/actual
  triple yet; the branch is reached, but no evaluated output changed.

Outcome:

No oracle-backed wrong parse was reproduced. `scratch/track0_lookup_branch_probe.py`
found that the branch is reachable on 155 eval candidate-score paths. Thirteen
paths read different surname frequencies under normalized-key versus
parser-strict-key semantics, but none changed the branch bonus and an in-memory
parser-key variant produced zero eval output diffs. This downgrades the premise
from "live bad branch" to "reachable inconsistent branch".

Verification log:

- Command: `uv run python scratch/track0_lookup_branch_probe.py`
- Result: `eval_cases: 795`, `branch_reached: 155`,
  `branch_freq_diffs: 13`, `branch_bonus_diffs: 0`,
  `current_failures: 20`, `parser_key_branch_failures: 20`,
  `eval_output_diffs: 0`.
- Command: `uv run pytest -q tests/test_regression_proposals.py`
- Result: `135 passed in 3.12s`.
- Command:
  `uv run python scratch/baseline_failures/run_eval.py --repo "D:\Google Drive\code\ai2\sinonym"`
- Result: `total: 795  pass: 775  fail: 20`; command exits 1 because
  failures remain and writes `scratch/baseline_failures/failures.tsv`.

Suggested commands:

```powershell
uv run pytest -q tests/test_regression_proposals.py
uv run python scratch/baseline_failures/run_eval.py --repo "D:\Google Drive\code\ai2\sinonym"
```

## Track 1: Resolver Skeleton

Owner: Codex / Euler subagent

Depends on:

- Track 0 is complete and explicitly downgraded.

Write scope:

- `sinonym/services/name_lookup.py`
- `sinonym/services/__init__.py`
- focused resolver tests, if useful

TODO:

- [x] Add `SurnameResolver` with dependencies on `NameDataStructures` and
  `NormalizationService`.
- [x] Implement private parser key derivation equivalent to the current
  parser-strict policy.
- [x] Implement private evidence key derivation equivalent to
  `surname_lookup_key(norm_light, norm)`.
- [x] Add public answer methods for parser frequency, logp, rank, surname
  membership, evidence frequency, and spelling resolution.
- [x] Do not expose public `parser_key` or `evidence_key` methods.
- [x] Keep raw `NameDataStructures` accessors unchanged in this slice.

Verification log:

- Command:
  `uv run pytest -q tests/test_surname_romanizations.py::test_surname_lookup_key_resolves_as_written_vs_remapped tests/test_surname_romanizations.py::test_evidence_frequencies_use_as_written_resolution tests/test_surname_romanizations.py::test_surname_resolver_parser_answers_match_current_parser_policy tests/test_surname_romanizations.py::test_surname_resolver_evidence_answers_match_current_evidence_policy`
- Result: `4 passed in 1.68s`.
- Command: `uv run pytest -q tests/test_surname_romanizations.py`
- Result: `14 passed in 1.72s`.
- Command:
  `uv run ruff check sinonym/services/name_lookup.py sinonym/services/__init__.py tests/test_surname_romanizations.py`
- Result: `All checks passed!`.
- Command:
  `uv run python scratch/baseline_failures/run_eval.py --repo "D:\Google Drive\code\ai2\sinonym"`
- Result: `total: 795  pass: 775  fail: 20`; command exits 1 because
  baseline failures remain.
- Command: `uv run python scripts/check_test_status.py`
- Result: `Tests are at expected baseline (16 failures, performance OK)`.

Suggested commands:

```powershell
uv run pytest -q tests/test_surname_romanizations.py::test_surname_lookup_key_resolves_as_written_vs_remapped tests/test_surname_romanizations.py::test_evidence_frequencies_use_as_written_resolution
uv run python scratch/baseline_failures/run_eval.py --repo "D:\Google Drive\code\ai2\sinonym"
```

## Track 2: Port Parser Call Sites

Owner: Codex

Depends on:

- Track 1 is complete.

Write scope:

- `sinonym/services/parsing.py`
- parser-focused tests

TODO:

- [x] Construct and store `SurnameResolver` inside `NameParsingService`.
- [x] Replace the inconsistent 3-token branch at
  [parsing.py:760](sinonym/services/parsing.py#L760) with resolver answer
  calls.
- [x] Replace parser uses of surname frequency, surname logp, surname rank, and
  surname membership where they currently require a hand-derived surname key.
- [x] Keep `_surname_spelling_share_logp` in `NameParsingService`; only replace
  lookup reads inside it as needed.
- [x] Delete `_surname_key_cached` if resolver-local caching makes it obsolete.
- [x] Do not touch `_given_name_key` or `_given_key_cached` in this track unless
  a touched line truly requires it.
- [x] Run performance tests if normalized-cache removal touches a hot path.

Notes:

- `_surname_key` remains as a compatibility delegate for
  `scripts/generate_chinese_name_corpus_data.py`; parser internals no longer
  use it. Track 5 should either migrate those script callers or explicitly keep
  the delegate.
- Compound fallback logp reads in `calculate_parse_score` remain raw data reads
  for `original_compound_format` compatibility.
- Wade-Giles initial-remapped surname detection is now a resolver answer method.
  Parser scoring still decides how to use that boolean for order preservation.

Verification log:

- Command:
  `uv run pytest -q tests/test_surname_romanizations.py tests/test_regression_proposals.py tests/test_regional_variants.py`
- Result: `183 passed in 2.54s`.
- Command:
  `uv run ruff check sinonym/services/parsing.py sinonym/services/name_lookup.py sinonym/services/__init__.py tests/test_surname_romanizations.py`
- Result: `All checks passed!`.
- Command:
  `uv run python scratch/baseline_failures/run_eval.py --repo "D:\Google Drive\code\ai2\sinonym"`
- Result: `total: 795  pass: 775  fail: 20`; command exits 1 because
  baseline failures remain.
- Command: `uv run pytest -q tests/test_performance.py`
- Result: `1 passed in 1.84s`.
- Command: `uv run python scripts/check_test_status.py`
- Result: `Tests are at expected baseline (16 failures, performance OK)`.

Suggested commands:

```powershell
uv run pytest -q tests/test_regression_proposals.py tests/test_surname_romanizations.py tests/test_regional_variants.py
uv run python scratch/baseline_failures/run_eval.py --repo "D:\Google Drive\code\ai2\sinonym"
uv run pytest -q tests/test_performance.py
```

Expected eval delta:

- Zero, unless Track 0 proved and fixed a concrete wrong parse.

## Sidecar Call-Site Audit

Owner: Noether subagent

Status: read-only audit complete; no files edited.

Use this as routing guidance for later tracks:

- Parser policy: `sinonym/services/parsing.py` parse pruning, scoring,
  `_surname_key`, `_surname_key_cached`, logp/rank/frequency comparisons, and
  detector parse-decision branches should move to parser answer methods.
- Evidence policy: `sinonym/services/batch_analysis.py` batch evidence fields,
  selected surname frequency fields, `_surname_lookup_key_for_token`, and the
  camelCase as-written branch in `sinonym/detector.py` should preserve
  as-written evidence semantics.
- Unclear policy: `sinonym/services/ethnicity.py` surname checks and
  `sinonym/services/non_person.py` rejection gates need explicit per-call-site
  decisions before migration. Do not mechanically swap these to either policy.
- Boundary caution: `NameDataStructures` raw accessors should remain until
  migration proves which call sites still legitimately need data-structure
  internals.

## Track 3: Port Batch Evidence Call Sites

Owner: Codex

Depends on:

- Track 1 is complete.
- Track 2 is green if this track would otherwise need parser-owned helpers.

Write scope:

- `sinonym/services/batch_analysis.py`
- batch/evidence tests

TODO:

- [x] Replace `_surname_lookup_key_for_token` with resolver evidence-frequency
  calls, or make it a thin private delegate.
- [x] Keep `_selected_surname_lookup_key` and
  `_selected_compound_surname_lookup_key` responsible only for choosing the
  selected surname span. They must not own evidence key policy or frequency
  reads after this track.
- [x] Preserve current evidence semantics for `fai`, `chien`, and direct-mass
  spellings.
- [x] Do not change batch routing policy or routed result materialization.
- [x] If `BatchAnalysisService` receives a resolver dependency, pass it
  explicitly through existing service construction.

Notes:

- Batch span selection still uses a private `_surname_lookup_key_for_token`
  delegate for matching only. Evidence frequency reads now call
  `SurnameResolver.evidence_frequency`.
- `_selected_surname_lookup_key` and `_selected_compound_surname_lookup_key`
  still choose spans; they do not read surname frequency.

Verification log:

- Command:
  `uv run pytest -q tests/test_batch.py tests/test_surname_romanizations.py::test_evidence_frequencies_use_as_written_resolution tests/test_bug_report_fixes.py::test_name_order_evidence_uses_cached_raw_tokens`
- Result: `44 passed in 1.81s`.
- Command:
  `uv run ruff check sinonym/services/batch_analysis.py tests/test_surname_romanizations.py`
- Result: `All checks passed!`.
- Command:
  `uv run python scratch/baseline_failures/run_eval.py --repo "D:\Google Drive\code\ai2\sinonym"`
- Result: `total: 795  pass: 775  fail: 20`; command exits 1 because
  baseline failures remain.

Suggested commands:

```powershell
uv run pytest -q tests/test_batch.py tests/test_surname_romanizations.py::test_evidence_frequencies_use_as_written_resolution tests/test_bug_report_fixes.py::test_name_order_evidence_uses_cached_raw_tokens
uv run python scratch/baseline_failures/run_eval.py --repo "D:\Google Drive\code\ai2\sinonym"
```

Expected eval delta:

- Zero.

## Track 4: Port Detector, Ethnicity, And Non-Person Call Sites

Owner: Codex

Depends on:

- Tracks 1, 2, and 3 are green.

Write scope:

- `sinonym/detector.py`
- `sinonym/services/ethnicity.py`
- `sinonym/services/non_person.py`
- tests covering those services

TODO:

- [x] Replace hand-derived surname frequency reads in detector bilingual and
  all-CJK branches with resolver answer methods.
- [x] Replace ethnicity frequency checks that OR multiple possible keys together
  with one explicit resolver policy.
- [x] Replace non-person surname evidence checks only when the correct policy is
  clear. If unclear, document the line and leave the existing behavior.
- [x] Do not alter classifier thresholds, non-person rules, or routing.

Notes:

- `NonPersonInputDetectionService._is_surname_like_token` intentionally keeps
  its direct `is_surname` checks for the Latin author-list heuristic. That rule
  combines per-part surname membership with compound-shape checks, and the
  correct resolver policy is less clear than the CJK first-surname frequency
  safeguard.
- Track 4 review found and fixed one extra raw surname-membership fallback in
  ethnicity. The author-list exception now has an inline comment at the direct
  membership site.

Verification log:

- Command:
  `uv run pytest -q tests/test_regression_proposals.py tests/test_non_chinese_rejection.py tests/test_mixed_scripts.py tests/test_bug_report_fixes.py`
- Result: `494 passed in 6.16s`.
- Command:
  `uv run ruff check sinonym/detector.py sinonym/services/ethnicity.py sinonym/services/non_person.py sinonym/services/name_lookup.py`
- Result: `All checks passed!`.
- Command:
  `uv run python scratch/baseline_failures/run_eval.py --repo "D:\Google Drive\code\ai2\sinonym"`
- Result: `total: 795  pass: 775  fail: 20`; command exits 1 because
  baseline failures remain.

Suggested commands:

```powershell
uv run pytest -q tests/test_regression_proposals.py tests/test_non_chinese_rejection.py tests/test_mixed_scripts.py tests/test_bug_report_fixes.py
uv run python scratch/baseline_failures/run_eval.py --repo "D:\Google Drive\code\ai2\sinonym"
```

Expected eval delta:

- Zero.

## Track 5: Boundary Tightening

Owner: Codex

Depends on:

- Tracks 2, 3, and 4 are green.

Write scope:

- `sinonym/services/initialization.py`
- `sinonym/services/name_lookup.py`
- tests for lookup boundaries
- `scripts/generate_chinese_name_corpus_data.py`
- `scripts/generate_name_statistics.py`
- `sinonym/services/parsing.py`

TODO:

- [x] Inventory remaining direct calls to `get_surname_freq`,
  `get_surname_freq_as_written`, `get_surname_logp`, `get_surname_rank`,
  `resolve_surname_spelling`, `surname_lookup_key`, and `is_surname`.
- [x] Decide which remaining direct calls are legitimate data-structure
  internals and which should move behind the resolver.
- [x] Consider `NewType` aliases only after call-site migration proves the
  shapes are stable.
- [x] Do not rename or privatize raw accessors unless the remaining call sites
  prove the churn will prevent real misuse.

Notes:

- Deleted `NameParsingService._surname_key`; `_surname_key_cached` was already
  gone. The historical corpus-training script now asks `SurnameResolver` for
  parser logp/rank answers directly.
- `scripts/generate_name_statistics.py` now uses `SurnameResolver` for the
  byline surname-membership helper.
- Remaining direct surname accessor/key-derivation calls are limited to:
  `NameDataStructures` methods/internals, `SurnameResolver` internals,
  `NameParsingService` compound-logp fallback for explicit compound aliases,
  and the documented Latin author-list exception in
  `NonPersonInputDetectionService._is_surname_like_token`.
- Direct reads of surname sets/tables, such as `surnames`,
  `surnames_normalized`, `surname_median_logprobs`, and Han corpus frequency
  tables, are still present outside the resolver. They are broader data access,
  not lookup-key derivation, and are intentionally only inventoried in this
  slice.
- NewType aliases are deferred. After call-site migration, the remaining
  shapes are stable enough to review, but adding aliases now would be churn
  without a measured safety gain.
- Resolver-local memoization was needed after benchmarking. A no-cache resolver
  benchmarked at `MEDIAN_NAMES_PER_SECOND=5043.40` versus clean `HEAD` at
  `6164.69`; adding private parser/evidence key caches raised the current tree
  to `6484.46`.

Verification log:

- Command:
  `rg -n "_surname_key\(|_surname_key_cached|_surname_lookup_key_for_token\(" sinonym scripts tests`
- Result: only `BatchAnalysisService._surname_lookup_key_for_token` remains,
  as the private span-selection delegate from Track 3.
- Command:
  `uv run pytest -q tests/test_surname_romanizations.py`
- Result: `14 passed in 1.65s`.
- Command:
  `uv run python -m py_compile scripts/generate_chinese_name_corpus_data.py scripts/generate_name_statistics.py`
- Result: completed with exit code 0.
- Command:
  `uv run ruff check scripts/generate_chinese_name_corpus_data.py scripts/generate_name_statistics.py sinonym/services/parsing.py tests/test_surname_romanizations.py`
- Result: `All checks passed!`.
- Command:
  `uv run python scratch/baseline_failures/run_eval.py --repo "D:\Google Drive\code\ai2\sinonym"`
- Result: `total: 795  pass: 775  fail: 20`; command exits 1 because
  baseline failures remain.

Suggested commands:

```powershell
rg -n "get_surname_freq\\(|get_surname_freq_as_written\\(|get_surname_logp\\(|get_surname_rank\\(|surname_lookup_key\\(|resolve_surname_spelling\\(|is_surname\\(" sinonym -g "!sinonym/data/**"
uv run ruff check .
uv run python scratch/baseline_failures/run_eval.py --repo "D:\Google Drive\code\ai2\sinonym"
```

Expected eval delta:

- Zero.

## Final Integration Checklist

- [x] Premise was either reproduced as a wrong-output bug or downgraded in the
  doc.
- [x] All verification logs are filled in.
- [x] The current eval baseline is recorded before and after each slice.
- [x] No public resolver method returns a raw surname key.
- [x] Each ported key-derivation helper is deleted or reduced to one private
  delegate with no other callers: `_surname_key`, `_surname_key_cached`,
  `_surname_lookup_key_for_token`, and inline `norm` / `norm_light` to raw
  `get_surname_freq` derivations. Verify this with `rg`.
- [x] `rg` shows no accidental new `get_normalized(...)` or `norm(...)` plus
  raw `get_surname_freq(...)` pairings outside documented exceptions.
- [x] Focused tests pass:

```powershell
uv run pytest -q tests/test_regression_proposals.py tests/test_surname_romanizations.py tests/test_batch.py tests/test_bug_report_fixes.py
```

- [x] Eval is unchanged except for explicitly approved fixes:

```powershell
uv run python scratch/baseline_failures/run_eval.py --repo "D:\Google Drive\code\ai2\sinonym"
```

- [x] Lint passes:

```powershell
uv run ruff check .
```

## Follow-Up Plan

Router boundary hardening remains worthwhile, but belongs in a separate plan.
That plan should cover `_required_int`, PP-only abstain decision precedence, and
TIMO materialization boundaries without mixing those concerns into lookup-key
policy.

## Open Decisions

- [ ] Exact resolver module name. Default proposal:
  `sinonym/services/name_lookup.py`.
- [ ] Whether to include a `GivenResolver` later. Do not add it in the first
  surname slice unless a touched line requires it.
- [ ] Whether remaining raw `NameDataStructures` accessors should become private
  after migration.
- [x] Whether resolver-local memoization is needed after performance testing.
