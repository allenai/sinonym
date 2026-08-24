# TIMO writer contract

`sinonym` is the TIMO integration surface for callers that need final,
directly writable author fields. It resolves paper-level (PP), optional
venue/year/source-level (VYS), scalar canonical, and source-preservation paths
inside Sinonym instead of asking each writer to reproduce that cascade.

For ordinary single-name normalization, use `ChineseNameDetector` as shown in
the main README. Use TIMO when author records must stay positionally aligned
and the caller needs an explicit write or suppression decision.

## Python example

The public contract consists of six classes from `sinonym.timo.interface`:

```python
from sinonym.timo.interface import (
    Instance,
    Prediction,
    Predictor,
    PredictorConfig,
    ResolvedAuthorFields,
    SourceAuthorFields,
)

predictor = Predictor(
    PredictorConfig(parallel="never"),
    artifacts_dir="",
)
request = Instance(
    pp_authors=[
        SourceAuthorFields(first_name="Li", last_name="Wei"),
        SourceAuthorFields(first_name="John", last_name="Smith"),
    ],
    # Supply only names outside the focal paper; Sinonym prepends its authors.
    vys_other_names=["Jane Doe"],
)

(paper,) = predictor.predict_batch([request])
assert isinstance(paper, Prediction)
assert all(isinstance(author, ResolvedAuthorFields) for author in paper.authors)

assert (paper.authors[0].first_name, paper.authors[0].last_name) == ("Wei", "Li")
assert paper.authors[0].resolution_provenance.value == "pp"
assert paper.authors[0].resolution_action.value == "assign"

assert (paper.authors[1].first_name, paper.authors[1].last_name) == ("John", "Smith")
assert paper.authors[1].resolution_provenance.value == "scalar"
assert paper.authors[1].resolution_action.value == "assign"
```

The TIMO model name is `sinonym`. It is the only supported TIMO contract.

## Request wire shape

One `Instance` represents one paper:

```json
{
  "pp_authors": [
    {"first_name": "Li", "last_name": "Wei"},
    {"first_name": "John", "last_name": "Smith"}
  ],
  "vys_other_names": ["Jane Doe"]
}
```

`pp_authors` is required and its order is authoritative for alignment. Each
author accepts optional strict-string `first_name`, `middle_names`,
`last_name`, and `suffix` fields. Duplicate names are valid: results are joined
to requests by position, never by name text.

Sinonym derives each focal name by joining the first, middle, and last source
fields in that order; the suffix is deliberately excluded from scalar and
batch inference. Do not send a second focal-name slice. When
`vys_other_names` is nonempty, Sinonym prepends the derived focal names
internally, so this field must contain only non-focal names from the VYS
context.

The VYS forms have these meanings:

| Input | Routing behavior |
| --- | --- |
| omitted | Defaults to `[]`; PP-only |
| `[]` | PP-only |
| nonempty list | PP/VYS routing using the derived focal slice plus these names |
| `null` | Validation error |

An empty `pp_authors` list is valid and produces one paper result with an empty
`authors` list.

### Missing and empty values

Request serialization omits fields whose value is `None`; an explicit empty
string is retained. On output, `first_name`, `middle_names`, and `last_name`
are always strings, using `""` when a component is empty. `suffix` remains
nullable because its missingness is meaningful.

The output suffix is final. A nonempty source suffix wins; otherwise a suffix
discovered by the selected semantic result fills it. If neither exists, the
source `None` versus `""` state is retained. Writers must not merge or
normalize the suffix again.

Unknown request fields and non-string component values fail Pydantic
validation. This catches misspelled or drifted wire schemas at the boundary.

## Response wire shape

TIMO returns one `Prediction` for each request and one directly writable,
aligned author object for each `pp_authors` entry:

```json
{
  "authors": [
    {
      "first_name": "Wei",
      "middle_names": "",
      "last_name": "Li",
      "suffix": null,
      "resolution_provenance": "pp",
      "resolution_action": "assign",
      "resolution_reason": "pp_selected",
      "chinese_detected": true
    },
    {
      "first_name": "John",
      "middle_names": "",
      "last_name": "Smith",
      "suffix": null,
      "resolution_provenance": "scalar",
      "resolution_action": "assign",
      "resolution_reason": "scalar_baseline",
      "chinese_detected": false
    }
  ]
}
```

`chinese_detected` reports the per-name Chinese recognition (the batch parse
success), stamped independently of the resolution path — a source-preserved
author can still carry `chinese_detected: true`.

The writer rule is intentionally small:

| `resolution_action` | Writer behavior |
| --- | --- |
| `assign` | Write the supplied fields. A semantic candidate or reviewed source rule supplied the roles. |
| `preserve_input` | Write the supplied fields. Policy retained the safe input assignment rather than applying a reorder. |
| `suppress` | Do not emit this author. Keep the response slot for alignment and diagnostics. |

Every response also explains how the final fields were obtained:

| `resolution_provenance` | Meaning |
| --- | --- |
| `pp` | Paper-author batch evidence |
| `vys` | Non-focal venue/year/source context |
| `scalar` | Single-name canonical analysis |
| `source` | Exact source preservation or a closed reviewed source-shape rule |

`resolution_reason` gives the specific policy explanation, such as
`pp_selected`, `scalar_baseline`, or `reviewed_non_person_pattern`. Reasons are
a closed enum, and each reason permits exactly one provenance/action pair. Use
the action, not ad hoc interpretation of reason strings, to decide whether to
write the author. The authoritative enum and mapping live in
[`sinonym/coretypes/routing_resolution.py`](../sinonym/coretypes/routing_resolution.py).

Strict spaced-native Japanese evidence is terminal scalar evidence rather than
a batch preference. Family-first assignments use
`japanese_native_spaced_strict_assignment`; strict given-first preservation
uses `japanese_native_spaced_strict_given_first_preserve_input`. Broader
one-sided or conflicting Japanese dictionary shapes remain nonterminal. Their
canonical sidecar preserves authored display order while retaining the inferred
semantic roles, making the displayed canonical text a fixed point.

A PP-only abstention normally retains its PP input assignment. For an
all-native authored surface that contains a real component boundary, TIMO
instead defers to scalar evidence only when the existing Japanese classifier is
affirmative at its `0.8` threshold on the exact space-preserving surface. This
narrow exception prevents a PP abstention from relabeling preserved native
Japanese or Korean-Hanja fields as pinyin; ordinary Chinese controls remain on
the PP path. A spaced Han row also remains on PP when its parse actually applies
a reviewed surname-position reading to the assigned source character (for
example, `曾` as `Zeng` or `仇` as `Qiu`); that contextual assignment is
stronger than the broad Japanese probability alone.

## How structured fields are used

For ordinary inference, source component labels provide lossless boundaries
and lineage rather than assumed semantic truth. Scalar and batch candidates
operate on the derived `first middle last` text. This lets Sinonym repair a
source record whose fields were populated with the wrong roles.

A deliberately closed set of reviewed source shapes may use field placement as
additional evidence. For example, a record whose only populated name component
is:

```json
{"last_name": "Masterov R. A."}
```

resolves to `first_name="R."`, `middle_names="A."`, and
`last_name="Masterov"`, with reason
`structured_surname_initial_tail_assignment`. This exception is narrow;
callers should submit source fields as received rather than anticipating it.

The same source-pattern tier recognizes a last-field-only Western
transliteration whose authored components are separated by U+00B7. It accepts
only two or three nonempty, person-like components with CJK content, preserving
their script and assigning them as given/surname or given/middle/surname. This
is not a global middle-dot split: populated first/middle fields, Latin-only
text, four-or-more components, organization markers, and malformed segments
remain outside the rule.

## `not_person` is not automatically suppression

The raw PP/VYS router's `not_person` result describes an intermediate batch
candidate. TIMO still evaluates scalar canonical normalization and safe source
materialization. A real person can therefore receive `assign` or
`preserve_input` even when that router candidate says `not_person`; ordinary
parser failure is not a writer deletion decision.

Only `resolution_action="suppress"` tells a writer to omit an author. It is
reserved for a reviewed semantic non-person pattern. The response keeps the
source-shaped slot so diagnostics and positional alignment remain intact.
Reviewed patterns include suffix-free, last-field-only Han institutions whose
text contains an established strong organization marker. Raw Hangul
organization surfaces use a separate all-token suffix grammar rather than
broadening Han/CJK parsing primitives.

## Validation and failures

The public `ChineseNameDetector.analyze_name_batch()` API is fail-fast: batch,
service, invariant, and unexpected implementation failures propagate. TIMO
uses a lean related-batch path with the same failure contract but omits public
canonical sidecar work because terminal resolution handles scalar candidates
separately.

TIMO validates paper counts, author ordering, and PP/VYS alignment. Schema
errors, invariant violations, and unexpected implementation failures
propagate. Only explicitly typed evidence or hard-materialization failures are
handled as policy outcomes; those cases preserve source fields and expose an
explicit resolution reason. Callers should fail the enclosing work item rather
than inventing another name fallback; any retry policy remains caller-owned
and bounded.
