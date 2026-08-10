# Routed V3 writer contract

`sinonym_routing_v3` is the integration surface for callers that need final,
directly writable author fields. It resolves paper-level (PP), optional
venue/year/source-level (VYS), scalar canonical, and source-preservation paths
inside Sinonym instead of asking each writer to reproduce that cascade.

For ordinary single-name normalization, use `ChineseNameDetector` as shown in
the main README. Use V3 when author records must stay positionally aligned and
the caller needs an explicit write or suppression decision.

## Python example

The supported Python contracts are exposed from `sinonym.timo.interface`:

```python
from sinonym.timo.interface import (
    PredictorConfig,
    RoutingInstanceV3,
    RoutingPredictorV3,
    SourceAuthorFields,
)

predictor = RoutingPredictorV3(
    PredictorConfig(parallel="never"),
    artifacts_dir="",
)
request = RoutingInstanceV3(
    pp_authors=[
        SourceAuthorFields(first_name="Li", last_name="Wei"),
        SourceAuthorFields(first_name="John", last_name="Smith"),
    ],
    # Supply only names outside the focal paper. V3 prepends the paper authors.
    vys_other_names=["Jane Doe"],
)

(paper,) = predictor.predict_batch([request])
resolved = [author.resolved_fields for author in paper.authors]

assert (resolved[0].first_name, resolved[0].last_name) == ("Wei", "Li")
assert resolved[0].resolution_provenance.value == "pp"
assert resolved[0].resolution_action.value == "assign"

assert (resolved[1].first_name, resolved[1].last_name) == ("John", "Smith")
assert resolved[1].resolution_provenance.value == "scalar"
assert resolved[1].resolution_action.value == "assign"
```

The TIMO model name is `sinonym_routing_v3`. The v1 and v2 request/response
schemas are unchanged; `sinonym_routing_v2` is the rollout rollback path.

## Request wire shape

One `RoutingInstanceV3` represents one paper:

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

V3 derives each focal name by joining the first, middle, and last source fields
in that order; the suffix is deliberately excluded from scalar and batch
inference. Do not send a second focal-name slice. When `vys_other_names` is
nonempty, V3 prepends the derived focal names internally, so this field must
contain only non-focal names from the VYS context.

The VYS forms have these meanings:

| Input | Routing behavior |
| --- | --- |
| omitted or `null` | PP-only |
| `[]` | PP-only; the empty-list distinction remains visible on the request DTO |
| nonempty list | PP/VYS routing using the derived focal slice plus these names |

An empty `pp_authors` list is valid and produces one paper result with an empty
`authors` list.

### Missing and empty values

Request serialization omits fields whose value is `None`; an explicit empty
string is retained. On output, `first_name`, `middle_names`, and `last_name`
are always strings, using `""` when a component is empty. `suffix` remains
nullable because its missingness is meaningful.

The suffix in `resolved_fields` is final. A nonempty source suffix wins;
otherwise a suffix discovered by the selected semantic result fills it. If
neither exists, the source `None` versus `""` state is retained. Writers must
not merge or normalize the suffix again.

Unknown request fields and non-string component values fail Pydantic
validation. This deliberately catches misspelled or drifted wire schemas at
the boundary.

## Response wire shape

V3 returns one paper prediction for each request and one aligned author slot
for each `pp_authors` entry:

```json
{
  "authors": [
    {
      "resolved_fields": {
        "first_name": "Wei",
        "middle_names": "",
        "last_name": "Li",
        "suffix": null,
        "resolution_provenance": "pp",
        "resolution_action": "assign",
        "resolution_reason": "pp_selected"
      }
    },
    {
      "resolved_fields": {
        "first_name": "John",
        "middle_names": "",
        "last_name": "Smith",
        "suffix": null,
        "resolution_provenance": "scalar",
        "resolution_action": "assign",
        "resolution_reason": "scalar_baseline"
      }
    }
  ]
}
```

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
the action—not ad hoc interpretation of reason strings—to decide whether to
write the author. The authoritative enum and mapping live in
[`sinonym/coretypes/routing_resolution.py`](../sinonym/coretypes/routing_resolution.py).

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
`structured_surname_initial_tail_assignment`. This exception is narrow; callers
should submit source fields as received rather than trying to anticipate it.

## `not_person` is not automatically suppression

The raw PP/VYS router's `not_person` result describes an intermediate batch
candidate. V3 still evaluates scalar canonical normalization and safe source
materialization. A real person can therefore receive `assign` or
`preserve_input` even when that router candidate says `not_person`; ordinary
parser failure is not a writer deletion decision.

Only `resolution_action="suppress"` tells a writer to omit an author. It is
reserved for a reviewed semantic non-person pattern. The response keeps the
source-shaped slot so diagnostics and positional alignment remain intact.

## Validation and failures

The public `ChineseNameDetector.analyze_name_batch()` API is forgiving of an
internal per-batch failure: it logs the failure and returns guarded per-name
results. Routed V3 intentionally uses a strict batch path because silently
falling back could change a terminal writer decision.

V3 validates paper counts, author ordering, and PP/VYS alignment. Schema
errors, invariant violations, and unexpected implementation failures
propagate. Only explicitly typed evidence or hard-materialization failures are
handled as policy outcomes; those cases preserve source fields and expose an
explicit resolution reason. Callers should fail the enclosing work item rather
than inventing another name fallback; any retry policy remains caller-owned and
bounded.
