# Scalar source-surname repartition evaluation

This note records the held-out evidence for the terminal resolver's
`_scalar_clean_source_surname_repartition_candidate` rule. The rule keeps a
structured, multi-token source surname when scalar parsing changes
only the boundary between that surname and the middle-name field. It refuses
source first-name initials and initials in the peeled surname prefix.

## Frozen rule: bucket 96

The rule was developed on hash bucket 95 and evaluated without retuning on
bucket 96. Both replay arms covered the same 4,982,002 author occurrences in
1,653,667 complete papers. They differed on 35,838 occurrences (0.719%), all
inside the frozen predicate; there were no routing-only or off-predicate
changes.

A random sample of 500 previously unseen activations was labelled from
source-only review packets before either prediction was revealed. Of those,
478 people and 4 nonpeople were judgeable; 18 rows were unjudgeable.

- Locked-label SHA-256:
  `b8d5ddaf83cd3f584ad5e74b7e53105d9d70c8aab01abe32a4ac3a6f088e353e`
- Sealed-prediction SHA-256:
  `2229227bb6a54c8be649312ebd041a8b3070ab8fdeea9145225f8f9675404859`

| Transition | Count |
|---|---:|
| Before wrong, after correct | 326 |
| Before correct, after wrong | 90 |
| Both wrong | 66 |
| Both correct | 0 |

Exact accuracy among judgeable activations rose from 90/482 (18.67%) to
326/482 (67.63%), a net gain of 236 exact tuples or 48.96 percentage points.
The one-sided paired sign-test result was `p = 8.69e-33`; the paper-cluster
bootstrap 95% interval was +41.79 to +56.02 points. The estimated overall
corpus gain was +0.352 percentage points.

The evidence supports the frozen rule for aggregate exact accuracy. It does not
claim that the rule is regression-free. One known regression is retained in
the routed parity fixture: `production-248247190`, position 0, keeps the
publisher surname `Ngoc Mai` although the identity-level boundary is `Mai`.

## Rejected refinement: bucket 97

After the bucket-96 reveal, regressions appeared concentrated in rows with a
blank source middle field. A token-prior gate was designed from that result and
tested on fresh hash bucket 97. Bucket 97 contained 4,993,759 author
occurrences in 1,655,159 papers. The gate blocked 7,767 of 36,356 ungated
activations, and every replay difference was inside that blocked set.

A locked sample contained 300 blocked activations: 295 judgeable people, 2
judgeable nonpeople, and 3 unjudgeable rows.

- Locked-label SHA-256:
  `36831afada10cc417b6410fcee2d677c72a4d208f9ce8946b9152de317fdfd47`
- Sealed-prediction SHA-256:
  `1f0ffa464ddd019a77a3491eba6bb10acc58110b544f64034670f8813fe01af3`

| Transition | Count |
|---|---:|
| Gate right, ungated wrong | 129 |
| Ungated right, gate wrong | 109 |
| Both wrong | 59 |
| Both right | 0 |

Although wins exceeded losses, the preregistered inference criteria failed:
the one-sided paired sign test was `p = 0.109`, and the paper-cluster bootstrap
95% interval was -3.38 to +17.00 points. High-confidence identity-backed rows
also favored the ungated rule (42 gate wins versus 77 losses). The gate was
therefore rejected without threshold retuning, and the bucket-96 rule remains
the selected behavior.

Any future refinement should be treated as a new experiment on an untouched
bucket. Buckets 95, 96, and 97 have all informed the current decision and must
not be reused as fresh validation data.
