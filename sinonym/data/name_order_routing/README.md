# Name Order Routing Label Fixtures

Compact labeled fixtures for the external routing rules in
`sinonym.pipeline.name_order_routing`.

These are deliberately minimized from scratch experiment artifacts. They keep
only the manual label, split metadata, and columns required by the tracked
routers. They omit paper IDs, source URLs, notes, packet text, and debug-only
columns.

## Files

- `pp_vys_abstain_labels.jsonl`: 1,000 PP/VYS/abstain routing examples from
  four 250-row labeling rounds. This fixture includes raw `name` because the
  promoted narrow name-prior guards require it. Labels are `pp`, `vys`, or
  `uncertain`.
- `pp_abstain_labels.jsonl`: 750 PP/abstain routing examples from the current
  corrected train, corrected holdout, and fresh holdout feature sets. Labels
  are `pp`, `abstain`, or `uncertain`.

## Semantics

For PP/VYS/abstain scoring, router output `abstain` means "emit the
preprocessed input-order parse." The regression test maps `abstain` to the
computed `input_order_candidate` before comparing against `pp`/`vys` labels.
