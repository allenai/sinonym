# East Asian name-order lexicon notices

The runtime assets `east_asian_roman_lexicons.json.gz` and
`japanese_native_lexicons.json.gz` are deterministic derived lookup lists used
only by the conservative Japanese, Korean, and Vietnamese family-first router.
They contain component names, not complete people.

## Japanese Personal Name Dataset

Source: <https://github.com/shuheilocale/japanese-personal-name-dataset>

Pinned commit: `c5220278652e7bae05b06cfaf527f1b09a100de6`

Copyright (c) 2022 shuheilocale. Distributed under the MIT License. The source
repository's `LICENSE` file contains the full notice and permission terms.

The derived assets use the repository's surname and male/female given-name CSV
files. The build script records and verifies the SHA-256 of every input.

## Popular Names by Country Dataset

Source: <https://github.com/sigpwned/popular-names-by-country-dataset>

Pinned commit: `eb62e13d4d62dd96cdfae79d293a02066352205f`

Distributed under CC0-1.0. The derived assets use only the Korean and
Vietnamese surname rows whose provenance is recorded in the generated JSON.

## Excluded research source

JMnedict/ENAMDICT was evaluated in the scratch experiment but is deliberately
not included in these package assets. Its EDRDG/CC BY-SA terms include
attribution and ongoing data-update requirements that are outside this static
asset's release contract.
