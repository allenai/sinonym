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

The Roman asset keeps two Japanese surname tiers. `japanese_surnames` contains
the 2,000-name source's directional evidence and may initiate a family-first
reorder. `japanese_possible_surnames` is its superset and may only establish
that input order is plausible; it cannot initiate a reorder. In candidate
selection, possible-surname evidence vetoes an unsupported endpoint reversal.
When strong evidence also supports family-first order, the veto requires a
strict given-first majority among the other paper authors.

The possible tier also includes Japanese spellings from the pinned country
surname source. This recovers alternate readings such as `Hirano`, `Nakashima`,
and `Yamasaki` without allowing them to initiate a reorder.

The reviewed table adds academic-author surname spellings that remain absent
from both open sources. Each row records a corpus occurrence and public identity
evidence. For example:

- `hiroya`: confirmed as the family name of Kou Hiroya by the
  [Tohoku University researcher page](https://www.pharm.tohoku.ac.jp/~hannou/hiroya/K_Hiroya-j.html)
  and [J-STAGE](https://www.jstage.jst.go.jp/article/cpb1958/43/3/43_3_529/_article/-char/en).
- `miwa`: confirmed as the family name of Takaya Miwa by
  [CiNii](https://cir.nii.ac.jp/crid/1390282679829118336?lang=en) and the
  [Japanese Society of Gastroenterological Surgery](https://www.jsgs.or.jp/journal/abstract/042101568_e.html).

The stored additions are exact observed Roman spellings. The runtime's existing
long-vowel aliases may match them only as ambiguity evidence; neither the stored
spellings nor their aliases generate directional evidence or paper-context
votes. The reviewed CSV keeps surname evidence separate from exact repairs. An
example surface documents each surname addition, while only a non-empty
`given_first_exact_surface` cell becomes a no-context repair. The exact tier
contains `Kou Hiroya`, `Takaya Miwa`, `Haruki Kadono`, `Shoji Kagami`, `Masaki
Takamoto`, and `Masaki Tomonaga`. The latter four were promoted only after all
692 exact normalized occurrences in the 691,008,961-row author corpus were
reviewed with no contrary or uncertain ordering. They are post-selection
reversal vetoes: unlike the two previously released surfaces, they do not
suppress an earlier candidate and therefore do not change routing metadata for
already-correct context-supported parses. All other reviewed surnames use the
general, candidate-aware possible-surname rule.

## Popular Names by Country Dataset

Source: <https://github.com/sigpwned/popular-names-by-country-dataset>

Pinned commit: `eb62e13d4d62dd96cdfae79d293a02066352205f`

Distributed under CC0-1.0. The derived assets use only the Korean and
Vietnamese surname rows whose provenance is recorded in the generated JSON.
The Roman asset also retains the folded Vietnamese surnames ranked 1-4 as a
separate thresholded list for the candidate-conditioned reorder-conflict rule;
it does not expose or infer a general runtime rank map.

## Excluded research source

JMnedict/ENAMDICT was evaluated in the scratch experiment but is deliberately
not included in these package assets. Its EDRDG/CC BY-SA terms include
attribution and ongoing data-update requirements that are outside this static
asset's release contract.
