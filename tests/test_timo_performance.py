"""Throughput of the TIMO writer (``Predictor.predict_batch``), the path production calls per paper.

``tests/test_performance.py`` covers ``ChineseNameDetector`` on single names; this covers the paper-level resolution
that the corpus priors, compound surnames and the scalar candidate chain run in. The bylines are generated
deterministically from the detector's own surname and given-name inventories plus fixed Western, Hispanic,
initials and hyphenated shapes, so the test needs no data file. Paper context only, single process.

The floor sits well under a laptop measurement (about 2,800 authors/s on real ORCID bylines and higher on these) so
that slower CI machines pass; ``scripts/benchmark_timo.py`` compares two trees for a relative check before a release.
"""

import random
import time

from sinonym import ChineseNameDetector
from sinonym.timo.interface import Instance, Predictor, PredictorConfig, SourceAuthorFields

MIN_AUTHORS_PER_SECOND = 800
N_PAPERS = 1500

WESTERN_FIRST = ["John", "Mary", "David", "Sarah", "Michael", "Lisa", "Giulia", "Thomas", "Anna", "Pierre"]
WESTERN_LAST = ["Smith", "Johnson", "Ferrari", "Müller", "Dubois", "Kowalski", "Andersson", "O'Brien", "van der Berg", "Rossi"]
HISPANIC = [("María José", "García López"), ("Juan Carlos", "Rivera Maza"), ("Ana", "de la Cruz"), ("Luis", "Fernández Morales")]


def build_papers(detector: ChineseNameDetector) -> list[Instance]:
    rng = random.Random(42)  # noqa: S311
    surnames = sorted(detector._data.surnames)[:200]  # noqa: SLF001
    givens = sorted(detector._data.given_names)[:400]  # noqa: SLF001
    papers = []
    for _ in range(N_PAPERS):
        authors = []
        for _ in range(rng.randint(3, 12)):
            shape = rng.random()
            sur, g1, g2 = rng.choice(surnames).title(), rng.choice(givens).title(), rng.choice(givens)
            if shape < 0.35:
                authors.append(SourceAuthorFields(first_name=f"{g1}-{g2}", last_name=sur))
            elif shape < 0.5:
                authors.append(SourceAuthorFields(first_name=sur, last_name=f"{g1}{g2}"))
            elif shape < 0.6:
                authors.append(SourceAuthorFields(first_name=g1, last_name=sur))
            elif shape < 0.8:
                middle = rng.choice(["", "A.", "J. R."]) or None
                first, last = rng.choice(WESTERN_FIRST), rng.choice(WESTERN_LAST)
                authors.append(SourceAuthorFields(first_name=first, middle_names=middle, last_name=last))
            elif shape < 0.9:
                first, last = rng.choice(HISPANIC)
                authors.append(SourceAuthorFields(first_name=first, last_name=last))
            else:
                initial = f"{rng.choice('ABCDEFGHJKLMNPRSTW')}."
                authors.append(SourceAuthorFields(first_name=initial, last_name=rng.choice([*WESTERN_LAST, sur])))
        papers.append(Instance(pp_authors=authors))
    return papers


def test_timo_writer_throughput():
    detector = ChineseNameDetector()
    papers = build_papers(detector)
    predictor = Predictor(PredictorConfig(parallel="never"), ".")
    predictor.predict_batch(papers[:100])  # load lazy assets and warm caches
    n_authors = sum(len(p.pp_authors) for p in papers)
    start = time.perf_counter()
    for i in range(0, len(papers), 500):
        predictor.predict_batch(papers[i:i + 500])
    rate = n_authors / (time.perf_counter() - start)
    print(f"TIMO writer: {n_authors:,} authors on {len(papers):,} papers at {rate:,.0f} authors/s")
    assert rate >= MIN_AUTHORS_PER_SECOND, f"{rate:,.0f} authors/s is under the {MIN_AUTHORS_PER_SECOND:,} floor"
