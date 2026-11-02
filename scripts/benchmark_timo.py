#!/usr/bin/env python3
"""Compare TIMO writer throughput of two sinonym trees on the same bylines (relative check before a release).

Each run is a fresh subprocess with the tree first on ``sys.path``, so the trees share nothing but the venv. Reports
the median authors/s per tree over ``--runs`` runs and fails when the candidate is slower than the baseline by more
than ``--max-slowdown`` (a fraction). The bylines are the deterministic synthetic papers of
``tests/test_timo_performance.py`` (a copy, so an older baseline tree without that test still runs); ``--vys`` adds a
venue pool (``--pool-size`` names, default 40) to every paper, which exercises the PP/VYS router instead of paper context only.
Usage: benchmark_timo.py --baseline <tree> --candidate <tree> [--runs 3] [--max-slowdown 0.15] [--vys [--pool-size 40]]
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys

WORKER = r"""
import json, random, sys, time
sys.path.insert(0, sys.argv[1])
from sinonym import ChineseNameDetector
from sinonym.timo.interface import Instance, Predictor, PredictorConfig, SourceAuthorFields
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
papers = build_papers(ChineseNameDetector())
if int(sys.argv[2]):
    rng = random.Random(7)  # noqa: S311
    pool = [f"{a.first_name} {a.last_name}" for p in papers[:300] for a in p.pp_authors][:2000]
    papers = [Instance(pp_authors=p.pp_authors, vys_other_names=rng.sample(pool, int(sys.argv[2]))) for p in papers]
pred = Predictor(PredictorConfig(parallel="never"), ".")
pred.predict_batch(papers[:100])
n = sum(len(p.pp_authors) for p in papers)
t = time.perf_counter()
for i in range(0, len(papers), 500):
    pred.predict_batch(papers[i:i + 500])
print(json.dumps({"authors_per_s": n / (time.perf_counter() - t)}))
"""


def measure(tree: str, *, pool_size: int, runs: int) -> float:
    rates = []
    for _ in range(runs):
        cmd = [sys.executable, "-c", WORKER, tree, str(pool_size)]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)  # noqa: S603
        rates.append(json.loads(out.stdout.strip().splitlines()[-1])["authors_per_s"])
    return statistics.median(rates)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-slowdown", type=float, default=0.15)
    ap.add_argument("--vys", action="store_true")
    ap.add_argument("--pool-size", type=int, default=40)
    a = ap.parse_args()
    size = a.pool_size if a.vys else 0
    base = measure(a.baseline, pool_size=size, runs=a.runs)
    cand = measure(a.candidate, pool_size=size, runs=a.runs)
    slowdown = 1 - cand / base
    print(f"baseline {base:,.0f} authors/s, candidate {cand:,.0f} authors/s, slowdown {100 * slowdown:+.1f}% "
          f"({f'PP/VYS, {size}-name pool' if size else 'paper context'}, median of {a.runs})")
    return 1 if slowdown > a.max_slowdown else 0


if __name__ == "__main__":
    sys.exit(main())
