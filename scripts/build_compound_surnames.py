#!/usr/bin/env python
"""Build ``sinonym/data/compound_surnames.csv.gz`` from corpus evidence (see data/COMPOUND_SURNAMES.md).

Inputs (all produced outside this repo, from a pre-normalization dump of production ``paper_authors``):
  --partition   partition_prior.parquet: per surname-tail surface, mention totals, distinct given names and the
                share of mentions that write the whole surface in ``last`` (``share_whole``)
  --tokens      partition_prior_tokens.parquet: per word, mentions as a given name vs as the surname word
  --orcid-gate  compound_orcid_gate.parquet: per surface, distinct recurring-ORCID people carrying it whole,
                split, or both (the evaluation ORCIDs excluded)

A surface ships when the mention-count evidence and the person-level evidence agree (tier ``listed``):
  2-3 words; >= MIN_MENTIONS mentions; whole-in-last share >= MIN_WHOLE_SHARE; >= MIN_GIVEN_NAMES distinct given
  names; no head word used as a given name more than MAX_GIVEN_SHARE of the time (family particles exempt); every
  word >= 2 letters; tail not a generational suffix; head not an East Asian surname nor a Chinese compound head;
  not a Vietnamese ``van X`` / ``thi X`` pair; and >= MIN_PEOPLE distinct ORCID people seen with it, of whom
  >= MIN_PEOPLE_WHOLE wrote it whole.

Tier ``vocabulary`` generalizes the listed tier to the Hispanic / Lusophone double-surname convention: a 2-word
tail whose head word opens >= VOCAB_MIN_SURFACES listed 2-word surfaces (``garcia``, ``rivera``, ...) and whose
tail word closes as many (``lopez``, ``garcia``, ...) is a double surname even when the pair itself was split
too often to be listed (``garcia garcia``). It needs >= VOCAB_MIN_MENTIONS mentions and the head-word given-name
gate, and a whole-in-last share >= VOCAB_MIN_WHOLE_SHARE: below that (``silva santos`` 0.09, ``pereira silva`` 0.08)
the corpus-scale ORCID ledger shows the join splitting almost as many people as it unites (2.1:1), because the pair
is really two people's conventions, not one surname. Letting any plain surname word be the tail was tried and rejected (690k arm: +268 people united, +105
split, 2.6:1): a rare pair's other sources usually abbreviate the head (``Lara / G. / Merino``), so a pair that
the corpus itself rarely writes whole is not worth joining. ``rivera maza`` therefore stays a known limit.

Tier ``corpus`` admits every 2-3 word tail the corpus itself writes whole in ``last`` at least CORPUS_MIN_MENTIONS *
CORPUS_MIN_WHOLE_SHARE times (>= CORPUS_MIN_MENTIONS mentions, whole share >= CORPUS_MIN_WHOLE_SHARE, >=
CORPUS_MIN_GIVEN_NAMES given names), with the same head-word and East Asian gates. On the 690k index 60% of the
"surname word dropped" splits were really boundary placements (the word in ``middle`` on one mention, in ``last`` on
another) whose surface was in the corpus but below the listed tier's thresholds.

No tier admits a 2-word tail that repeats one word (``silva silva``, ``velasco velasco``): a real doubled surname is
rare, and the corpus-scale ledger shows such joins splitting 108 ORCID people for 177 united (1.6:1); the usual
source of the shape is a middle name that repeats the surname.

The asset carries only the surface, its tier and rounded counts. Rebuild with
  uv run python scripts/build_compound_surnames.py --partition ... --tokens ... --orcid-gate ... \\
      --output sinonym/data/compound_surnames.csv.gz
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
from pathlib import Path

import duckdb

MIN_MENTIONS = 50
MIN_WHOLE_SHARE = 0.25
MIN_GIVEN_NAMES = 5
MAX_GIVEN_SHARE = 0.3
MIN_PEOPLE = 3
MIN_PEOPLE_WHOLE = 2
VOCAB_MIN_SURFACES = 5
VOCAB_MIN_MENTIONS = 5
VOCAB_MIN_WHOLE_SHARE = 0.15
CORPUS_MIN_MENTIONS = 5
CORPUS_MIN_WHOLE_SHARE = 0.2
CORPUS_MIN_GIVEN_NAMES = 2
SUFFIX_TAILS = ("junior", "jr", "sr", "senior", "filho", "neto", "sobrinho", "ii", "iii", "iv")
# family particles are allowed as head words even though "da" / "van" / "de" also occur as given names elsewhere
PARTICLES = (
    "de", "del", "della", "delle", "dello", "degli", "dei", "di", "da", "das", "dos", "do", "du", "des", "van", "von",
    "der", "den", "ter", "ten", "te", "la", "le", "les", "lo", "el", "al", "bin", "ibn", "abu", "mc", "mac", "o", "st",
    "saint", "san", "santa", "sao", "fitz", "y", "e", "i",
)
# Chinese compound surnames (Au Yeung, Ou Yang, Sze To, ...) stay with sinonym's curated list
CHINESE_COMPOUND_HEADS = (
    "au", "ou", "sze", "szeto", "situ", "duanmu", "zhuge", "sima", "dongfang", "shangguan", "gongsun", "xiahou", "ouyang",
)
EAST_ASIAN_HEADS = (
    "nguyen", "tran", "le", "pham", "hoang", "huynh", "phan", "vu", "vo", "dang", "bui", "do", "ho", "ngo", "duong",
    "ly", "dinh", "trinh", "mai", "dao", "doan", "luong", "ta", "truong", "lam", "cao", "wang", "li", "zhang", "liu",
    "chen", "yang", "huang", "zhao", "wu", "zhou", "xu", "sun", "ma", "zhu", "hu", "guo", "he", "lin", "gao", "luo",
    "zheng", "liang", "xie", "song", "tang", "han", "feng", "deng", "peng", "zeng", "xiao", "tian", "dong", "pan",
    "yuan", "yu", "ye", "jiang", "fu", "fang", "shi", "lu", "kim", "lee", "park", "choi",
)
# "Van Le", "Thi Tran": a Vietnamese middle particle before the surname, not a compound surname
VIETNAMESE_MIDDLE = ("van", "thi")
VIETNAMESE_TAILS = (
    *EAST_ASIAN_HEADS,
    "viet", "duc", "minh", "anh", "hung", "thanh", "tuan", "quang", "hai", "hong", "long", "nam", "son", "binh",
    "cong", "chi", "dung", "hieu", "huy", "khanh", "hoa", "thu", "lan", "linh", "ngoc", "phuong", "thao", "trang",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--orcid-gate", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    con = duckdb.connect()
    sfx = ",".join(f"'{t}'" for t in SUFFIX_TAILS)
    heads = ",".join(f"'{t}'" for t in EAST_ASIAN_HEADS)
    particles = ",".join(f"'{t}'" for t in PARTICLES)
    cn_heads = ",".join(f"'{t}'" for t in CHINESE_COMPOUND_HEADS)
    vn_middle = ",".join(f"'{t}'" for t in VIETNAMESE_MIDDLE)
    vn_tails = ",".join(f"'{t}'" for t in VIETNAMESE_TAILS)
    doubled = "(len(string_split(tail, ' ')) = 2 AND string_split(tail, ' ')[1] = string_split(tail, ' ')[2])"
    vietnamese = (
        f"(len(string_split(tail, ' ')) = 2 AND string_split(tail, ' ')[1] IN ({vn_middle})"
        f" AND string_split(tail, ' ')[2] IN ({vn_tails}))"
    )
    query = f"""
        WITH p AS (
          SELECT tail, n_total, share_whole, firsts FROM '{a.partition}'
          WHERE words BETWEEN 2 AND 3 AND n_total >= {MIN_MENTIONS} AND share_whole >= {MIN_WHOLE_SHARE}
            AND firsts >= {MIN_GIVEN_NAMES}
            AND regexp_extract(tail, '([^ ]+)$', 1) NOT IN ({sfx})
            AND string_split(tail, ' ')[1] NOT IN ({heads})
            AND NOT {doubled}
        ),
        p2 AS (
          SELECT p.* FROM p
          WHERE NOT EXISTS (SELECT 1 FROM unnest(string_split(p.tail, ' ')) AS u(w) JOIN '{a.tokens}' t ON t.t = u.w
                            WHERE u.w <> regexp_extract(p.tail, '([^ ]+)$', 1) AND u.w NOT IN ({particles})
                              AND t.given_share > {MAX_GIVEN_SHARE})
            AND NOT EXISTS (SELECT 1 FROM unnest(string_split(p.tail, ' ')) AS u(w) WHERE length(u.w) < 2)
            AND string_split(p.tail, ' ')[1] NOT IN ({cn_heads})
            AND NOT {vietnamese}
        )
        SELECT p2.tail, p2.n_total, p2.share_whole, p2.firsts, g.people, g.people_whole, g.people_both
        FROM p2 JOIN '{a.orcid_gate}' g USING (tail)
        WHERE g.people >= {MIN_PEOPLE} AND g.people_whole >= {MIN_PEOPLE_WHOLE}
        ORDER BY g.people DESC, p2.n_total DESC"""  # noqa: S608
    con.execute(f"CREATE TABLE listed AS {query}")
    vocab = con.execute(f"""
        WITH two AS (SELECT string_split(tail, ' ') w FROM listed WHERE len(string_split(tail, ' ')) = 2),
             heads AS (SELECT w[1] h FROM two GROUP BY 1 HAVING COUNT(*) >= {VOCAB_MIN_SURFACES}),
             tails AS (SELECT w[2] t FROM two GROUP BY 1 HAVING COUNT(*) >= {VOCAB_MIN_SURFACES})
        SELECT p.tail, p.n_total, p.share_whole, p.firsts
        FROM '{a.partition}' p
        JOIN heads ON heads.h = string_split(p.tail, ' ')[1]
        JOIN tails ON tails.t = string_split(p.tail, ' ')[2]
        JOIN '{a.tokens}' th ON th.t = heads.h
        WHERE p.words = 2 AND p.n_total >= {VOCAB_MIN_MENTIONS} AND p.share_whole >= {VOCAB_MIN_WHOLE_SHARE}
          AND p.tail NOT IN (SELECT tail FROM listed) AND heads.h <> tails.t
          AND (heads.h IN ({particles}) OR th.given_share <= {MAX_GIVEN_SHARE})
          AND heads.h NOT IN ({heads}) AND tails.t NOT IN ({vn_tails})
          AND NOT EXISTS (SELECT 1 FROM unnest(string_split(p.tail, ' ')) AS u(w) WHERE length(u.w) < 2)
        ORDER BY p.n_total DESC""").fetchall()  # noqa: S608
    corpus = con.execute(f"""
        SELECT p.tail, p.n_total, p.share_whole, p.firsts
        FROM '{a.partition}' p
        JOIN '{a.tokens}' th ON th.t = string_split(p.tail, ' ')[1]
        WHERE p.words BETWEEN 2 AND 3 AND p.n_total >= {CORPUS_MIN_MENTIONS} AND p.share_whole >= {CORPUS_MIN_WHOLE_SHARE}
          AND p.firsts >= {CORPUS_MIN_GIVEN_NAMES}
          AND p.tail NOT IN (SELECT tail FROM listed)
          AND (string_split(p.tail, ' ')[1] IN ({particles}) OR th.given_share <= {MAX_GIVEN_SHARE})
          AND string_split(p.tail, ' ')[1] NOT IN ({heads}) AND string_split(p.tail, ' ')[-1] NOT IN ({vn_tails})
          AND string_split(p.tail, ' ')[1] NOT IN ({cn_heads})
          AND NOT {vietnamese}
          AND NOT {doubled.replace("tail", "p.tail")}
          AND regexp_extract(p.tail, '([^ ]+)$', 1) NOT IN ({sfx})
          AND NOT EXISTS (SELECT 1 FROM unnest(string_split(p.tail, ' ')) AS u(w) WHERE length(u.w) < 2)
        ORDER BY p.n_total DESC""").fetchall()  # noqa: S608
    vocab = [v for v in vocab if v[0] not in {c[0] for c in corpus}]
    rows = con.execute("SELECT * FROM listed").fetchall()
    out = Path(a.output)
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["surface", "tier", "mentions", "whole_share", "given_names", "people", "people_whole", "people_split_today"])
        for tail, n, share, firsts, people, people_whole, both in rows:
            w.writerow([tail, "listed", int(n), round(share, 2), int(firsts), int(people), int(people_whole), int(both)])
        for tail, n, share, firsts in vocab:
            w.writerow([tail, "vocabulary", int(n), round(share, 2), int(firsts), "", "", ""])
        for tail, n, share, firsts in corpus:
            w.writerow([tail, "corpus", int(n), round(share, 2), int(firsts), "", "", ""])
    digest = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    print(f"{len(rows):,} listed + {len(vocab):,} vocabulary + {len(corpus):,} corpus surfaces -> {out} (sha256 {digest})")
    print("top by people split today:", [r[0] for r in sorted(rows, key=lambda r: -r[6])[:12]])


if __name__ == "__main__":
    main()
