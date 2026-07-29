"""Regression tests for conservative East Asian family-first routing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sinonym.services.east_asian_name_order import EastAsianNameOrderService

if TYPE_CHECKING:
    from sinonym import ChineseNameDetector


def test_japanese_romanized_routes_only_unambiguous_dictionary_direction(
    detector: ChineseNameDetector,
) -> None:
    routed = detector.normalize_person_name("Shirakawa Hideki")
    ambiguous = detector.normalize_person_name("Motoki Kouzaki")

    assert routed is not None
    assert routed.text == "Hideki Shirakawa"
    assert routed.normalized.given_name == "Hideki"
    assert routed.normalized.surname == "Shirakawa"
    assert routed.source.order == ("surname", "given")
    assert ambiguous is not None
    assert ambiguous.text == "Motoki Kouzaki"
    assert ambiguous.source.order == ("given", "surname")


def test_structured_japanese_romanized_matches_raw_directional_route(
    detector: ChineseNameDetector,
) -> None:
    raw = detector.normalize_person_name("Kumagai Jin")
    structured = detector.normalize_person_name_components(first_name="Kumagai", last_name="Jin")

    assert raw is not None
    assert structured is not None
    assert structured.text == raw.text == "Jin Kumagai"
    assert structured.normalized == raw.normalized
    assert structured.source_text == "Kumagai Jin"
    assert structured.source.given_name == "Kumagai"
    assert structured.source.surname == "Jin"
    assert structured.source.given_tokens == ("Kumagai",)
    assert structured.source.surname_tokens == ("Jin",)
    assert structured.source.order == ("given", "surname")


def test_structured_japanese_romanized_preserves_ambiguous_or_given_first_pairs(
    detector: ChineseNameDetector,
) -> None:
    for first_name, last_name in (("Motoki", "Kouzaki"), ("Akira", "Kurosawa")):
        surface = f"{first_name} {last_name}"
        raw = detector.normalize_person_name(surface)
        structured = detector.normalize_person_name_components(first_name=first_name, last_name=last_name)

        assert raw is not None
        assert structured is not None
        assert structured.text == raw.text == surface
        assert structured.normalized == raw.normalized
        assert structured.source.given_name == first_name
        assert structured.source.surname == last_name
        assert structured.source.order == ("given", "surname")


def test_japanese_native_uses_classifier_then_component_boundary() -> None:
    decision = EastAsianNameOrderService().infer(
        "中田英寿",
        japanese_probability=lambda _name: 1.0,
    )

    assert decision is not None
    assert decision.first_name == "英寿"
    assert decision.last_name == "中田"
    assert decision.source_order == ("surname", "given")


def test_japanese_native_preserves_when_classifier_abstains() -> None:
    decision = EastAsianNameOrderService().infer(
        "中田英寿",
        japanese_probability=lambda _name: 0.79,
    )

    assert decision is None


def test_korean_routes_strict_shapes_and_preserves_ambiguous_romanization(
    detector: ChineseNameDetector,
) -> None:
    romanized = detector.normalize_person_name("Kim Min-jun")
    ambiguous = detector.normalize_person_name("Kim Yuna")
    native = detector.normalize_person_name("김민수")

    assert romanized is not None
    assert romanized.text == "Min-jun Kim"
    assert romanized.source.order == ("surname", "given")
    assert ambiguous is not None
    assert ambiguous.text == "Kim Yuna"
    assert native is not None
    assert native.text == "민수 김"
    assert native.source.order == ("surname", "given")


def test_vietnamese_requires_unicode_evidence_and_preserves_given_span(
    detector: ChineseNameDetector,
) -> None:
    # A diacritic is what identifies a bare surname match as Vietnamese, because most of the
    # surname list is short and doubles as Korean/Chinese/Western syllables. `Le Van Thanh` stays
    # given-first; the admitted exceptions are listed in
    # ASCII_ROUTABLE_VIETNAMESE_SURNAMES and have their own tests below.
    unicode_name = detector.normalize_person_name("Nguyễn Văn An")
    ascii_name = detector.normalize_person_name("Le Van Thanh")

    assert unicode_name is not None
    assert unicode_name.text == "An Văn Nguyễn"
    assert unicode_name.normalized.given_name == "An"
    assert unicode_name.normalized.middle_name == "Văn"
    assert unicode_name.normalized.surname == "Nguyễn"
    assert unicode_name.source.order == ("surname", "middle", "given")
    assert ascii_name is not None
    assert ascii_name.text == "Le Van Thanh"


def test_bare_ascii_admitted_vietnamese_surnames_route_family_first(
    detector: ChineseNameDetector,
) -> None:
    # Each admitted token is in no other lexicon, so it cannot preempt the Korean or Japanese
    # routes, and each is near-exclusively a surname. Blind labelling put the leading token as the
    # surname in 99.3-99.8% of sampled rows.
    for surface, surname, given, middle in (
        ("Nguyen Van Hieu", "Nguyen", "Hieu", "Van"),
        ("Nguyen Minh Duc", "Nguyen", "Duc", "Minh"),
        ("Nguyen Thi Oanh", "Nguyen", "Oanh", "Thi"),
        ("NGUYEN XUAN THO", "Nguyen", "Tho", "Xuan"),
        ("Tran Quoc Khanh", "Tran", "Khanh", "Quoc"),
        ("Tran Tinh Hien", "Tran", "Hien", "Tinh"),
        ("Pham Huu Tiep", "Pham", "Tiep", "Huu"),
        ("Pham Hai Yen", "Pham", "Yen", "Hai"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.normalized.given_name == given, surface
        assert routed.normalized.middle_name == middle, surface
        assert routed.source.order == ("surname", "middle", "given"), surface


def test_hoang_is_not_admitted_where_both_tokens_are_surnames(
    detector: ChineseNameDetector,
) -> None:
    # `hoang` clears the no-other-lexicon test but fails the surname-purity one in ONE shape: where
    # both tokens of a two-token name are Vietnamese surnames it is the GIVEN name two times in
    # three, so "Hoang Nguyen" and "Hoang Pham" would be flipped backwards. That shape stays
    # declined for every head admitted on the without-surname-partner list, which is what makes the
    # rest of the list safe to admit.
    for surface in ("Hoang Nguyen", "Hoang Pham", "Hoang Tran", "Vu Nguyen", "Ngo Le"):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname != surface.split()[0], surface


def test_unlisted_bare_ascii_vietnamese_surnames_still_need_a_diacritic(
    detector: ChineseNameDetector,
) -> None:
    # The short entries stay gated. `Mai`, `Le`, `Do`, `Kim`, `Ha`, `Ho` and `Ly` are ordinary given
    # names elsewhere, and `kim`/`ha`/`ho` are Korean surnames too — admitting them here would
    # preempt the Korean route, since Vietnamese is tried first. Relaxing the whole 46-entry list
    # would move 550,469 mentions, of which `kim` alone is 181,705 ("Kim Overvad", "Kim Krisberg").
    for surface in ("Le Van Thanh", "Do Van Hung", "Mai Smith", "Ly Van Nam", "Ha Van Tien"):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.source.order != ("surname", "middle", "given"), surface
        assert routed.normalized.surname != surface.split()[0], surface


def test_nguyen_relaxation_does_not_preempt_the_korean_route(
    detector: ChineseNameDetector,
) -> None:
    # Regression guard for the routing order: _infer_vietnamese runs before _infer_korean, so a
    # laxer Vietnamese gate could swallow names the Korean guards are meant to decide. These are
    # the exact cases those guards own.
    for surface, surname in (
        ("Kim Rudolph-Lund", "Rudolph-Lund"),
        ("Ha van den Hout", "van den Hout"),
        ("Ha Jae-Sung", "Ha"),
        ("Oh Young-Jin", "Oh"),
        ("Kim Ji Hoon", "Kim"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface


def test_comma_order_remains_authoritative(detector: ChineseNameDetector) -> None:
    canonical = detector.normalize_person_name("Kim, Min-jun")

    assert canonical is not None
    assert canonical.text == "Min-jun Kim"
    assert canonical.source.order == ("surname", "given")


def test_spaced_kanji_surname_first_is_routed_given_first(
    detector: ChineseNameDetector,
) -> None:
    # Spaced native kanji was only handled in the compact form, so "佐藤 優" fell through
    # to the generic given-first assumption and swapped the roles (given "佐藤"). It is now
    # routed family-first like the compact "佐藤優", matching the given-first canonical.
    for surface, expected_text, given, surname in (
        ("佐藤 優", "優 佐藤", "優", "佐藤"),
        ("高橋 洋一", "洋一 高橋", "洋一", "高橋"),
        ("田中 太郎", "太郎 田中", "太郎", "田中"),
        ("中村 修二", "修二 中村", "修二", "中村"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.text == expected_text
        assert routed.normalized.given_name == given
        assert routed.normalized.surname == surname
        assert routed.source.order == ("surname", "given")
        # matches the compact form's routing
        compact = detector.normalize_person_name(surface.replace(" ", ""))
        assert compact is not None
        assert compact.text == expected_text


def test_spaced_kanji_given_first_or_ambiguous_is_not_double_swapped(
    detector: ChineseNameDetector,
) -> None:
    # A spaced kanji name that is already given-first (or where the reverse is also
    # dictionary-plausible) must be left as-is, not swapped back.
    for surface in ("優 佐藤", "太郎 田中"):
        result = detector.normalize_person_name(surface)
        assert result is not None
        assert result.text == surface
        assert result.source.order == ("given", "surname")


def test_spaced_han_chinese_name_not_routed_as_japanese(
    detector: ChineseNameDetector,
) -> None:
    # Chinese spaced-han names must not be swapped by the Japanese spaced-kanji handler
    # (ML-Japanese gated + native-dictionary evidence). The Chinese pipeline handles them.
    for surface in ("王 伟", "李 明", "陈 明"):
        result = detector.normalize_person_name(surface)
        assert result is not None
        assert result.text == surface
        assert result.source.order == ("given", "surname")


def test_european_diacritic_partner_not_swapped_to_east_asian(
    detector: ChineseNameDetector,
) -> None:
    # A Western given that is a CJK-surname homograph ("Kim") was swapped to the
    # family name whenever the partner carried a non-ASCII char — but Nordic/German
    # diacritics (ø/å/ö/ü) are European, not East-Asian evidence. "Kim" is a very common
    # Scandinavian GIVEN name, so these must stay given-first (given=Kim).
    for surface, given, surname in (
        ("Kim Brøsen", "Kim", "Brøsen"),
        ("Kim Møller", "Kim", "Møller"),
        ("Kim Haugbølle", "Kim", "Haugbølle"),
        ("Kim Hørslev-Petersen", "Kim", "Hørslev-Petersen"),  # was Korean-path via hyphen
        ("Kim Jørgensen", "Kim", "Jørgensen"),
        ("Kim Müller", "Kim", "Müller"),
        ("Kim Nygård", "Kim", "Nygård"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.given_name == given, surface
        assert routed.normalized.surname == surname, surface


def test_mccune_reischauer_korean_still_routes(
    detector: ChineseNameDetector,
) -> None:
    # The European-diacritic gate must NOT block McCune-Reischauer romanized Korean, whose
    # breve vowels (ŏ/ŭ) are inside the East-Asian repertoire — these must still swap
    # family-first (surname = the Korean-surname token).
    for surface, surname in (
        ("Hwang Chŏl-su", "Hwang"),
        ("Kim Sŏ-yŏn", "Kim"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.source.order == ("surname", "given"), surface


def test_vietnamese_repertoire_preserved(
    detector: ChineseNameDetector,
) -> None:
    # Well-formed Vietnamese (horn/breve/tone marks are all in-repertoire) must still route
    # surname-first; the gate only excludes European-exclusive diacritics.
    for surface, surname in (
        ("Nguyễn Văn Anh", "Nguyễn"),
        ("Trần Thị Mai", "Trần"),
        ("Nguyễn Duy Cường", "Nguyễn"),  # ư = u+horn, in repertoire
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.source.order[0] == "surname", surface


def test_non_european_noise_glyphs_do_not_block_east_asian_routing(
    detector: ChineseNameDetector,
) -> None:
    # Turkish İ, Cyrillic homoglyphs, and Hepburn/McCune macrons are NOT European-exclusive
    # diacritics, so an otherwise East-Asian name carrying one still routes family-first
    # (these were regressions under the earlier Vietnamese-repertoire whitelist).
    tin = detector.normalize_person_name("Nguyễn Khac Tin")  # clean Vietnamese control
    assert tin is not None
    assert tin.normalized.surname == "Nguyễn"
    cyrillic = detector.normalize_person_name("Nguyễn Lam Аnh")  # 'А' is Cyrillic U+0410
    assert cyrillic is not None
    assert cyrillic.normalized.surname == "Nguyễn"


def test_east_asian_order_hard_and_ambiguous_cases_characterization(
    detector: ChineseNameDetector,
) -> None:
    """HARD / AMBIGUOUS cases — documented, deliberately NOT fixed.

    A diacritic identifies a name as European but does NOT determine its order; the fix
    only declines the *forced* East-Asian swap for European-exclusive diacritics and lets
    the Western given-first default apply. The cases below have no clean linguistic
    invariant and would need an ORCID/Western-surname signal + an LLM judge (see PR19).
    """
    # (A) is gone. The pure-ASCII Western hyphenated partner used to abstain and swap here on
    # the grounds that no character rule separates it from a hyphenated Korean given name. Two
    # rules now separate most of it without a lexicon: a two-letter surname with nothing Korean
    # on the given side (labelled 34.3% wrong), and a given part longer than a romanized Korean
    # syllable (92% wrong). "Jo Leonardi-Bee", "Jo Rycroft-Malone" and "Kim Dam-Johansen" all
    # moved to test_two_letter_korean_surname_needs_a_korean_given_part and
    # test_overlong_given_part_is_a_western_surname_not_a_korean_syllable. What remains hard is
    # a short unlisted Korean syllable, still unfixed and still costing names like "Jo Hea-Soog".

    # (B) A shared diacritic (é is both French and Vietnamese) cannot disambiguate ethnicity,
    # so "Kim André" still routes as Vietnamese (surname = Kim). Documented limitation.
    andre = detector.normalize_person_name("Kim André")
    assert andre is not None
    assert andre.normalized.surname == "Kim"

    # (C) Mojibake recovered: "Cƣờng" uses U+01A3 (a corrupted "ư"). The blocklist gate
    # only declines the swap on European-EXCLUSIVE diacritics, and U+01A3 is not one, so a
    # name that is otherwise plainly Vietnamese still routes surname-first (was a casualty
    # of the earlier repertoire whitelist).
    mojibake = detector.normalize_person_name("Nguyễn Duy Cƣờng")
    assert mojibake is not None
    assert mojibake.normalized.surname == "Nguyễn"


def test_single_letter_leading_token_is_an_initial_not_a_korean_surname(
    detector: ChineseNameDetector,
) -> None:
    # "O" is the only single-letter Korean surname romanization, so every "O <hyphenated>"
    # row used to route as the surname 오 — but in bibliographic data a lone leading letter
    # is an initial, and these are Western names ("O Braun-Falco" is Otto Braun-Falco).
    for surface in (
        "O Braun-Falco",
        "O Guntinas-Lichius",
        "O Siggaard-Andersen",
        "O Lyon-Caen",
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname != "O", surface
        assert routed.source.order != ("surname", "given"), surface


def test_multi_letter_korean_surnames_still_route_family_first(
    detector: ChineseNameDetector,
) -> None:
    # The single-letter refusal must not touch the ordinary Korean routes.
    for surface, surname in (
        ("Kim Dong-il", "Kim"),
        ("Park Chan-Wook", "Park"),
        ("Lee Sang-Ho", "Lee"),
        ("Oh Young-Jin", "Oh"),
        ("Ahn Chang-Jun", "Ahn"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.source.order == ("surname", "given"), surface


def test_two_letter_korean_surname_needs_a_korean_given_part(
    detector: ChineseNameDetector,
) -> None:
    # Two-letter Korean surnames double as Western given names, and the surname lexicon alone
    # cannot separate "Jo Leonardi-Bee" (Jo is English) from "Jo Jae-Yoon" (Jo is Korean). With
    # nothing Korean on the given side these are Western people, so the leading token stays given.
    for surface, surname in (
        ("Jo Leonardi-Bee", "Leonardi-Bee"),
        ("Jo Rycroft-Malone", "Rycroft-Malone"),
        ("An Dooms-Goossens", "Dooms-Goossens"),
        ("Yu Deuerling-Zheng", "Deuerling-Zheng"),
        ("Ra Sanchez-Gomez", "Sanchez-Gomez"),
        ("Na Rodriguez-Perez", "Rodriguez-Perez"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.source.order != ("surname", "given"), surface


def test_two_letter_korean_surname_routes_when_a_given_part_is_korean(
    detector: ChineseNameDetector,
) -> None:
    # One recognised Korean given syllable is enough corroboration to keep the family-first read.
    for surface, surname in (
        ("Ha Jae-Sung", "Ha"),
        ("Jo Jae-Yoon", "Jo"),
        ("Oh Kwang-Soo", "Oh"),
        ("Yi Seon-ung", "Yi"),
        ("Ji Won Suk", "Ji"),
        ("Ho Kyung Sung", "Ho"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.source.order[0] == "surname", surface


def test_overlong_given_part_is_a_western_surname_not_a_korean_syllable(
    detector: ChineseNameDetector,
) -> None:
    # Romanized Korean syllables top out near six characters, so a longer hyphen part is a
    # Western surname element and the leading Korean-surname homograph is really a given name.
    # This reaches the lengths the two-letter rule cannot ("Kim", "Lee", "Hwang").
    for surface, surname in (
        ("Kim Rudolph-Lund", "Rudolph-Lund"),
        ("Lee Gillespie-White", "Gillespie-White"),
        ("Lee Laurent-Applegate", "Laurent-Applegate"),
        ("Kim Theilgaard-Monch", "Theilgaard-Monch"),
        ("Kim Padgett-Clarke", "Padgett-Clarke"),
        ("Min Chen-Gaddini", "Chen-Gaddini"),
        ("Kim Dam-Johansen", "Dam-Johansen"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.source.order != ("surname", "given"), surface


def test_unlisted_korean_syllables_still_route_family_first(
    detector: ChineseNameDetector,
) -> None:
    # None of these has a single part in the syllable lexicon — "jeong", "sook", "kyoung",
    # "myong", "byeong" are all missing from it. They keep routing because every part is within
    # syllable length, which is the whole point of testing length rather than lexicon membership:
    # it protects the Korean names the incomplete lexicon cannot vouch for.
    for surface, surname in (
        ("Kim Kyoung-Duck", "Kim"),
        ("Lee Byeong-Do", "Lee"),
        ("Han Myong-Sook", "Han"),
        ("Hwang Jenn-Kang", "Hwang"),
        ("Park Jeong-sook", "Park"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.source.order[0] == "surname", surface


def test_spaced_kanji_routes_on_one_sided_dictionary_evidence(
    detector: ChineseNameDetector,
) -> None:
    # The surname asset holds 2,000 entries against 69,002 given names, so most real spaced
    # kanji names match on exactly one side. Requiring both left 615,837 names / 3.74M occ
    # reordered wrongly; blind judges called that class family-first 187/187.
    for surface, surname in (
        ("三浦 耕吉郎", "三浦"),   # leading token is a known surname, 耕吉郎 unknown
        ("小川 福次郎", "小川"),
        ("山瀬 豊", "山瀬"),       # trailing token is a known given name, 山瀬 unknown
        ("松中 成浩", "松中"),
        ("梅川 尚嗣", "梅川"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.source.order == ("surname", "given"), surface


def test_spaced_kanji_abstains_when_the_evidence_is_two_sided(
    detector: ChineseNameDetector,
) -> None:
    # Two-sided evidence is the ambiguity that one-sided routing must not swallow: a
    # reverse-plausible pair, a pair of surnames, and a pair of given names all keep the
    # given-first default, which is right for the reversed inputs the corpus does contain.
    for surface, surname in (
        ("優 佐藤", "佐藤"),        # reverse plausible: given + surname
        ("和則 西﨑", "西﨑"),
        ("吉行 水畑", "水畑"),      # both tokens are known surnames
        ("智幸 小枝", "小枝"),      # both tokens are known given names
        ("歩 中野渡", "中野渡"),    # neither token is in either asset
        ("ジョン スミス", "スミス"),  # katakana Western name
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.source.order != ("surname", "given"), surface


def test_hangul_compound_surnames_take_the_second_syllable(
    detector: ChineseNameDetector,
) -> None:
    # The seven two-syllable Korean surnames: the default 1+2 split lands inside the surname, so
    # 남궁원 shipped as surname 남 (a different, much commoner surname) with given 궁원.
    for surface, surname, given in (
        ("남궁원", "남궁", "원"),
        ("황보관", "황보", "관"),
        ("제갈돈", "제갈", "돈"),
        ("사공준", "사공", "준"),
        ("선우영", "선우", "영"),
        ("서문희", "서문", "희"),
        ("독고석", "독고", "석"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.normalized.given_name == given, surface
        assert routed.source.order == ("surname", "given"), surface


def test_hangul_single_syllable_surnames_are_unchanged(detector: ChineseNameDetector) -> None:
    # 강원실 has a compound-looking head (강원 is a province) but 강 is the surname, and 남 / 황 / 서 /
    # 선 are commoner surnames than the compounds that start with them, so only an exact match moves.
    for surface, surname in (("김민수", "김"), ("이상훈", "이"), ("강원실", "강"), ("남기웅", "남"),
                             ("황영조", "황"), ("서정원", "서"), ("선동열", "선")):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface


def test_bare_ascii_vietnamese_routes_outside_the_two_surname_shape(
    detector: ChineseNameDetector,
) -> None:
    # Distinctive heads route family-first, so "Van Minh" and "Huu Tai" stop being surnames.
    for surface, surname, given in (
        ("Vu Van Quang", "Vu", "Quang"),
        ("Hoang Van Minh", "Hoang", "Minh"),
        ("Phan Van Kiem", "Phan", "Kiem"),
        ("Bui Huu Tai", "Bui", "Tai"),
        ("Huynh Quang Huy", "Huynh", "Huy"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface
        assert routed.normalized.given_name == given, surface


def test_bare_ascii_vietnamese_declines_a_pair_of_surnames_and_foreign_homographs(
    detector: ChineseNameDetector,
) -> None:
    # A trailing Vietnamese surname marks an inverted byline at any length, so the shape is declined
    # whether or not a middle token sits between the two surnames; the excluded heads stay excluded
    # because they are ordinary Korean surnames, Japanese given names or Western names.
    for surface, surname in (
        ("Vu Nguyen", "Nguyen"),
        ("Hoang Pham", "Pham"),
        ("Truong Khang Nguyen", "Nguyen"),
        ("Hoang Xuan Tran", "Tran"),
        ("Dinh Chau Phan", "Phan"),
        ("Kim Overvad", "Overvad"),
        ("Le Corbusier", "Corbusier"),
        ("Ho Jin Kim", "Kim"),
        ("Mai Sato", "Sato"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface


def test_bare_ascii_vietnamese_declines_a_hyphenated_trailing_surname(
    detector: ChineseNameDetector,
) -> None:
    # The inverted-byline marker survives hyphenation: the trailing token's first half is the family
    # name, so the whole compound is, and blind labelling put the family name last on 8 of 8 such
    # rows. A hyphen whose first half is not a surname is an ordinary given name.
    for surface, surname in (
        ("Thuong Le-Tien", "Le-Tien"),
        ("Vu Thuy Khanh Le-Trilling", "Le-Trilling"),
        ("Truong Nguyen-Ba", "Nguyen-Ba"),
        ("Hoang Le-Huu", "Le-Huu"),
        ("Dinh Vo-Ngoc", "Vo-Ngoc"),
    ):
        routed = detector.normalize_person_name(surface)
        assert routed is not None, surface
        assert routed.normalized.surname == surname, surface

    given_first_hyphen = detector.normalize_person_name("Ngo Si-Huy")
    assert given_first_hyphen is not None
    assert given_first_hyphen.normalized.surname == "Ngo"
