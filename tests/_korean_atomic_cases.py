"""Shared reviewed cases for atomic Korean given-name tokens."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AtomicKoreanGivenCase:
    """One expected surface across scalar, batch, and TIMO adapters."""

    raw_name: str
    formatted_name: str
    formatted_given: str
    surname: str
    source_given_override: str | None = None

    @property
    def source_given(self) -> str:
        """Return the equivalent given field used by the structured adapter."""
        return self.source_given_override or self.formatted_given.replace("-", " ")


ATOMIC_KOREAN_GIVEN_CASES = (
    AtomicKoreanGivenCase("Lee Young", "Young Lee", "Young", "Lee"),
    AtomicKoreanGivenCase("So Young Yun", "Young-Yun So", "Young-Yun", "So"),
    AtomicKoreanGivenCase("Lee Hoon", "Hoon Lee", "Hoon", "Lee"),
    AtomicKoreanGivenCase("Choi Seon", "Seon Choi", "Seon", "Choi"),
    AtomicKoreanGivenCase("Hana Choi", "Hana Choi", "Hana", "Choi"),
    AtomicKoreanGivenCase("Choi Seungbo", "Seungbo Choi", "Seungbo", "Choi", "SeungBo"),
    AtomicKoreanGivenCase("Lim Woong", "Woong Lim", "Woong", "Lim"),
)
