"""Absolutist-word index (Al-Mosaiwi and Johnstone, 2018) on synthetic sentences."""
import pytest

from src.common import features as fx


def test_dictionary_is_the_19_word_list() -> None:
    assert len(fx.ABSOLUTIST_TERMS) == 19 == len(set(fx.ABSOLUTIST_TERMS))
    assert not {"worthless", "hopeless", "terrible", "awful"} & set(fx.ABSOLUTIST_TERMS)


def test_counts_over_total_words() -> None:
    # 10 words, 3 absolutist (always, every, whole).
    assert fx.absolutist_rate("i always eat every meal of the whole day here") == pytest.approx(3 / 10)
    assert fx.absolutist_rate("the cat sat on the mat") == 0


def test_whole_words_only() -> None:
    assert fx.absolutist_rate("allowance fully musty eventually wholesome") == 0
    # "every" must not count inside "everyone", which counts once on its own.
    assert fx.absolutist_rate("everyone everybody") == pytest.approx(1 / 2)
    assert fx.absolutist_rate("all, always. never!") == pytest.approx(3 / 3)


def test_case_insensitive() -> None:
    assert fx.absolutist_rate("ALL Always NeVeR") == pytest.approx(1.0)


def test_empty_text() -> None:
    assert fx.absolutist_rate("") == 0
    assert fx.absolutist_rate("<laughter> [sigh]") == 0
