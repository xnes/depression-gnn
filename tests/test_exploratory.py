"""Criterion contrast (self-report vs. discourse) on synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import exploratory as ex
from src.common.data import PHQ8_ITEMS
from src.common.features import load_mapping


def _data(n: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    ids = np.arange(n)
    mapping = load_mapping()
    items = pd.DataFrame({it: np.clip(np.round(rng.normal(0.5 + 1.5 * y, 0.5)), 0, 3) for it in PHQ8_ITEMS}, index=ids)
    items.iloc[0, 0] = np.nan                                   # a missing item score is tolerated
    feats = pd.DataFrame(index=ids)
    for crit in mapping["criteria"]:
        signal = 0.8 if crit["id"] == "C1" else 0.0            # only C1 is expressed in language
        feats[f"dsm_{crit['id']}"] = rng.normal(signal * y, 1.0) / 1000
    return feats, y, items


def test_criterion_contrast_shows_the_gap() -> None:
    feats, y, items = _data()
    c = ex.criterion_contrast(feats, y, items).set_index("criterion")
    assert list(c.index) == [f"C{i}" for i in range(1, 9)]
    assert (c["r_item"] > 0.5).all()                           # self-report separates the classes
    assert c.loc["C1", "r_language"] > 0.2 and c.loc["C1", "p_language"] < 0.05
    assert c.loc["C2", "gap"] > c.loc["C1", "gap"]             # no language signal -> larger gap


def test_rank_biserial_bounds() -> None:
    r, p = ex.rank_biserial(np.array([5.0, 6, 7]), np.array([1.0, 2, 3]))
    assert r == 1.0
    r, _ = ex.rank_biserial(np.array([1.0, 2, 3]), np.array([1.0, 2, 3]))
    assert r == 0.0


def test_figures_render() -> None:
    feats, y, items = _data()
    c = ex.criterion_contrast(feats, y, items)
    assert ex.fig_criterion_by_class(feats, y, c).axes
    assert ex.fig_report_vs_discourse(c).axes
