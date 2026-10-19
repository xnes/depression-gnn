"""Exploratory analyses behind the dissertation figures and tables, recomputed on training data only.

The dissertation pooled DAIC-WOZ and E-DAIC for some of these tests (Figures 17-18). Here every test
uses the DAIC-WOZ training cohort, so nothing that is later used for evaluation informs the
exploratory findings. All outputs are aggregates.

    empath_anova          F-test of each of the 194 Empath categories, depressed vs. control
    absolutist_tests      Mann-Whitney (depressed > control) and Spearman with the PHQ-8 total
    feature_correlations  Spearman of every professional feature with the PHQ-8 total
    criterion_direction   DSM-5 criterion scores vs. the PHQ-8 total and the matching item, with
                          the expected sign from the mapping
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, mannwhitneyu, spearmanr

from src.common import eda
from src.common.features import load_mapping


def empath_anova(cohort, categories: list[str]) -> pd.DataFrame:
    rows = []
    for j, cat in enumerate(categories):
        x = cohort.empath[:, j]
        ctrl, dep = x[cohort.y == 0], x[cohort.y == 1]
        f, p = f_oneway(ctrl, dep) if x.std() > 0 else (np.nan, np.nan)
        rows.append({"category": cat, "F": f, "p": p, "mean_control": ctrl.mean(),
                     "mean_depressed": dep.mean()})
    return pd.DataFrame(rows).sort_values("F", ascending=False).reset_index(drop=True)


def absolutist_tests(cohort) -> dict:
    rate = cohort.features.loc[cohort.ids, "absolutist_rate"].to_numpy()
    mw = mannwhitneyu(rate[cohort.y == 1], rate[cohort.y == 0], alternative="greater")
    rho, p = spearmanr(rate, cohort.phq)
    return {"n": len(rate), "median_control": float(np.median(rate[cohort.y == 0])),
            "median_depressed": float(np.median(rate[cohort.y == 1])),
            "mannwhitney_U": float(mw.statistic), "mannwhitney_p_one_sided": float(mw.pvalue),
            "spearman_rho_phq8": float(rho), "spearman_p": float(p)}


def feature_correlations(cohort, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        x = cohort.features.loc[cohort.ids, col].to_numpy(float)
        ok = ~np.isnan(x)
        if ok.sum() < 3 or np.nanstd(x) == 0:
            rows.append({"feature": col, "rho": np.nan, "p": np.nan, "n": int(ok.sum())})
            continue
        rho, p = spearmanr(x[ok], cohort.phq[ok])
        rows.append({"feature": col, "rho": rho, "p": p, "n": int(ok.sum())})
    return pd.DataFrame(rows).sort_values("rho", key=np.abs, ascending=False).reset_index(drop=True)


def criterion_direction(cohort, items: pd.DataFrame | None = None) -> pd.DataFrame:
    """Each criterion score is mean(present) - mean(absent), so the expected sign is positive.

    items: optional PHQ-8 item scores indexed by Participant_ID (from eda.build_session_table).
    """
    mapping = load_mapping()
    rows = []
    for crit in mapping["criteria"]:
        x = cohort.features.loc[cohort.ids, f"dsm_{crit['id']}"].to_numpy(float)
        rho_t, p_t = spearmanr(x, cohort.phq)
        row = {"criterion": crit["id"], "label": crit["label"], "rho_total": rho_t, "p_total": p_t}
        if items is not None:
            item = items.loc[cohort.ids, crit["phq8_item"]].to_numpy(float)
            rho_i, p_i = spearmanr(x, item, nan_policy="omit")  # some training rows lack item scores
            row.update({"phq8_item": crit["phq8_item"], "rho_item": rho_i, "p_item": p_i})
        row["direction_as_expected"] = bool(rho_t > 0)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures

def fig_empath_anova(anova: pd.DataFrame, top: int = 20):
    import matplotlib.pyplot as plt
    mapped = {c for crit in load_mapping()["criteria"] for c in crit["present"] + crit["absent"]}
    t = anova.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * top + 1))
    colors = ["#2a78d6" if c in mapped else "#c3c2b7" for c in t["category"]]
    ax.barh(t["category"], t["F"], color=colors, height=0.7)
    ax.set_xlabel("ANOVA F (depressed vs. control), training split", color=eda.INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#2a78d6"), plt.Rectangle((0, 0), 1, 1, color="#c3c2b7")]
    ax.legend(handles, ["in DSM-5 mapping", "not in mapping"], frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def fig_absolutist(cohort):
    import matplotlib.pyplot as plt
    rate = 1000 * cohort.features.loc[cohort.ids, "absolutist_rate"].to_numpy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
    for cls in (0, 1):
        a1.boxplot(rate[cohort.y == cls], positions=[cls], widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor=eda.CLASS_COLORS[cls], edgecolor=eda.INK),
                   medianprops=dict(color=eda.INK))
    a1.set_xticks([0, 1], [eda.CLASS_NAMES[0], eda.CLASS_NAMES[1]])
    eda._style(a1, "absolutist words per 1k words")
    a2.scatter(cohort.phq, rate, s=12, c=[eda.CLASS_COLORS[int(v)] for v in cohort.y],
               edgecolors="white", linewidths=0.3)
    a2.axvline(10, color=eda.INK, linestyle="--", linewidth=0.8)
    a2.set_xlabel("PHQ-8 total score", color=eda.INK)
    eda._style(a2, "absolutist words per 1k words")
    fig.tight_layout()
    return fig


def fig_rho_bars(table: pd.DataFrame, label_col: str, rho_col: str = "rho", xlabel: str = "Spearman rho with PHQ-8 total"):
    import matplotlib.pyplot as plt
    t = table.dropna(subset=[rho_col]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.3 * len(t) + 1))
    ax.barh(t[label_col], t[rho_col], color=["#2a78d6" if v >= 0 else "#eb6834" for v in t[rho_col]], height=0.7)
    ax.axvline(0, color=eda.MUTED, linewidth=0.8)
    ax.set_xlabel(xlabel, color=eda.INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig
