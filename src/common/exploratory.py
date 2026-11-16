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


def rank_biserial(x_dep: np.ndarray, x_ctrl: np.ndarray) -> tuple[float, float]:
    """Rank-biserial effect size (depressed > control is positive) and two-sided Mann-Whitney p."""
    x_dep, x_ctrl = x_dep[~np.isnan(x_dep)], x_ctrl[~np.isnan(x_ctrl)]
    res = mannwhitneyu(x_dep, x_ctrl, alternative="two-sided")
    return 2 * res.statistic / (len(x_dep) * len(x_ctrl)) - 1, float(res.pvalue)


def criterion_contrast(features: pd.DataFrame, y: np.ndarray, items: pd.DataFrame) -> pd.DataFrame:
    """Per DSM-5 criterion: class separation of the self-reported PHQ-8 item and of the language score.

    features: criterion scores (dsm_C1..dsm_C8) indexed like `items`; y: binary labels in the same order.
    The item effect is large by construction (the label is a sum of items); it is a ceiling, not a result.
    """
    rows = []
    for crit in load_mapping()["criteria"]:
        lang = features[f"dsm_{crit['id']}"].to_numpy(float)
        item = items[crit["phq8_item"]].to_numpy(float)
        r_lang, p_lang = rank_biserial(lang[y == 1], lang[y == 0])
        r_item, p_item = rank_biserial(item[y == 1], item[y == 0])
        rho, p_rho = spearmanr(lang, item, nan_policy="omit")
        rows.append({"criterion": crit["id"], "label": crit["label"], "phq8_item": crit["phq8_item"],
                     "r_item": r_item, "p_item": p_item, "r_language": r_lang, "p_language": p_lang,
                     "gap": r_item - r_lang, "rho_language_item": rho, "p_rho": p_rho})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures

ITEM_COLOR, LANGUAGE_COLOR = "#898781", "#6250d6"  # series identity, distinct from the class colors


def fig_criterion_by_class(features: pd.DataFrame, y: np.ndarray, contrast: pd.DataFrame):
    """Small multiples: language criterion score by class, one panel per DSM-5 criterion."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6), sharey=False)
    for ax, (_, row) in zip(axes.ravel(), contrast.iterrows()):
        x = features[f"dsm_{row['criterion']}"].to_numpy(float) * 1000
        for cls in (0, 1):
            ax.boxplot(x[y == cls], positions=[cls], widths=0.55, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=eda.CLASS_COLORS[cls], edgecolor=eda.INK, linewidth=0.8),
                       medianprops=dict(color=eda.INK, linewidth=1.4),
                       whiskerprops=dict(color=eda.INK, linewidth=0.8), capprops=dict(color=eda.INK, linewidth=0.8))
        ax.set_xticks([0, 1], ["Ctrl", "Dep"], fontsize=8)
        stars = "*" * sum(row["p_language"] < t for t in (0.05, 0.01, 0.001))
        # Statistics go in the title block so they never collide with the whiskers.
        ax.set_title(f"{row['criterion']} {row['label']}\n"
                     f"r = {row['r_language']:.2f}{stars}   rho(item) = {row['rho_language_item']:.2f}",
                     fontsize=8, color=eda.INK)
        ax.tick_params(labelsize=7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("criterion score (x 1000)", fontsize=8, color=eda.INK)
    fig.tight_layout()
    return fig


def fig_report_vs_discourse(contrast: pd.DataFrame):
    """Dumbbell: class separation of the self-reported item vs. the language score, per criterion."""
    import matplotlib.pyplot as plt
    t = contrast.iloc[::-1]
    ypos = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hlines(ypos, t["r_language"], t["r_item"], color="#c3c2b7", linewidth=2, zorder=1)
    ax.scatter(t["r_item"], ypos, s=60, color=ITEM_COLOR, edgecolors="white", linewidths=1.5, zorder=2,
               label="PHQ-8 item (self-report)")
    ax.scatter(t["r_language"], ypos, s=60, color=LANGUAGE_COLOR, edgecolors="white", linewidths=1.5, zorder=3,
               label="Language score (Empath)")
    for yp, (_, row) in zip(ypos, t.iterrows()):
        if row["p_language"] < 0.05:
            ax.text(row["r_language"], yp + 0.28, "*", ha="center", fontsize=10, color=eda.INK)
    ax.axvline(0, color=eda.MUTED, linewidth=0.8)
    ax.set_yticks(ypos, [f"{r['criterion']} {r['label']}" for _, r in t.iterrows()], fontsize=8)
    ax.set_xlabel("class separation, rank-biserial r (depressed vs. control)", color=eda.INK)
    ax.set_xlim(-0.3, 1.0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # Legend above the plot area so it never covers a criterion row.
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


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
