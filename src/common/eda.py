"""Exploratory analysis of the DAIC-WOZ and E-DAIC cohorts.

Every output is an aggregate: per-cohort tables and distribution figures. No transcript text,
token list, or per-participant row is written to disk.

Cohorts:
    DAIC-WOZ train / dev / test   official AVEC 2017 partitions, manual transcripts
    E-DAIC                        ASR transcripts; sessions whose ID is in the DAIC-WOZ training
                                  split are excluded (the rule used in the legacy notebooks)
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.common import data

COHORTS = ["DAIC-WOZ train", "DAIC-WOZ dev", "DAIC-WOZ test", "E-DAIC"]
CLASS_NAMES = {0: "Control", 1: "Depressed"}
CLASS_COLORS = {0: "#2a78d6", 1: "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
DIVERGING = LinearSegmentedColormap.from_list("orange_gray_blue", ["#eb6834", "#f0efec", "#2a78d6"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#f0efec", "#2a78d6", "#104281"])

TAG_RE = re.compile(r"<[^>]*>|\[[^\]]*\]")
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with paralinguistic tags removed."""
    return WORD_RE.findall(TAG_RE.sub(" ", text.lower()))


def session_stats(turns: pd.DataFrame) -> tuple[dict, set[str]]:
    """Aggregate statistics of one session and its vocabulary (kept in memory only)."""
    text = " ".join(turns["text"].astype(str))
    tokens = tokenize(text)
    n_tags = len(TAG_RE.findall(text))
    speech = (turns["end"] - turns["start"]).clip(lower=0).sum(min_count=1)
    stats = {
        "n_turns": len(turns),
        "n_words": len(tokens),
        "speech_min": speech / 60 if pd.notna(speech) else np.nan,
        "n_tags": n_tags,
        "has_tags": n_tags > 0,
        "tags_per_1k_words": 1000 * n_tags / len(tokens) if tokens else np.nan,
    }
    if "asr_confidence" in turns:
        stats["asr_confidence"] = turns["asr_confidence"].mean()
    return stats, set(tokens)


def build_session_table(root: Path) -> pd.DataFrame:
    """One row per labeled participant: cohort, labels, and session statistics.

    Adds the out-of-vocabulary rate against the DAIC-WOZ training vocabulary, a proxy for
    lexical shift (including ASR errors) between training and evaluation transcripts.
    """
    daic = data.load_daic_labels(root)
    edaic = data.load_edaic_labels(root)
    train_ids = set(daic.loc[daic["split"] == "train", "Participant_ID"])
    edaic = edaic[~edaic["Participant_ID"].isin(train_ids)]

    rows, vocabularies = [], {}
    sources = [(daic, "DAIC-WOZ", data.load_daic_turns), (edaic, "E-DAIC", data.load_edaic_turns)]
    for labels, corpus, loader in sources:
        for rec in labels.to_dict("records"):
            pid = rec["Participant_ID"]
            cohort = f"DAIC-WOZ {rec['split']}" if corpus == "DAIC-WOZ" else "E-DAIC"
            row = {"Participant_ID": pid, "cohort": cohort,
                   "PHQ8_Score": rec["PHQ8_Score"], "PHQ8_Binary": rec["PHQ8_Binary"],
                   "Gender": rec.get("Gender", np.nan),
                   **{item: rec.get(item, np.nan) for item in data.PHQ8_ITEMS}}
            turns = loader(root, pid)
            row["transcript_available"] = turns is not None and len(turns) > 0
            if row["transcript_available"]:
                stats, vocab = session_stats(turns)
                row.update(stats)
                vocabularies[(cohort, pid)] = vocab
            rows.append(row)
    table = pd.DataFrame(rows)

    train_vocab = set().union(*(v for (c, _), v in vocabularies.items() if c == "DAIC-WOZ train"))
    table["oov_rate"] = [
        np.nan if c == "DAIC-WOZ train" or (c, p) not in vocabularies or not vocabularies[(c, p)]
        else len(vocabularies[(c, p)] - train_vocab) / len(vocabularies[(c, p)])
        for c, p in zip(table["cohort"], table["Participant_ID"])
    ]
    return table


def _iqr(s: pd.Series) -> str:
    s = s.dropna()
    if s.empty:
        return "n/a"
    q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{q2:.0f} [{q1:.0f}-{q3:.0f}]"


def cohort_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per-cohort characteristics for the dataset table of the paper."""
    out = {}
    for cohort in COHORTS:
        t = table[table["cohort"] == cohort]
        if t.empty:
            continue
        a = t[t["transcript_available"]]
        out[cohort] = {
            "labeled sessions": len(t),
            "sessions with transcript": len(a),
            "depressed (PHQ-8 >= 10)": f"{int(a['PHQ8_Binary'].sum())} ({100 * a['PHQ8_Binary'].mean():.1f}%)",
            "PHQ-8 mean (SD)": f"{a['PHQ8_Score'].mean():.1f} ({a['PHQ8_Score'].std():.1f})",
            "item scores available": bool(a[data.PHQ8_ITEMS].notna().all(axis=None)) if len(a) else False,
            "Gender coded 0 / 1": (f"{int((a['Gender'] == 0).sum())} / {int((a['Gender'] == 1).sum())}"
                                   if a["Gender"].notna().any() else "n/a"),
            "participant turns, median [IQR]": _iqr(a["n_turns"]),
            "words, median [IQR]": _iqr(a["n_words"]),
            "speech minutes, median [IQR]": _iqr(a["speech_min"]),
            "sessions with >= 1 tag (%)": f"{100 * a['has_tags'].mean():.1f}",
            "tags per 1k words, median": f"{a['tags_per_1k_words'].median():.1f}",
            "OOV vs train vocabulary, median (%)": ("n/a" if a["oov_rate"].isna().all()
                                                    else f"{100 * a['oov_rate'].median():.1f}"),
            "ASR confidence, median": (f"{a['asr_confidence'].median():.2f}"
                                       if "asr_confidence" in a and a["asr_confidence"].notna().any()
                                       else "n/a"),
        }
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- figures

def _style(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK)


def _paired_boxes(ax, table: pd.DataFrame, column: str, cohorts: list[str]) -> None:
    """Side-by-side box plots per cohort, one box per class."""
    width = 0.34
    for k, cls in enumerate((0, 1)):
        values = [table[(table["cohort"] == c) & (table["PHQ8_Binary"] == cls)][column].dropna()
                  for c in cohorts]
        pos = np.arange(len(cohorts)) + (k - 0.5) * (width + 0.04)
        ax.boxplot(values, positions=pos, widths=width, patch_artist=True, showfliers=True,
                   boxprops=dict(facecolor=CLASS_COLORS[cls], edgecolor=INK, linewidth=0.8),
                   medianprops=dict(color=INK, linewidth=1.5),
                   whiskerprops=dict(color=INK, linewidth=0.8), capprops=dict(color=INK, linewidth=0.8),
                   flierprops=dict(marker="o", markersize=3, markerfacecolor=CLASS_COLORS[cls],
                                   markeredgecolor=INK, markeredgewidth=0.4))
    ax.set_xticks(range(len(cohorts)), cohorts)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=CLASS_COLORS[c], edgecolor=INK) for c in (0, 1)]
    ax.legend(handles, [CLASS_NAMES[0], CLASS_NAMES[1]], frameon=False, loc="upper right")


def fig_phq8_distribution(table: pd.DataFrame):
    t = table[table["transcript_available"]]
    cohorts = [c for c in COHORTS if (t["cohort"] == c).any()]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    values = [t[t["cohort"] == c]["PHQ8_Score"] for c in cohorts]
    parts = ax.violinplot(values, positions=range(len(cohorts)), showextrema=False, widths=0.8)
    for body in parts["bodies"]:
        body.set_facecolor("#c3c2b7")
        body.set_edgecolor(MUTED)
        body.set_alpha(1)
    rng = np.random.default_rng(0)
    for i, c in enumerate(cohorts):
        tc = t[t["cohort"] == c]
        x = i + rng.uniform(-0.12, 0.12, len(tc))
        ax.scatter(x, tc["PHQ8_Score"], s=9, c=[CLASS_COLORS[int(b)] for b in tc["PHQ8_Binary"]],
                   edgecolors="white", linewidths=0.3, zorder=3)
        ax.text(i, 25.6, f"n={len(tc)}\n{100 * tc['PHQ8_Binary'].mean():.0f}% dep.",
                ha="center", va="bottom", fontsize=8, color=INK)
    ax.axhline(data.PHQ8_CUTOFF, color=INK, linestyle="--", linewidth=0.8)
    ax.text(len(cohorts) - 0.5, data.PHQ8_CUTOFF + 0.3, "cutoff = 10", fontsize=8, color=INK, ha="right")
    ax.set_xticks(range(len(cohorts)), cohorts)
    ax.set_ylim(-1, 30)
    _style(ax, "PHQ-8 total score")
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=5,
                          markerfacecolor=CLASS_COLORS[c], markeredgecolor="white") for c in (0, 1)]
    ax.legend(handles, [CLASS_NAMES[0], CLASS_NAMES[1]], frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def fig_phq8_item_profile(table: pd.DataFrame):
    """Mean PHQ-8 item score by cohort and class (cohorts without item scores are omitted)."""
    t = table[table["transcript_available"]]
    columns, matrix = [], []
    for c in COHORTS:
        tc = t[t["cohort"] == c]
        if tc.empty or tc[data.PHQ8_ITEMS].isna().all(axis=None):
            continue
        for cls in (0, 1):
            columns.append(f"{c}\n{CLASS_NAMES[cls]}")
            matrix.append(tc[tc["PHQ8_Binary"] == cls][data.PHQ8_ITEMS].mean().values)
    matrix = np.array(matrix).T
    fig, ax = plt.subplots(figsize=(1.1 * len(columns) + 2.2, 3.8))
    im = ax.imshow(matrix, cmap=SEQUENTIAL, vmin=0, vmax=3, aspect="auto")
    for (i, j), v in np.ndenumerate(matrix):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                color="white" if v > 1.8 else INK)
    ax.set_xticks(range(len(columns)), columns, fontsize=8)
    ax.set_yticks(range(len(data.PHQ8_ITEMS)), [s.replace("PHQ8_", "") for s in data.PHQ8_ITEMS])
    fig.colorbar(im, ax=ax, label="mean item score (0-3)", shrink=0.8)
    fig.tight_layout()
    return fig


def fig_phq8_item_correlation(table: pd.DataFrame):
    """Spearman correlation among PHQ-8 items and total, one panel per cohort with items."""
    t = table[table["transcript_available"]]
    cohorts = [c for c in COHORTS if (t["cohort"] == c).any()
               and t[t["cohort"] == c][data.PHQ8_ITEMS].notna().all(axis=None)]
    cols = data.PHQ8_ITEMS + ["PHQ8_Score"]
    labels = [s.replace("PHQ8_", "") for s in cols]
    fig, axes = plt.subplots(1, len(cohorts), figsize=(4.2 * len(cohorts), 4), squeeze=False)
    for ax, c in zip(axes[0], cohorts):
        corr = t[t["cohort"] == c][cols].corr(method="spearman").values
        im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
        for (i, j), v in np.ndenumerate(corr):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color=INK)
        ax.set_xticks(range(len(cols)), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(cols)), labels, fontsize=7)
        ax.set_title(c, fontsize=9, color=INK)
    fig.colorbar(im, ax=axes[0].tolist(), label="Spearman rho", shrink=0.8)
    return fig


def fig_session_length(table: pd.DataFrame):
    t = table[table["transcript_available"]]
    cohorts = [c for c in COHORTS if (t["cohort"] == c).any()]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    _paired_boxes(ax, t, "n_words", cohorts)
    _style(ax, "participant words per session")
    fig.tight_layout()
    return fig


def fig_transcript_quality(table: pd.DataFrame):
    """Two panels: paralinguistic tag rate and OOV rate against the training vocabulary."""
    t = table[table["transcript_available"]]
    cohorts = [c for c in COHORTS if (t["cohort"] == c).any()]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    _paired_boxes(a1, t, "tags_per_1k_words", cohorts)
    _style(a1, "paralinguistic tags per 1k words")
    eval_cohorts = [c for c in cohorts if c != "DAIC-WOZ train"]
    _paired_boxes(a2, t.assign(oov_pct=100 * t["oov_rate"]), "oov_pct", eval_cohorts)
    _style(a2, "word types outside train vocabulary (%)")
    for ax in (a1, a2):
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    return fig


FIGURES = {
    "fig_phq8_distribution": fig_phq8_distribution,
    "fig_phq8_item_profile": fig_phq8_item_profile,
    "fig_phq8_item_correlation": fig_phq8_item_correlation,
    "fig_session_length": fig_session_length,
    "fig_transcript_quality": fig_transcript_quality,
}


def run_eda(root: Path, out_dir: Path) -> pd.DataFrame:
    """Build the session table in memory and write the cohort summary and figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table = build_session_table(root)
    summary = cohort_summary(table)
    summary.to_csv(out_dir / "cohort_summary.csv")
    (out_dir / "cohort_summary.tex").write_text(summary.to_latex(escape=True), encoding="utf-8")
    for name, make in FIGURES.items():
        fig = make(table)
        fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    return summary
