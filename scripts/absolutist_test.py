"""Absolutist-word index on the training pool (DAIC-WOZ train + dev), depressed vs. control.

Writes aggregates only:
    <out_dir>/absolutist_test_pool.json   one-sided Mann-Whitney, rank-biserial r, class means/medians
    <out_dir>/absolutist_terms_pool.csv   per-term occurrences per 1k words, by class

    DAIC_DATA_DIR=... PYTHONPATH=. python scripts/absolutist_test.py results/eda
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from src.common import data
from src.common.features import ABSOLUTIST_TERMS, _WORD, absolutist_rate, clean_text

# Fixed expressions in which "all" or "ever" is not absolutist; counted to size their share.
EXPRESSIONS = ["all right", "at all", "ever since"]


def pool_texts(root: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    labels = data.load_daic_labels(root)
    labels = labels[labels["split"].isin(["train", "dev"])]
    texts, y, phq = [], [], []
    for row in labels.itertuples():
        turns = data.load_daic_turns(root, row.Participant_ID)
        if turns is None or turns.empty:
            continue
        texts.append(" ".join(turns["text"].astype(str)))
        y.append(row.PHQ8_Binary)
        phq.append(row.PHQ8_Score)
    return texts, np.array(y), np.array(phq)


def term_table(texts: list[str], y: np.ndarray) -> pd.DataFrame:
    patterns = {t: re.compile(rf"(?<![a-z']){re.escape(t)}(?![a-z'])") for t in ABSOLUTIST_TERMS + EXPRESSIONS}
    cleaned = [clean_text(t) for t in texts]
    words = np.array([len(_WORD.findall(c)) for c in cleaned])
    rows = []
    for term, pat in patterns.items():
        counts = np.array([len(pat.findall(c)) for c in cleaned])
        row = {"term": term, "kind": "dictionary" if term in ABSOLUTIST_TERMS else "expression"}
        for cls, name in [(0, "control"), (1, "depressed")]:
            m = y == cls
            row[f"per_1k_{name}"] = 1000 * counts[m].sum() / words[m].sum()
            row[f"participants_using_{name}"] = int((counts[m] > 0).sum())
        rows.append(row)
    table = pd.DataFrame(rows)
    dic = table["kind"] == "dictionary"
    for name in ["control", "depressed"]:
        table[f"share_{name}"] = np.where(dic, table[f"per_1k_{name}"] / table.loc[dic, f"per_1k_{name}"].sum(), np.nan)
    return table.sort_values(["kind", "per_1k_depressed"], ascending=[True, False]).reset_index(drop=True)


def main(out_dir: str) -> None:
    texts, y, phq = pool_texts(data.data_root())
    rate = 1000 * np.array([absolutist_rate(t) for t in texts])
    dep, ctrl = rate[y == 1], rate[y == 0]
    mw = mannwhitneyu(dep, ctrl, alternative="greater")
    rho, p_rho = spearmanr(rate, phq)
    result = {
        "cohort": "DAIC-WOZ train + dev (training pool)", "dictionary": "Al-Mosaiwi and Johnstone (2018), 19 words",
        "n": int(len(rate)), "n_depressed": int(len(dep)), "n_control": int(len(ctrl)),
        "unit": "absolutist words per 1,000 participant words",
        "mean_depressed": float(dep.mean()), "mean_control": float(ctrl.mean()),
        "median_depressed": float(np.median(dep)), "median_control": float(np.median(ctrl)),
        "mannwhitney_U": float(mw.statistic), "mannwhitney_p_one_sided": float(mw.pvalue),
        "alternative": "depressed > control",
        "rank_biserial_r": float(2 * mw.statistic / (len(dep) * len(ctrl)) - 1),
        "spearman_rho_phq8": float(rho), "spearman_p": float(p_rho),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "absolutist_test_pool.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    table = term_table(texts, y)
    table.to_csv(out / "absolutist_terms_pool.csv", index=False)
    print(json.dumps(result, indent=2))
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/eda")
