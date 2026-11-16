"""Auditability analyses for the main model (GAT-GCN-FeExtr over the Empath category graph).

1. Convergence: do the Empath categories that receive more attention in depressed participants
   coincide with the categories that separate the classes by ANOVA on the training split?
2. Case report: for one correctly detected test participant, the ensemble probability and the
   categories whose attention most exceeds the control mean of the test cohort (raw attention is
   dominated by categories every participant attends to, so it does not show what is unusual in a
   case). The report carries no participant id and no transcript text.

Inputs: aggregate files in results/ and the per-participant attention and probability files that
notebooks/02_experiments.ipynb writes outside the repository. Outputs: aggregates only.

    PYTHONPATH=. python scripts/auditability.py <OUTPUT_DIR>/v2 results/experiments/v2
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr

from src.common.features import load_mapping

MAIN = "GAT-GCN-FeExtr_(Empath_graph)"
TOP_K = 20


def mapping_membership() -> dict[str, str]:
    """Empath category -> criterion ids that use it (e.g. 'C1+' or 'C2-')."""
    out: dict[str, list[str]] = {}
    for crit in load_mapping()["criteria"]:
        for c in crit["present"]:
            out.setdefault(c, []).append(f"{crit['id']}+")
        for c in crit["absent"]:
            out.setdefault(c, []).append(f"{crit['id']}-")
    return {k: ", ".join(v) for k, v in out.items()}


def convergence(by_class: pd.DataFrame, anova: pd.DataFrame, k: int = TOP_K) -> tuple[pd.DataFrame, dict]:
    """Overlap of the top-k attention-delta categories with the top-k ANOVA categories."""
    common = by_class.index.intersection(anova.index)
    att = by_class.loc[common, "delta"].sort_values(ascending=False)
    f = anova.loc[common, "F"].sort_values(ascending=False)
    top_att, top_f = list(att.index[:k]), list(f.index[:k])
    shared = [c for c in top_att if c in top_f]
    rho, p_rho = spearmanr(att.loc[common], f.loc[common], nan_policy="omit")
    p_overlap = hypergeom.sf(len(shared) - 1, len(common), k, k)  # P(overlap >= observed) by chance
    member = mapping_membership()
    table = pd.DataFrame({
        "rank_attention": range(1, k + 1), "category": top_att,
        "attention_delta": att.iloc[:k].round(5).to_numpy(),
        "anova_rank_train": [int(f.index.get_loc(c)) + 1 for c in top_att],
        "in_anova_top_k": [c in top_f for c in top_att],
        "dsm5_mapping": [member.get(c, "") for c in top_att],
    })
    summary = {"n_common_categories": int(len(common)), "k": k, "shared": shared,
               "n_shared": len(shared), "expected_by_chance": round(k * k / len(common), 2),
               "p_overlap_hypergeometric": float(p_overlap),
               "spearman_delta_vs_F": float(rho), "spearman_p": float(p_rho)}
    return table, summary


def case_report(attention: pd.DataFrame, probs: pd.DataFrame, tau: float, k: int = 10) -> dict:
    """Report for the true positive with the median probability among true positives."""
    probs = probs.set_index("Participant_ID")
    tp = probs[(probs["label"] == 1) & (probs["prob"] >= tau)].sort_values("prob")
    pid = tp.index[len(tp) // 2]
    labels = probs.loc[attention.index, "label"]
    mean_ctrl, mean_dep = attention[labels == 0].mean(), attention[labels == 1].mean()
    row = attention.loc[pid]
    member = mapping_membership()
    excess = (row - mean_ctrl).sort_values(ascending=False).head(k)
    return {
        "participant": "anonymized test participant (true positive, median probability among true positives)",
        "ensemble_probability": round(float(probs.loc[pid, "prob"]), 3),
        "threshold": tau, "decision": "screen positive", "n_true_positives": int(len(tp)),
        "ranking": "attention minus control mean (test cohort)",
        "_pid": pid,  # used internally to join criterion scores; removed before writing
        "top_categories": [
            {"category": c, "attention": round(float(row[c]), 5), "excess_over_control": round(float(e), 5),
             "control_mean": round(float(mean_ctrl[c]), 5), "depressed_mean": round(float(mean_dep[c]), 5),
             "above_depressed_mean": bool(row[c] > mean_dep[c]), "dsm5_mapping": member.get(c, "")}
            for c, e in excess.items()
        ],
    }


def criterion_profile(scores: pd.Series, reference: pd.DataFrame) -> list[dict]:
    """DSM-5 criterion scores of one participant against the training-pool class profile.

    z is measured against the control distribution; higher means further toward the depressive
    pattern, because every criterion score is oriented that way (Eq. criterion).
    """
    labels = {c["id"]: c["label"] for c in load_mapping()["criteria"]}
    rows = []
    for col in reference.index:
        ref = reference.loc[col]
        z = (scores[col] - ref["control_mean"]) / ref["control_sd"] if ref["control_sd"] > 0 else np.nan
        closer_to = "depressed" if abs(scores[col] - ref["depressed_mean"]) < abs(scores[col] - ref["control_mean"]) else "control"
        rows.append({"feature": col, "label": labels.get(col.replace("dsm_", ""), col),
                     "score": round(float(scores[col]), 5), "z_vs_control": round(float(z), 2),
                     "closer_to": closer_to})
    return rows


def main(prob_dir: Path, results: Path) -> None:
    by_class = pd.read_csv(results / "attention_by_class_test.csv", index_col=0)
    anova = pd.read_csv(Path("results/eda/empath_anova_train.csv")).set_index("category")
    table, summary = convergence(by_class, anova)
    table.to_csv(results / "attention_anova_convergence.csv", index=False)
    (results / "attention_anova_convergence.json").write_text(json.dumps(summary, indent=2))

    metrics = pd.read_csv(results / "metrics.csv")
    tau = float(metrics[(metrics["model"] == MAIN.replace("_", " ")) & (metrics["cohort"] == "daic_test")]["tau"].iloc[0])
    attention = pd.read_csv(prob_dir / f"attention_test_{MAIN}.csv", index_col=0)
    probs = pd.read_csv(prob_dir / f"probs_{MAIN}_daic_test.csv")
    report = case_report(attention, probs, tau)
    scores_path = prob_dir / "criterion_scores_daic_test.csv"
    reference_path = Path("results/eda/dsm5_reference_pool.csv")
    if scores_path.exists() and reference_path.exists():
        scores = pd.read_csv(scores_path, index_col=0)
        pid = report.pop("_pid")
        report["dsm5_criteria"] = criterion_profile(scores.loc[pid], pd.read_csv(reference_path, index_col=0))
    else:
        report.pop("_pid")
        report["dsm5_criteria"] = "pending: run Part C of notebooks/01_dataset_eda.ipynb"
    (results / "case_report_example.json").write_text(json.dumps(report, indent=2))

    print(table.to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
