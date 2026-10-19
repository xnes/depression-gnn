"""ROC-AUC with bootstrap CIs and paired bootstrap differences against a reference model.

Reads the per-participant probability files written by notebooks/02_experiments.ipynb (kept outside
the repository) and writes only aggregates:

    PYTHONPATH=. python scripts/paired_auc.py <OUTPUT_DIR> results/experiments/v2
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.common.evaluation import bootstrap_auc, paired_bootstrap_auc

MODELS = ["GAT-GCN-FeExtr (Empath graph)", "EmpathGCN", "InductGCN", "GAT text-only",
          "GCN+GAT hybrid", "GAT-GCN-FeExtr (word graph)"]
COHORTS = ["daic_test", "edaic"]


def main(prob_dir: Path, out_dir: Path, reference: str = MODELS[0]) -> pd.DataFrame:
    rows = []
    for cohort in COHORTS:
        probs = {m: pd.read_csv(prob_dir / f"probs_{m.replace(' ', '_')}_{cohort}.csv") for m in MODELS}
        y = probs[reference]["label"].to_numpy()
        for m in MODELS:
            if not (probs[m]["Participant_ID"].to_numpy() == probs[reference]["Participant_ID"].to_numpy()).all():
                raise ValueError(f"Participant order differs for {m} on {cohort}.")
        for m in MODELS:
            auc, lo, hi = bootstrap_auc(y, probs[m]["prob"].to_numpy())
            row = {"cohort": cohort, "model": m, "AUC": round(auc, 3), "AUC 95% CI": f"[{lo:.3f}, {hi:.3f}]"}
            if m != reference:
                d = paired_bootstrap_auc(y, probs[reference]["prob"].to_numpy(), probs[m]["prob"].to_numpy())
                row.update({"delta vs reference": round(d["delta"], 3),
                            "delta 95% CI": f"[{d['low']:+.3f}, {d['high']:+.3f}]",
                            "P(reference not better)": round(d["p_not_better"], 3)})
            rows.append(row)
    table = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "paired_auc.csv", index=False)
    return table


if __name__ == "__main__":
    print(main(Path(sys.argv[1]), Path(sys.argv[2])).to_string(index=False))
