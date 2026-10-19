"""Assemble the four cohorts (DAIC-WOZ train/dev/test, E-DAIC) as model inputs. No torch here."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import data
from src.common.features import EmpathScorer, build_feature_table, clean_text


@dataclass
class Cohort:
    """Participants of one cohort. Texts are participant turns only, with tags removed."""
    name: str
    ids: np.ndarray
    y: np.ndarray
    texts: list[str]
    empath: np.ndarray          # (n, len(EmpathScorer.categories))
    features: pd.DataFrame      # indexed by Participant_ID; may hold other cohorts' rows too
    phq: np.ndarray | None = None  # PHQ-8 total, for exploratory analysis only (never a model input)

    def subset(self, idx: np.ndarray) -> "Cohort":
        return Cohort(self.name, self.ids[idx], self.y[idx], [self.texts[i] for i in idx],
                      self.empath[idx], self.features, None if self.phq is None else self.phq[idx])


def build_cohorts(root: Path, scorer: EmpathScorer, psycholinguistic: str = "dsm5") -> dict[str, Cohort]:
    """Load labels and sessions, compute features once, and return cohorts by name.

    Sessions without a usable transcript are dropped (this is where dev goes from 35 to 33).
    E-DAIC excludes participants of the DAIC-WOZ training split.
    """
    daic = data.load_daic_labels(root)
    edaic = data.load_edaic_labels(root)
    train_ids = set(daic.loc[daic["split"] == "train", "Participant_ID"])
    edaic = edaic[~edaic["Participant_ID"].isin(train_ids)]

    groups = {f"daic_{s}": (daic[daic["split"] == s], data.load_daic_session) for s in data.DAIC_SPLIT_FILES}
    groups["edaic"] = (edaic, data.load_edaic_session)

    sessions, members = {}, {}
    for name, (labels, loader) in groups.items():
        rows = []
        for pid, y, score in zip(labels["Participant_ID"], labels["PHQ8_Binary"], labels["PHQ8_Score"]):
            s = loader(root, int(pid))
            if s is None:
                continue
            key = (name, int(pid))  # E-DAIC re-releases DAIC-WOZ ids: keep cohorts apart
            sessions[key] = s
            rows.append((int(pid), int(y), float(score)))
        members[name] = rows

    table = build_feature_table(sessions, scorer, psycholinguistic)
    cohorts = {}
    for name, rows in members.items():
        ids = np.array([p for p, _, _ in rows])
        texts = [clean_text(" ".join(sessions[(name, p)].loc[lambda d: d["speaker"] == "Participant", "text"]))
                 for p, _, _ in rows]
        feats = table.loc[[(name, p) for p, _, _ in rows]]
        feats.index = ids
        cohorts[name] = Cohort(name, ids, np.array([y for _, y, _ in rows]), texts,
                               scorer.matrix(texts, scorer.categories), feats,
                               np.array([s for _, _, s in rows]))
    return cohorts
