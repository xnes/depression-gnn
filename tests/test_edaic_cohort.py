"""The external E-DAIC cohorts must share no participant with DAIC-WOZ (pool or test)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common import data
from src.paper1_graph_model import corpus

ITEMS = {item: 1 for item in data.PHQ8_ITEMS}
DAIC = {"train": [300, 301], "dev": [302, 303], "test": [304, 305]}
# E-DAIC re-releases every DAIC-WOZ id above, plus new ids in its train, dev and test splits.
EDAIC = {"train": [300, 301, 302, 600], "dev": [303, 304, 601], "test": [650, 651]}


class StubScorer:
    """Stands in for EmpathScorer so the test needs no lexicon."""
    categories = ["sleep"]

    def scores(self, text: str) -> dict:
        return {"sleep": 0.0}

    def matrix(self, texts, categories) -> np.ndarray:
        return np.zeros((len(texts), len(categories)), np.float32)


def _session(path: Path, sep: str, columns: list[str]) -> None:
    rows = [(0.0, 1.0, "Ellie", "how are you"), (1.5, 3.0, "Participant", "i feel tired")]
    if sep == "\t":
        pd.DataFrame(rows, columns=columns).to_csv(path, sep=sep, index=False)
    else:
        pd.DataFrame({"Start_Time": [0.0], "End_Time": [2.0], "Text": ["i sleep badly"]}).to_csv(path, index=False)


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    oracle, transcripts = tmp_path / "DAIC-WOZ" / "oracle", tmp_path / "DAIC-WOZ" / "transcripts"
    labels_e, data_e = tmp_path / "E-DAIC" / "labels", tmp_path / "E-DAIC" / "data"
    for d in (oracle, transcripts, labels_e, data_e):
        d.mkdir(parents=True)
    for split, ids in DAIC.items():
        pd.DataFrame([{"Participant_ID": p, "PHQ8_Binary": p % 2, "PHQ8_Score": 12 * (p % 2), **ITEMS}
                      for p in ids]).to_csv(oracle / data.DAIC_SPLIT_FILES[split], index=False)
        for p in ids:
            _session(transcripts / f"{p}_TRANSCRIPT.csv", "\t", ["start_time", "stop_time", "speaker", "value"])
    for split, ids in EDAIC.items():
        # PHQ_Binary disagrees with the cutoff for 601 (score 12, binary 0), as in the released files.
        pd.DataFrame([{"Participant_ID": p, "Gender": "male", "PHQ_Binary": int(p % 2 and p != 601),
                       "PHQ_Score": 12 * (p % 2)} for p in ids]).to_csv(labels_e / data.EDAIC_SPLIT_FILES[split],
                                                                       index=False)
        for p in ids:
            _session(data_e / f"{p}_Transcript.csv", ",", [])
    return tmp_path


def test_edaic_labels_cover_all_splits_and_use_the_cutoff(root: Path) -> None:
    labels = data.load_edaic_labels(root).set_index("Participant_ID")
    assert set(labels.index) == {p for ids in EDAIC.values() for p in ids}
    assert labels.loc[650, "split"] == "test"
    assert labels.loc[601, "PHQ8_Binary"] == 1  # recomputed from the total, not the released flag


def test_external_cohorts_share_nobody_with_daic(root: Path) -> None:
    cohorts = corpus.build_cohorts(root, StubScorer())
    daic_ids = set().union(*(set(cohorts[f"daic_{s}"].ids) for s in DAIC))
    pool = set(cohorts["daic_train"].ids) | set(cohorts["daic_dev"].ids)
    for name in corpus.EXTERNAL_COHORTS:
        ids = set(cohorts[name].ids)
        assert not ids & pool, f"{name} overlaps the training pool"
        assert not ids & set(cohorts["daic_test"].ids), f"{name} overlaps the DAIC-WOZ test"
        assert not ids & daic_ids
    assert set(cohorts["edaic"].ids) == {600, 601, 650, 651}
    assert set(cohorts["edaic_test"].ids) == {650, 651}


def test_controlled_cohort_is_daic_test_on_asr(root: Path) -> None:
    cohorts = corpus.build_cohorts(root, StubScorer())
    controlled = cohorts[corpus.CONTROLLED_COHORT]
    assert set(controlled.ids) == {304}  # the only DAIC-WOZ test id that E-DAIC re-releases here
    assert corpus.CONTROLLED_COHORT not in corpus.EXTERNAL_COHORTS
    assert controlled.texts != [] and "tired" not in controlled.texts[0]  # ASR text, not the DAIC one


def test_overlap_is_rejected() -> None:
    def fake(name, ids):
        return corpus.Cohort(name, np.array(ids), np.zeros(len(ids)), [""] * len(ids),
                             np.zeros((len(ids), 1)), pd.DataFrame())
    cohorts = {"daic_train": fake("daic_train", [1]), "daic_dev": fake("daic_dev", [2]),
               "daic_test": fake("daic_test", [3]), "edaic": fake("edaic", [2, 9])}
    with pytest.raises(ValueError):
        corpus.check_external_disjoint(cohorts)
