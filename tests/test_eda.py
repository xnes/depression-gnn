"""EDA tests on a tiny synthetic corpus that mimics the DAIC-WOZ / E-DAIC folder layout."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.common import data, eda

ITEMS = {item: 1 for item in data.PHQ8_ITEMS}
SECRET_WORD = "zanzibarword"  # appears only in transcripts; must never reach an output file


def _daic_transcript(path: Path, participant_words: str) -> None:
    rows = [
        (0.0, 1.0, "Ellie", "how are you doing today"),
        (1.5, 4.0, "Participant", f"i am fine {SECRET_WORD} <laughter>"),
        (4.0, 4.1, "Participant", "<sync>"),
        (5.0, 6.0, "Ellie", "tell me more"),
        (6.5, 9.5, "Participant", participant_words),
    ]
    pd.DataFrame(rows, columns=["start_time", "stop_time", "speaker", "value"]).to_csv(
        path, sep="\t", index=False)


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    oracle = tmp_path / "DAIC-WOZ" / "oracle"
    transcripts = tmp_path / "DAIC-WOZ" / "transcripts"
    labels_e = tmp_path / "E-DAIC" / "labels"
    data_e = tmp_path / "E-DAIC" / "data"
    for d in (oracle, transcripts, labels_e, data_e):
        d.mkdir(parents=True)

    pd.DataFrame([
        {"Participant_ID": 300, "PHQ8_Binary": 0, "PHQ8_Score": 3, "Gender": 1, **ITEMS},
        {"Participant_ID": 301, "PHQ8_Binary": 1, "PHQ8_Score": 15, "Gender": 0, **ITEMS},
    ]).to_csv(oracle / data.DAIC_SPLIT_FILES["train"], index=False)
    pd.DataFrame([
        {"Participant_ID": 302, "PHQ8_Binary": 0, "PHQ8_Score": 2, "Gender": 0, **ITEMS},
        {"Participant_ID": 451, "PHQ8_Binary": 1, "PHQ8_Score": 12, "Gender": 1, **ITEMS},
    ]).to_csv(oracle / data.DAIC_SPLIT_FILES["dev"], index=False)
    pd.DataFrame([{"Participant_ID": 303, "PHQ_Binary": 1, "PHQ_Score": 11, "Gender": 1}]).to_csv(
        oracle / data.DAIC_SPLIT_FILES["test"], index=False)

    for pid in (300, 301, 302, 303, 451):
        _daic_transcript(transcripts / f"{pid}_TRANSCRIPT.csv", "i sleep badly most nights")

    e_items = {f"PHQ_8{i.removeprefix('PHQ8_')}": 2 for i in data.PHQ8_ITEMS}
    pd.DataFrame([
        {"Participant_ID": 300, "PHQ_8Total": 3, **e_items},   # in DAIC-WOZ train: excluded
        {"Participant_ID": 600, "PHQ_8Total": 16, **e_items},
        {"Participant_ID": 601, "PHQ_8Total": 4, **e_items},
    ]).to_csv(labels_e / "Detailed_PHQ8_Labels.csv", index=False)
    for pid in (300, 600):
        pd.DataFrame({"Start_Time": [0.0, 3.0], "End_Time": [2.0, 5.0],
                      "Text": ["i feel tired every day", f"unknownword {SECRET_WORD}"],
                      "Confidence": [0.9, 0.7]}).to_csv(data_e / f"{pid}_Transcript.csv", index=False)
    return tmp_path


def test_daic_turns_keep_participant_only(root: Path) -> None:
    turns = data.load_daic_turns(root, 300)
    assert list(turns.columns) == data.TURN_COLUMNS
    assert len(turns) == 2  # two participant turns; sync marker and interviewer turns dropped
    assert not turns["text"].str.contains("how are you").any()


def test_incomplete_session_is_excluded(root: Path) -> None:
    assert data.load_daic_turns(root, 451) is None


def test_label_columns_are_normalized(root: Path) -> None:
    labels = data.load_daic_labels(root)
    assert {"PHQ8_Binary", "PHQ8_Score", "split"} <= set(labels.columns)
    assert labels.loc[labels["Participant_ID"] == 303, "PHQ8_Score"].item() == 11
    edaic = data.load_edaic_labels(root)
    assert set(data.PHQ8_ITEMS) <= set(edaic.columns)
    assert edaic.set_index("Participant_ID")["PHQ8_Binary"].to_dict() == {300: 0, 600: 1, 601: 0}


def test_session_table(root: Path) -> None:
    table = eda.build_session_table(root).set_index("Participant_ID")
    assert 300 in table.index and table.loc[300, "cohort"] == "DAIC-WOZ train"
    assert set(table[table["cohort"] == "E-DAIC"].index) == {600, 601}
    assert not table.loc[451, "transcript_available"]
    assert not table.loc[601, "transcript_available"]  # labeled but no transcript file
    assert table.loc[300, "n_turns"] == 2
    assert table.loc[300, "n_words"] == 9  # tags excluded from the word count
    assert table.loc[300, "n_tags"] == 1
    assert table.loc[300, "speech_min"] == pytest.approx(5.5 / 60)
    assert pd.isna(table.loc[300, "oov_rate"])
    assert table.loc[302, "oov_rate"] == 0
    assert table.loc[600, "oov_rate"] > 0
    assert table.loc[600, "asr_confidence"] == pytest.approx(0.8)


def test_run_eda_writes_only_aggregates(root: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    summary = eda.run_eda(root, out)
    assert summary.loc["sessions with transcript", "DAIC-WOZ dev"] == 1
    assert summary.loc["labeled sessions", "E-DAIC"] == 2
    for name in eda.FIGURES:
        assert (out / f"{name}.pdf").exists() and (out / f"{name}.png").exists()
    for path in out.glob("*"):
        if path.suffix in {".csv", ".tex"}:
            assert SECRET_WORD not in path.read_text(encoding="utf-8")
