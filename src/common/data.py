"""Load DAIC-WOZ and E-DAIC labels and participant turns.

Ported from legacy/data_loader.py. Raw data lives outside the repository; its root is read
from the environment variable DAIC_DATA_DIR. Expected layout under that root (the layout used
in the original Colab notebooks):

    DAIC-WOZ/oracle/train_split_Depression_AVEC2017.csv
    DAIC-WOZ/oracle/dev_split_Depression_AVEC2017.csv
    DAIC-WOZ/oracle/full_test_split.csv
    DAIC-WOZ/transcripts/<id>_TRANSCRIPT.csv      tab-separated: start_time, stop_time, speaker, value
    E-DAIC/labels/{train,dev,test}_split.csv      official AVEC 2019 splits (275 labeled participants)
    E-DAIC/labels/detailed_lables.csv             PHQ-8 item scores for all 275 (file name as released)
    E-DAIC/data/<id>_Transcript.csv               comma-separated, ASR output with a Text column

Functions here return labels and participant turns only. Interviewer turns are dropped on load.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

PHQ8_ITEMS = [
    "PHQ8_NoInterest", "PHQ8_Depressed", "PHQ8_Sleep", "PHQ8_Tired",
    "PHQ8_Appetite", "PHQ8_Failure", "PHQ8_Concentrating", "PHQ8_Moving",
]
PHQ8_CUTOFF = 10

DAIC_SPLIT_FILES = {
    "train": "train_split_Depression_AVEC2017.csv",
    "dev": "dev_split_Depression_AVEC2017.csv",
    "test": "full_test_split.csv",
}
# DAIC-WOZ documentation: these sessions lack the interviewer (Ellie) turns.
INCOMPLETE_SESSIONS = frozenset({451, 458, 480})
SYNC_MARKERS = frozenset({"<synch>", "<sync>", "[sync]"})

EDAIC_SPLIT_FILES = {"train": "train_split.csv", "dev": "dev_split.csv", "test": "test_split.csv"}
EDAIC_ITEMS_FILE = "detailed_lables.csv"
_EDAIC_ITEM_COLUMNS = {
    "PHQ8_1_NoInterest": "PHQ8_NoInterest", "PHQ8_2_Depressed": "PHQ8_Depressed",
    "PHQ8_3_Sleep": "PHQ8_Sleep", "PHQ8_4_Tired": "PHQ8_Tired", "PHQ8_5_Appetite": "PHQ8_Appetite",
    "PHQ8_6_Failure": "PHQ8_Failure", "PHQ8_7_Concentration": "PHQ8_Concentrating",
    "PHQ8_8_Psychomotor": "PHQ8_Moving",
}

TURN_COLUMNS = ["start", "end", "text"]


def data_root(path: str | Path | None = None) -> Path:
    """Return the raw-data root, from the argument or DAIC_DATA_DIR."""
    value = path or os.environ.get("DAIC_DATA_DIR")
    if not value:
        raise RuntimeError("Set DAIC_DATA_DIR to the folder that holds DAIC-WOZ/ and E-DAIC/.")
    root = Path(value)
    if not root.is_dir():
        raise FileNotFoundError(f"DAIC_DATA_DIR does not exist: {root}")
    return root


def normalize_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonize PHQ-8 column names across the AVEC split files and E-DAIC labels."""
    renames = {"PHQ_Binary": "PHQ8_Binary", "PHQ_Score": "PHQ8_Score",
               "PHQ_8Total": "PHQ8_Score", "PHQ8_Total": "PHQ8_Score"}
    df = df.rename(columns={c: renames.get(c, re.sub(r"^PHQ_8(?=[A-Z])", "PHQ8_", c))
                            for c in df.columns})
    df["Participant_ID"] = df["Participant_ID"].astype(int)
    if "PHQ8_Binary" not in df.columns:
        df["PHQ8_Binary"] = (df["PHQ8_Score"] >= PHQ8_CUTOFF).astype(int)
    return df


def load_daic_labels(root: Path) -> pd.DataFrame:
    """Labels for the three official AVEC 2017 splits, with a 'split' column."""
    oracle = root / "DAIC-WOZ" / "oracle"
    frames = []
    for split, name in DAIC_SPLIT_FILES.items():
        df = normalize_label_columns(pd.read_csv(oracle / name))
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_daic_turns(root: Path, participant_id: int) -> pd.DataFrame | None:
    """Participant turns of one DAIC-WOZ session, or None when the session is unusable."""
    if participant_id in INCOMPLETE_SESSIONS:
        return None
    path = root / "DAIC-WOZ" / "transcripts" / f"{participant_id}_TRANSCRIPT.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t")
    df = df.dropna(subset=["value"])
    df = df[~df["value"].isin(SYNC_MARKERS)]
    df = df[df["speaker"] == "Participant"]
    return df.rename(columns={"start_time": "start", "stop_time": "end", "value": "text"})[
        TURN_COLUMNS].reset_index(drop=True)


def load_daic_session(root: Path, participant_id: int) -> pd.DataFrame | None:
    """Full DAIC-WOZ session (both speakers) with columns start, end, speaker, text.

    Only for timing features: interviewer rows give the end time a response follows. Callers
    must not use interviewer text.
    """
    if participant_id in INCOMPLETE_SESSIONS:
        return None
    path = root / "DAIC-WOZ" / "transcripts" / f"{participant_id}_TRANSCRIPT.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t").dropna(subset=["value"])
    df = df[~df["value"].isin(SYNC_MARKERS)]
    df = df.rename(columns={"start_time": "start", "stop_time": "end", "value": "text"})
    df["speaker"] = df["speaker"].where(df["speaker"] == "Participant", "Interviewer")
    df = df[["start", "end", "speaker", "text"]].reset_index(drop=True)
    return df if (df["speaker"] == "Participant").any() else None


def load_edaic_session(root: Path, participant_id: int) -> pd.DataFrame | None:
    """E-DAIC ASR segments in the same shape as load_daic_session (participant speech only)."""
    turns = load_edaic_turns(root, participant_id)
    if turns is None or turns.empty:
        return None
    return turns.assign(speaker="Participant")[["start", "end", "speaker", "text"]]


def load_edaic_labels(root: Path) -> pd.DataFrame:
    """All labeled E-DAIC participants (official train, dev and test splits), with a 'split' column.

    The binary label is recomputed from the PHQ-8 total (>= 10): the released PHQ_Binary column marks
    some participants with totals of 10-16 as negative, which contradicts the cutoff used throughout.
    Item scores come from the detailed label file when it is present.
    """
    labels = root / "E-DAIC" / "labels"
    frames = []
    for split, name in EDAIC_SPLIT_FILES.items():
        # Gender is coded as text here and 0/1 in DAIC-WOZ with no documented correspondence; dropped.
        df = pd.read_csv(labels / name).drop(columns=["PHQ_Binary", "PHQ8_Binary"], errors="ignore")
        df = normalize_label_columns(df)[["Participant_ID", "PHQ8_Score", "PHQ8_Binary"]]
        frames.append(df.assign(split=split))
    out = pd.concat(frames, ignore_index=True)
    if (labels / EDAIC_ITEMS_FILE).exists():
        items = pd.read_csv(labels / EDAIC_ITEMS_FILE).rename(
            columns={"Participant": "Participant_ID", **_EDAIC_ITEM_COLUMNS})
        out = out.merge(items[["Participant_ID", *PHQ8_ITEMS]], on="Participant_ID", how="left")
    return out


def external_edaic_labels(daic: pd.DataFrame, edaic: pd.DataFrame) -> pd.DataFrame:
    """E-DAIC participants whose ID appears in no DAIC-WOZ split (train, dev or test).

    E-DAIC re-releases every DAIC-WOZ session with an ASR transcript. Excluding only the DAIC-WOZ
    training split would leave development participants (part of the training pool) and test
    participants (already scored on DAIC-WOZ) in the external cohort.
    """
    return edaic[~edaic["Participant_ID"].isin(set(daic["Participant_ID"]))].reset_index(drop=True)


def asr_daic_test_labels(daic: pd.DataFrame, edaic: pd.DataFrame) -> pd.DataFrame:
    """DAIC-WOZ test participants that E-DAIC re-releases, with DAIC-WOZ labels.

    For the controlled transcription analysis only: same participants and interviews as the DAIC-WOZ
    test partition, scored on the E-DAIC ASR transcript. Never part of the external evaluation.
    """
    test = daic[daic["split"] == "test"]
    return test[test["Participant_ID"].isin(set(edaic["Participant_ID"]))].reset_index(drop=True)


def load_edaic_turns(root: Path, participant_id: int) -> pd.DataFrame | None:
    """ASR segments of one E-DAIC session (participant speech only), or None if missing."""
    path = root / "E-DAIC" / "data" / f"{participant_id}_Transcript.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path).dropna(subset=["Text"])
    out = pd.DataFrame({
        "start": df.get("Start_Time", pd.Series(float("nan"), index=df.index)),
        "end": df.get("End_Time", pd.Series(float("nan"), index=df.index)),
        "text": df["Text"].astype(str),
    })
    if "Confidence" in df.columns:
        out["asr_confidence"] = df["Confidence"]
    return out.reset_index(drop=True)
