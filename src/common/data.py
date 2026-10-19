"""Load DAIC-WOZ and E-DAIC labels and participant turns.

Ported from legacy/data_loader.py. Raw data lives outside the repository; its root is read
from the environment variable DAIC_DATA_DIR. Expected layout under that root (the layout used
in the original Colab notebooks):

    DAIC-WOZ/oracle/train_split_Depression_AVEC2017.csv
    DAIC-WOZ/oracle/dev_split_Depression_AVEC2017.csv
    DAIC-WOZ/oracle/full_test_split.csv
    DAIC-WOZ/transcripts/<id>_TRANSCRIPT.csv      tab-separated: start_time, stop_time, speaker, value
    E-DAIC/labels/Detailed_PHQ8_Labels.csv
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
    return normalize_label_columns(pd.read_csv(root / "E-DAIC" / "labels" / "Detailed_PHQ8_Labels.csv"))


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
