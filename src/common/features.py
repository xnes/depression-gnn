"""Per-participant feature views (dissertation Sections 4.5.2-4.5.4).

View 1 (lexical, TF-IDF) lives in the graph builders. This module computes the "professional"
features attached to document nodes:

    temporal        latency, pauses, silence/speech ratio, total speech time
    paralinguistic  per-turn rates of transcript tags, in four groups
    psycholinguistic  DSM-5 criterion scores from the PHQ-8/DSM-5 to Empath mapping, plus the
                      absolutist-word rate

Fixes relative to legacy/feature_extraction.py:
    * Tags are matched literally. The legacy `count_patterns` wrapped unescaped terms in \\b...\\b,
      so tag counts were always 0 and `laughter_count` counted single-letter words instead.
    * No minimum turn duration filter (decision of 2026-10-17).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

MAPPING_PATH = Path(__file__).resolve().parents[2] / "data" / "mapping" / "phq8_dsm5_empath.json"

# Tag groups from legacy/feature_extraction.py, kept as written (including the corpus typo).
TAG_GROUPS = {
    "laughter": ["[laughter]", "<laughter>", "<laguhter>"],
    "stress": ["<sigh>", "<crying>", "<sobbing>", "<grunt>", "<tisk>", "<psh>", "<pft>",
               "<vocal fry>", "<gasp>", "<groan>", "<hiss sound>", "<lip smacking>",
               "<crying and laughing>", "<sobbing and laughing>"],
    "breath": ["<clears throat>", "<cough>", "<sniff>", "<sniffle>", "<sneeze>", "<yawn>",
               "<sharp exhale>", "<yawning>"],
    "hesitation": ["<deep breath>", "<sharp inhale>", "<exhaling>", "<sharp inhaling>",
                   "<mutters>", "<mumbling>"],
}

# Al-Mosaiwi and Johnstone (2018), as listed in legacy/feature_extraction.py.
ABSOLUTIST_TERMS = [
    "always", "never", "everything", "nothing", "nobody", "everyone", "everybody", "no one",
    "completely", "totally", "absolutely", "utterly", "entire", "entirely", "forever",
    "impossible", "perfect", "awful", "terrible", "disastrous", "worst", "best", "ruined",
    "hopeless", "worthless", "helpless", "pointless", "definite", "undeniable", "unquestionable",
    "clearly", "obviously", "definitely", "undoubtedly", "none", "nowhere", "nor", "zero", "can't",
]

# Legacy EmpathGCN domain groups (legacy/Empath_GCN_Hybrid.ipynb, cell 12). Kept for ablation.
EMPATH_DOMAIN_GROUPS = {
    "negative_affect": ["sadness", "negative_emotion", "suffering", "anger", "disgust", "hate", "envy"],
    "positive_affect": ["joy", "positive_emotion", "love", "cheerfulness", "optimism", "contentment"],
    "social_connection": ["friends", "family", "affection", "trust", "communication", "help", "sympathy"],
    "cognitive_load": ["confusion", "doubt", "memory", "thought", "perception", "consciousness", "attention"],
    "somatic_health": ["sleep", "health", "medical_emergency", "body", "pain", "eating", "exercise"],
    "motivation_agency": ["giving", "achievement", "leadership", "power", "competing", "work"],
    "time_perspective": ["past", "future", "time"],
    "social_isolation": ["alone", "shame", "awkwardness", "rejection", "neglect"],
}

PAUSE_THRESHOLD_S = 0.5
_TAG_PATTERNS = {g: re.compile("|".join(re.escape(t) for t in terms)) for g, terms in TAG_GROUPS.items()}
_ANY_TAG = re.compile(r"<[^>]*>|\[[^\]]*\]")
_ABSOLUTIST = re.compile(r"(?<![a-z'])(?:" + "|".join(re.escape(t) for t in ABSOLUTIST_TERMS) + r")(?![a-z'])")
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def load_mapping(path: Path = MAPPING_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean_text(text: str) -> str:
    """Lowercase text with transcript tags removed."""
    return _ANY_TAG.sub(" ", str(text).lower())


# ----------------------------------------------------------------------------- temporal

def temporal_features(session: pd.DataFrame) -> dict:
    """Timing markers from one full session (both speakers, in order).

    Expects columns start, end, speaker ('Participant' or interviewer). Interviewer turns are used
    for their timestamps only (latency), never for their text.

    latency            mean gap between the end of an interviewer turn and the start of the next
                       participant turn (negative gaps clipped to 0)
    pause_count        gaps longer than 0.5 s between consecutive participant turns
    silence_speech     total such gap time divided by total participant speech time
    speech_min         total participant speech time in minutes
    """
    s = session.reset_index(drop=True)
    is_p = s["speaker"].eq("Participant").to_numpy()
    start, end = s["start"].to_numpy(float), s["end"].to_numpy(float)
    latencies, gaps = [], []
    for i in range(1, len(s)):
        if not is_p[i]:
            continue
        gap = start[i] - end[i - 1]
        if is_p[i - 1]:
            if gap > PAUSE_THRESHOLD_S:
                gaps.append(gap)
        else:
            latencies.append(max(0.0, gap))
    speech = np.clip(end[is_p] - start[is_p], 0, None).sum()
    return {
        "latency": float(np.mean(latencies)) if latencies else np.nan,
        "pause_count": float(len(gaps)),
        "silence_speech": float(np.sum(gaps) / speech) if speech > 0 else np.nan,
        "speech_min": float(speech / 60) if speech > 0 else np.nan,
    }


# ----------------------------------------------------------------------------- paralinguistic

def paralinguistic_features(turns: list[str]) -> dict:
    """Occurrences of each tag group per participant turn."""
    n = max(len(turns), 1)
    text = " ".join(str(t).lower() for t in turns)
    return {f"tag_{g}": len(p.findall(text)) / n for g, p in _TAG_PATTERNS.items()}


# ----------------------------------------------------------------------------- psycholinguistic

class EmpathScorer:
    """Thin wrapper so tests can inject a fake lexicon."""

    def __init__(self, lexicon=None):
        if lexicon is None:
            from empath import Empath
            lexicon = Empath()
        self.lexicon = lexicon
        self.categories = sorted(self.lexicon.cats)

    def scores(self, text: str) -> dict[str, float]:
        result = self.lexicon.analyze(clean_text(text), normalize=True)
        return result or {c: 0.0 for c in self.categories}

    def matrix(self, texts: list[str], categories: list[str]) -> np.ndarray:
        return np.array([[self.scores(t).get(c, 0.0) for c in categories] for t in texts],
                        dtype=np.float32)


def criterion_scores(empath: dict[str, float], mapping: dict) -> dict:
    """One signed score per DSM-5 criterion: mean(present) - mean(absent).

    Categories missing from the Empath lexicon are skipped (they would always score 0).
    """
    missing = set(mapping["missing_from_empath"])
    out = {}
    for crit in mapping["criteria"]:
        present = [empath.get(c, 0.0) for c in crit["present"] if c not in missing]
        absent = [empath.get(c, 0.0) for c in crit["absent"] if c not in missing]
        value = (np.mean(present) if present else 0.0) - (np.mean(absent) if absent else 0.0)
        out[f"dsm_{crit['id']}"] = float(value)
    return out


def domain_scores(empath: dict[str, float], groups: dict = EMPATH_DOMAIN_GROUPS) -> dict:
    return {f"dom_{g}": float(np.mean([empath.get(c, 0.0) for c in cats])) for g, cats in groups.items()}


def absolutist_rate(text: str) -> float:
    cleaned = clean_text(text)
    n_words = len(_WORD.findall(cleaned))
    return len(_ABSOLUTIST.findall(cleaned)) / n_words if n_words else 0.0


# ----------------------------------------------------------------------------- assembly

def build_feature_table(sessions: dict[int, pd.DataFrame], scorer: EmpathScorer,
                        psycholinguistic: str = "dsm5", mapping: dict | None = None) -> pd.DataFrame:
    """Feature table indexed by Participant_ID.

    sessions: key (participant id, or any hashable such as (cohort, id)) -> full session with
              columns start, end, speaker, text. For corpora without interviewer turns or
              timestamps, temporal features come out as NaN.
    psycholinguistic: "dsm5" (criterion scores from the mapping, default) or "domains" (legacy
              EMPATH_DOMAIN_GROUPS).
    """
    mapping = mapping or load_mapping()
    rows = {}
    for pid, s in sessions.items():
        p_turns = s.loc[s["speaker"] == "Participant", "text"].astype(str).tolist()
        text = " ".join(p_turns)
        emp = scorer.scores(text)
        psy = criterion_scores(emp, mapping) if psycholinguistic == "dsm5" else domain_scores(emp)
        rows[pid] = {**temporal_features(s), **paralinguistic_features(p_turns), **psy,
                     "absolutist_rate": absolutist_rate(text)}
    table = pd.DataFrame.from_dict(rows, orient="index")
    if not isinstance(table.index, pd.MultiIndex):
        table = table.rename_axis("Participant_ID")
    return table


FEATURE_GROUPS = {
    "temporal": ["latency", "pause_count", "silence_speech", "speech_min"],
    "paralinguistic": [f"tag_{g}" for g in TAG_GROUPS],
}


def view_columns(table: pd.DataFrame, views: tuple[str, ...]) -> list[str]:
    """Columns of the requested views: temporal, paralinguistic, psycholinguistic."""
    cols = []
    for v in views:
        if v == "psycholinguistic":
            cols += [c for c in table.columns if c.startswith(("dsm_", "dom_"))] + ["absolutist_rate"]
        else:
            cols += FEATURE_GROUPS[v]
    return cols
