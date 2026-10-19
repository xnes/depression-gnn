"""Features, graph construction and threshold rule on synthetic inputs (no torch needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common import features as fx
from src.common.evaluation import evaluate, threshold_min_fn, wilson
from src.paper1_graph_model.graph import EmpathGraphBuilder, WordGraphBuilder, assemble, pagerank, ppmi


class FakeLexicon:
    cats = {"sadness": [], "joy": [], "sleep": [], "work": [], "shame": []}

    def analyze(self, text, normalize=True):
        words = text.split()
        n = max(len(words), 1)
        return {c: sum(w == c for w in words) / n for c in self.cats}


# ----------------------------------------------------------------------------- features

def test_tags_are_counted_literally_not_as_regex() -> None:
    # Regression for legacy count_patterns: "[laughter]" was a character class and "\b" never
    # matched around "<...>", so real tags counted 0 and single letters counted as laughter.
    out = fx.paralinguistic_features(["i was fine <laughter> really", "a dog a cat <sigh>"])
    assert out["tag_laughter"] == pytest.approx(0.5)
    assert out["tag_stress"] == pytest.approx(0.5)
    assert fx.paralinguistic_features(["a e i o u"])["tag_laughter"] == 0


def test_temporal_features() -> None:
    s = pd.DataFrame({
        "start": [0.0, 2.0, 5.0, 10.0, 12.5],
        "end": [1.0, 4.0, 6.0, 11.0, 13.5],
        "speaker": ["Interviewer", "Participant", "Participant", "Interviewer", "Participant"],
        "text": ["q", "a", "b", "q", "c"],
    })
    out = fx.temporal_features(s)
    assert out["latency"] == pytest.approx((1.0 + 1.5) / 2)
    assert out["pause_count"] == 1                      # 4.0 -> 5.0 between participant turns
    assert out["silence_speech"] == pytest.approx(1.0 / 4.0)
    assert out["speech_min"] == pytest.approx(4.0 / 60)


def test_absolutist_rate_ignores_tags_and_substrings() -> None:
    assert fx.absolutist_rate("i never sleep <laughter> always") == pytest.approx(2 / 4)
    assert fx.absolutist_rate("bestow nevermind") == 0


def test_criterion_scores_skip_missing_categories() -> None:
    mapping = fx.load_mapping()
    scores = fx.criterion_scores({"sadness": 0.4, "grief": 9.9, "joy": 0.2}, mapping)
    assert set(scores) == {f"dsm_C{i}" for i in range(1, 9)}
    c1 = next(c for c in mapping["criteria"] if c["id"] == "C1")
    n_present = len([c for c in c1["present"] if c not in mapping["missing_from_empath"]])
    assert scores["dsm_C1"] == pytest.approx(0.4 / n_present)  # grief is missing: ignored
    assert scores["dsm_C2"] < 0                                 # anhedonia: joy is "absent"


def test_feature_table_and_views() -> None:
    session = pd.DataFrame({"start": [0.0, 1.5], "end": [1.0, 3.0],
                            "speaker": ["Interviewer", "Participant"],
                            "text": ["how are you", "never sad <sigh>"]})
    table = fx.build_feature_table({7: session}, fx.EmpathScorer(FakeLexicon()))
    assert table.index.name == "Participant_ID"
    cols = fx.view_columns(table, ("temporal", "paralinguistic", "psycholinguistic"))
    assert set(cols) <= set(table.columns) and "absolutist_rate" in cols
    assert table.loc[7, "tag_stress"] == 1.0


# ----------------------------------------------------------------------------- graph

def test_ppmi_and_pagerank() -> None:
    cooc = np.array([[0, 4, 0], [4, 0, 1], [0, 1, 0]], float)
    adj = ppmi(cooc, np.array([5, 6, 2.0]), 10)
    assert (adj >= 0).all() and np.allclose(adj, adj.T) and adj[0, 2] == 0
    pr = pagerank(np.array([[0, 2, 1], [2, 0, 1], [1, 1, 0]], float))  # no dangling node
    assert pr.shape == (3,) and pr.sum() == pytest.approx(1, abs=1e-3)
    assert pr[2] < pr[0]


def test_assemble_layout_and_normalization() -> None:
    g = assemble(np.eye(3, dtype=np.float32), np.ones((2, 3), np.float32), np.ones((1, 3), np.float32))
    assert g.x.shape == (6, 3) and (g.n_vocab, g.n_train, g.n_new) == (3, 2, 1)
    assert g.train_slice == slice(3, 5) and g.new_slice == slice(5, 6)
    dense = np.zeros((6, 6))
    dense[g.edge_index[0], g.edge_index[1]] = g.edge_weight
    assert np.allclose(dense, dense.T)
    assert dense[3, 5] == 0 and dense[5, 3] == 0   # documents connect only through vocabulary


def _corpus():
    rng = np.random.default_rng(0)
    pos = ["sad tired alone never sleep night", "hopeless tired sad crying empty alone"]
    neg = ["work friends happy travel family", "music friends happy family weekend"]
    texts, y = [], []
    for i in range(20):
        base = pos if i % 2 else neg
        texts.append(" ".join(rng.permutation((" ".join(base) + " today really").split())) * 3)
        y.append(i % 2)
    return texts, np.array(y)


def test_word_graph_builder_is_inductive() -> None:
    texts, y = _corpus()
    b = WordGraphBuilder(vocab_size=8, window_size=4).fit(texts, y)
    assert len(b.vocab) == 8
    g_train, g_new = b.graph(), b.graph(["sad tired alone today", "unknown words only"])
    assert g_train.n_new == 0 and g_new.n_new == 2
    assert np.allclose(g_new.x[: g_new.n_vocab + g_new.n_train], g_train.x)  # train part unchanged
    assert g_new.x[g_new.new_slice][1].sum() == 0                            # OOV doc: no features


def test_empath_graph_builder_selects_by_anova() -> None:
    rng = np.random.default_rng(1)
    y = np.array([0, 1] * 10)
    emp = rng.random((20, 5)).astype(np.float32) * 0.1
    emp[:, 2] += y * 0.5                       # category 2 separates the classes
    b = EmpathGraphBuilder(n_categories=2).fit(emp, list("abcde"), y)
    assert "c" in b.categories and len(b.categories) == 2
    assert b.graph(emp[:3]).n_new == 3


# ----------------------------------------------------------------------------- evaluation

def test_threshold_rule() -> None:
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.10, 0.30, 0.35, 0.50, 0.40, 0.55])
    t, info = threshold_min_fn(p, y, max_fpr=0.25)    # at most 1 FP among 4 negatives
    # FN = 0 with FP = 1 for tau in (0.35, 0.40]; ties go to the higher threshold.
    assert info["feasible"] and info["fn"] == 0 and info["fp"] == 1 and t == pytest.approx(0.40)
    t, info = threshold_min_fn(p, y, max_fpr=0.0)
    assert info["fp"] == 0 and info["fn"] == 1


def test_paired_bootstrap_auc() -> None:
    from src.common.evaluation import bootstrap_auc, paired_bootstrap_auc
    rng = np.random.default_rng(0)
    y = np.array([0] * 30 + [1] * 15)
    good = y + rng.normal(0, 0.4, len(y))      # informative scores
    noise = rng.random(len(y))                 # uninformative scores
    auc, lo, hi = bootstrap_auc(y, good, b=300)
    assert lo <= auc <= hi and auc > 0.8
    res = paired_bootstrap_auc(y, good, noise, b=300)
    assert res["delta"] > 0 and res["low"] > 0 and res["p_not_better"] < 0.05
    same = paired_bootstrap_auc(y, good, good, b=50)
    assert same["delta"] == 0 and same["p_not_better"] == 1.0


def test_evaluate_and_wilson() -> None:
    y = np.array([0, 0, 1, 1])
    m = evaluate(y, np.array([0.1, 0.6, 0.7, 0.2]), 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    lo, hi = wilson(5, 10)
    assert lo < 0.5 < hi
