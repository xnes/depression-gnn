"""End-to-end smoke test of the models and protocol on a tiny synthetic corpus.

Skipped where torch_geometric is not installed (run it in Colab).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch_geometric")
import torch  # noqa: E402

from src.paper1_graph_model import experiment as ex  # noqa: E402
from src.paper1_graph_model.corpus import Cohort  # noqa: E402

HP = {"hidden_dim": 16, "heads": 2, "dropout": 0.5, "lr": 5e-3, "weight_decay": 1e-4,
      "epochs": 5, "m": 3.5, "vocab_size": 8, "window_size": 4, "n_categories": 3}
CATS = list("abcde")


def _cohort(name: str, n: int, seed: int) -> Cohort:
    rng = np.random.default_rng(seed)
    y = np.array([i % 2 for i in range(n)])
    words = {0: "work friends happy travel family music", 1: "sad tired alone never sleep hopeless"}
    texts = [" ".join(rng.permutation((words[int(c)] + " today really fine").split())) * 4 for c in y]
    emp = rng.random((n, 5)).astype(np.float32)
    emp[:, 1] += y
    ids = np.arange(seed * 100, seed * 100 + n)
    feats = pd.DataFrame({"latency": rng.random(n), "pause_count": rng.random(n),
                          "silence_speech": np.nan, "speech_min": rng.random(n),
                          "tag_laughter": 0.0, "tag_stress": 0.0, "tag_breath": 0.0, "tag_hesitation": 0.0,
                          "dsm_C1": rng.random(n) + y, "absolutist_rate": rng.random(n)}, index=ids)
    return Cohort(name, ids, y, texts, emp, feats)


@pytest.mark.parametrize("arch,graph,views", [
    ("inductgcn", "words", ()),
    ("gat", "words", ()),
    ("gcn_gat", "words", ("temporal", "paralinguistic", "psycholinguistic")),
    ("gcn", "empath", ("psycholinguistic",)),
    ("gat", "empath", ("temporal", "paralinguistic", "psycholinguistic")),
])
def test_run_final(arch: str, graph: str, views: tuple) -> None:
    spec = ex.ModelSpec(f"{arch}-{graph}", arch, graph, views, empath_categories=CATS)
    pool, test = _cohort("pool", 20, 1), _cohort("test", 6, 2)
    result = ex.run_final(spec, HP, pool, [test], n_folds=2, seeds_per_fold=2, device=torch.device("cpu"))
    assert result["n_members"] == 4
    probs = result["probs"]["test"]
    assert probs.shape == (6,) and np.all((probs >= 0) & (probs <= 1))
    assert set(result["metrics"]["test"]) >= {"f1_macro", "recall", "threshold"}
    assert result["oof_metrics"]["n"] == 20
    if arch in ("gat", "gcn_gat"):
        att = ex.attention_to_vocabulary(result, spec, test, device=torch.device("cpu"))
        assert att.shape[0] == 6 and att.shape[1] >= (8 if graph == "words" else 3)
        assert list(att.index) == list(test.ids)


def test_merge_cohorts() -> None:
    a, b = _cohort("a", 4, 1), _cohort("b", 3, 2)
    merged = ex.merge_cohorts("ab", a, b)
    assert len(merged.y) == 7 and merged.empath.shape == (7, 5) and len(merged.texts) == 7
    assert list(merged.features.index) == list(merged.ids)
    with pytest.raises(ValueError):
        ex.merge_cohorts("aa", a, a)


def test_tune_runs_two_trials() -> None:
    spec = ex.ModelSpec("tune-smoke", "gcn", "empath", ("psycholinguistic",), empath_categories=CATS)
    study = ex.tune(spec, _cohort("train", 20, 3), n_trials=2, n_folds=2, n_repeats=1, seed=0,
                    device=torch.device("cpu"))
    assert len(study.trials) == 2 and 0 <= study.best_value <= 1
    assert set(study.best_params) == {"lr", "weight_decay", "epochs", "m"}  # restricted space
