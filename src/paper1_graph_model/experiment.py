"""Experiment protocol for Paper 1: tune and calibrate on the training pool only, score held-out
cohorts once.

Protocol (v2, 2026-10-18):
1. Training pool = DAIC-WOZ train + dev (140 participants). Held-out cohorts: DAIC-WOZ official test
   and E-DAIC, each scored once.
2. Architecture and graph hyperparameters are fixed at the values the dissertation reports as selected
   (Table 9). Optuna (TPE) tunes only learning rate, weight decay, epochs and the class-weight
   multiplier m, with repeated stratified k-fold cross-validation inside the pool. Objective: average
   precision of out-of-fold probabilities. A "full" space (all of Table 9) remains available.
3. Cross-fitted ensemble: for each of the k folds, `seeds_per_fold` members are trained on the other
   folds. The ensemble is all k * seeds_per_fold members (20 by default). Out-of-fold probabilities come
   from the same members, so the threshold is calibrated on exactly the models that score the held-out
   cohorts. (v1 calibrated on fold models and applied the threshold to models refit on all data, whose
   probabilities lived on a different scale.)
4. Threshold: fewest false negatives with false-positive rate <= max_fpr (dissertation FP_max = 6 of
   21 dev negatives). FP_max is not tuned: it encodes a clinical trade-off.

Graph vocabulary, ANOVA selection and feature scaling are always refit on the fold's training part.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from src.common.evaluation import evaluate, threshold_min_fn
from src.common.features import view_columns
from src.paper1_graph_model.corpus import Cohort
from src.paper1_graph_model.graph import EmpathGraphBuilder, Graph, WordGraphBuilder
from src.paper1_graph_model.models import ARCHITECTURES

DEFAULT_MAX_FPR = 6 / 21  # dissertation FP_max = 6 on the 21 negatives of the dev cohort

# Values reported as selected in dissertation Table 9; InducTGCN keeps the legacy embedding size 64.
FIXED = {"hidden_dim": 32, "heads": 2, "dropout": 0.5, "vocab_size": 250, "window_size": 20,
         "n_categories": 150}
FIXED_BY_ARCH = {"inductgcn": {"hidden_dim": 64}}


@dataclass
class ModelSpec:
    name: str
    arch: str                       # key of ARCHITECTURES
    graph: str                      # "words" or "empath"
    views: tuple[str, ...] = ()     # professional feature views; () = graph only
    empath_categories: list[str] = field(default_factory=list)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def merge_cohorts(name: str, *cohorts: Cohort) -> Cohort:
    """Concatenate cohorts with distinct participant ids (e.g. DAIC-WOZ train and dev)."""
    ids = np.concatenate([c.ids for c in cohorts])
    if len(set(ids)) != len(ids):
        raise ValueError("Participant ids overlap; cohorts cannot be merged.")
    phq = None if any(c.phq is None for c in cohorts) else np.concatenate([c.phq for c in cohorts])
    return Cohort(name, ids, np.concatenate([c.y for c in cohorts]), sum((c.texts for c in cohorts), []),
                  np.vstack([c.empath for c in cohorts]),
                  pd.concat([c.features.loc[c.ids] for c in cohorts]), phq)


def fixed_hp(spec: ModelSpec) -> dict:
    return {**FIXED, **FIXED_BY_ARCH.get(spec.arch, {})}


def suggest(trial, spec: ModelSpec, space: str = "restricted") -> dict:
    """Restricted: optimization hyperparameters only. Full: all of dissertation Table 9 except FP_max."""
    hp = fixed_hp(spec)
    hp.update({
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 5e-5, 5e-3, log=True),
        "epochs": trial.suggest_int("epochs", 300, 600, step=50),
        "m": trial.suggest_float("m", 2.0, 5.0),
    })
    if space == "full":
        hp["hidden_dim"] = trial.suggest_categorical("hidden_dim", [16, 24, 32, 48, 64])
        hp["dropout"] = trial.suggest_float("dropout", 0.40, 0.60)
        if spec.arch in ("gat", "gcn_gat"):
            hp["heads"] = trial.suggest_categorical("heads", [2, 4])
        if spec.graph == "words":
            hp["vocab_size"] = trial.suggest_categorical("vocab_size", [150, 200, 250, 300])
            hp["window_size"] = trial.suggest_categorical("window_size", [15, 20, 22, 25])
        else:
            hp["n_categories"] = trial.suggest_categorical("n_categories", [100, 125, 150, 175, 194])
    return hp


def hp_from_params(spec: ModelSpec, params: dict) -> dict:
    """Optuna best_params completed with the fixed hyperparameters."""
    return {**fixed_hp(spec), **params}


# ----------------------------------------------------------------------------- building blocks

class LabelSmoothingCE(nn.Module):
    """Class-weighted cross-entropy with label smoothing (epsilon = 0.1)."""

    def __init__(self, weight: torch.Tensor, smoothing: float = 0.1):
        super().__init__()
        self.weight, self.smoothing = weight, smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        soft = torch.full_like(logp, self.smoothing / (logits.size(1) - 1))
        soft.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        return (-(soft * logp).sum(-1) * self.weight[target]).mean()


def class_weights(y: np.ndarray, m: float) -> np.ndarray:
    """Dissertation Eq. 4.5: w1 = m N / (2 N1), w0 = N / (2 N0)."""
    n, n1 = len(y), int(y.sum())
    return np.array([n / (2 * (n - n1)), m * n / (2 * n1)], dtype=np.float32)


_BUILDER_CACHE: dict = {}


def fit_builder(spec: ModelSpec, hp: dict, train: Cohort):
    """Fit (or reuse) the graph builder for this training subset and graph hyperparameters."""
    if spec.graph == "words":
        key = ("words", train.name, tuple(train.ids), hp["vocab_size"], hp["window_size"])
        make = lambda: WordGraphBuilder(hp["vocab_size"], hp["window_size"]).fit(train.texts, train.y)
    else:
        key = ("empath", train.name, tuple(train.ids), hp["n_categories"])
        make = lambda: EmpathGraphBuilder(hp["n_categories"]).fit(train.empath, spec.empath_categories, train.y)
    if key not in _BUILDER_CACHE:
        _BUILDER_CACHE[key] = make()
    return _BUILDER_CACHE[key]


def new_graph(spec: ModelSpec, builder, cohort: Cohort | None) -> Graph:
    if cohort is None:
        return builder.graph()
    return builder.graph(cohort.texts if spec.graph == "words" else cohort.empath)


class FeatureScaler:
    """Standardize professional features with training statistics; impute NaN with the train mean."""

    def __init__(self, spec: ModelSpec, train: Cohort):
        self.cols = view_columns(train.features, spec.views) if spec.views else []
        x = train.features.loc[train.ids, self.cols].to_numpy(float)
        self.mean = np.nanmean(x, axis=0) if self.cols else np.zeros(0)
        self.mean = np.where(np.isnan(self.mean), 0.0, self.mean)
        std = np.nanstd(x, axis=0) if self.cols else np.zeros(0)
        self.std = np.where((std > 0) & ~np.isnan(std), std, 1.0)

    def __call__(self, cohort: Cohort) -> np.ndarray:
        if not self.cols:
            return np.zeros((len(cohort.ids), 0), np.float32)
        x = cohort.features.loc[cohort.ids, self.cols].to_numpy(float)
        x = np.where(np.isnan(x), self.mean, x)
        return ((x - self.mean) / self.std).astype(np.float32)


def _tensors(g: Graph, device):
    return (torch.tensor(g.x, device=device), torch.tensor(g.edge_index, device=device),
            torch.tensor(g.edge_weight, device=device))


def train_member(spec: ModelSpec, hp: dict, g: Graph, prof: np.ndarray, y: np.ndarray,
                 seed: int, device) -> nn.Module:
    set_seed(seed)
    x, ei, ew = _tensors(g, device)
    p = torch.tensor(prof, device=device) if prof.shape[1] else None
    target = torch.tensor(y, dtype=torch.long, device=device)
    model = ARCHITECTURES[spec.arch](g.n_vocab, prof.shape[1], hidden_dim=hp["hidden_dim"],
                                     heads=hp.get("heads", 2), dropout=hp["dropout"]).to(device)
    loss_fn = LabelSmoothingCE(torch.tensor(class_weights(y, hp["m"]), device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-5)
    for _ in range(hp["epochs"]):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(x, ei, ew, p, g.train_slice), target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        sched.step()
    return model.eval()


@torch.no_grad()
def predict(model: nn.Module, g: Graph, prof: np.ndarray, device) -> np.ndarray:
    x, ei, ew = _tensors(g, device)
    p = torch.tensor(prof, device=device) if prof.shape[1] else None
    return F.softmax(model(x, ei, ew, p, g.new_slice), dim=1)[:, 1].cpu().numpy()


@dataclass
class Member:
    model: nn.Module
    builder: object
    scaler: FeatureScaler

    def predict(self, spec: ModelSpec, cohort: Cohort, device) -> np.ndarray:
        return predict(self.model, new_graph(spec, self.builder, cohort), self.scaler(cohort), device)


def fit_members(spec: ModelSpec, hp: dict, train: Cohort, seeds: list[int], device) -> list[Member]:
    builder = fit_builder(spec, hp, train)
    scaler = FeatureScaler(spec, train)
    g_train = new_graph(spec, builder, None)
    return [Member(train_member(spec, hp, g_train, scaler(train), train.y, s, device), builder, scaler)
            for s in seeds]


# ----------------------------------------------------------------------------- tuning

def cv_oof(spec: ModelSpec, hp: dict, pool: Cohort, n_folds: int, n_repeats: int, seed: int,
           device) -> np.ndarray:
    """Out-of-fold probabilities averaged over repeats (one member per fold)."""
    total = np.zeros(len(pool.y))
    splitter = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)
    for k, (tr, va) in enumerate(splitter.split(pool.texts, pool.y)):
        member = fit_members(spec, hp, pool.subset(tr), [seed + k], device)[0]
        total[va] += member.predict(spec, pool.subset(va), device)
    return total / n_repeats


def tune(spec: ModelSpec, pool: Cohort, n_trials: int, n_folds: int, n_repeats: int, seed: int,
         device, storage: str | None = None, space: str = "restricted", study_name: str | None = None):
    import optuna
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
                                storage=storage, study_name=study_name or spec.name,
                                load_if_exists=storage is not None)

    def objective(trial):
        hp = suggest(trial, spec, space)
        return average_precision_score(pool.y, cv_oof(spec, hp, pool, n_folds, n_repeats, seed, device))

    remaining = n_trials - len([t for t in study.trials if t.state.is_finished()])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    return study


# ----------------------------------------------------------------------------- final run

def run_final(spec: ModelSpec, hp: dict, pool: Cohort, cohorts: list[Cohort], n_folds: int = 5,
              seeds_per_fold: int = 4, seed: int = 42, max_fpr: float = DEFAULT_MAX_FPR,
              device=None) -> dict:
    """Cross-fitted ensemble: calibrate tau on its own out-of-fold probabilities, score cohorts once."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oof = np.zeros(len(pool.y))
    members: list[Member] = []
    folds = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for k, (tr, va) in enumerate(folds.split(pool.texts, pool.y)):
        seeds = [seed + 100 * k + s for s in range(seeds_per_fold)]
        fold_members = fit_members(spec, hp, pool.subset(tr), seeds, device)
        oof[va] = np.mean([m.predict(spec, pool.subset(va), device) for m in fold_members], axis=0)
        members += fold_members
    tau, tau_info = threshold_min_fn(oof, pool.y, max_fpr)
    probs = {c.name: np.mean([m.predict(spec, c, device) for m in members], axis=0) for c in cohorts}
    return {
        "spec": spec.name, "hp": hp, "n_members": len(members), "tau": tau, "tau_info": tau_info,
        "oof_pr_auc": average_precision_score(pool.y, oof),
        "oof_metrics": evaluate(pool.y, oof, tau),
        "metrics": {c.name: evaluate(c.y, probs[c.name], tau) for c in cohorts},
        "probs": probs, "members": members,
    }


@torch.no_grad()
def attention_to_vocabulary(result: dict, spec: ModelSpec, cohort: Cohort, device=None) -> pd.DataFrame:
    """Mean second-layer attention from each vocabulary node to each document of `cohort`.

    Averaged over heads and members. Members fitted on different folds may select different
    vocabularies; each name is averaged over the members that contain it.
    Rows: Participant_ID; columns: words or Empath categories. Only for "gat" and "gcn_gat".
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = []
    for m in result["members"]:
        g = new_graph(spec, m.builder, cohort)
        x, ei, ew = _tensors(g, device)
        _, (att_index, alpha) = m.model.encode(x, ei, ew, attention=True)
        src, dst = att_index.cpu().numpy()
        a = alpha.mean(dim=1).cpu().numpy() if alpha.dim() > 1 else alpha.cpu().numpy()
        start = g.new_slice.start
        keep = (src < g.n_vocab) & (dst >= start)
        total = np.zeros((g.n_new, g.n_vocab))
        np.add.at(total, (dst[keep] - start, src[keep]), a[keep])
        names = list(m.builder.vocab) if spec.graph == "words" else m.builder.categories
        frames.append(pd.DataFrame(total, index=cohort.ids, columns=names))
    return pd.concat(frames).groupby(level=0).mean().loc[cohort.ids]
