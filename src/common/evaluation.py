"""Metrics, confidence intervals, and the screening threshold (dissertation Sections 4.8, 4.11)."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

THRESHOLD_GRID = np.round(np.arange(0.20, 0.6001, 0.01), 2)  # dissertation Eq. 4.6: tau in [0.2, 0.6]


def threshold_min_fn(probs: np.ndarray, y: np.ndarray, max_fpr: float,
                     grid: np.ndarray = THRESHOLD_GRID) -> tuple[float, dict]:
    """Smallest-FN threshold whose false-positive rate stays within `max_fpr`.

    The dissertation states the constraint as FP <= 6 on the development set (21 negatives after
    preprocessing). Expressed as a rate it transfers to calibration sets of other sizes.
    Ties in FN are broken by fewer FP, then by the higher threshold. If no threshold satisfies the
    constraint, the threshold with the fewest FP is returned and `feasible` is False.
    """
    y = np.asarray(y).astype(int)
    n_neg = max(int((y == 0).sum()), 1)
    rows = []
    for t in grid:
        pred = probs >= t
        fn = int(((y == 1) & ~pred).sum())
        fp = int(((y == 0) & pred).sum())
        rows.append((t, fn, fp))
    feasible = [r for r in rows if r[2] / n_neg <= max_fpr]
    if feasible:
        t, fn, fp = min(feasible, key=lambda r: (r[1], r[2], -r[0]))
    else:
        t, fn, fp = min(rows, key=lambda r: (r[2], r[1], -r[0]))
    return float(t), {"fn": fn, "fp": fp, "feasible": bool(feasible), "max_fpr": max_fpr}


def wilson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    z = norm.ppf(0.5 + level / 2)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def bootstrap_f1(y: np.ndarray, pred: np.ndarray, b: int = 1000, seed: int = 42,
                 level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI for macro F1 (B = 1000, seed 42, as in the dissertation)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(b):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(f1_score(y[idx], pred[idx], average="macro"))
    lo, hi = np.percentile(stats, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


def _bootstrap_indices(y: np.ndarray, b: int, seed: int) -> list[np.ndarray]:
    """Resamples with replacement that keep both classes (AUC is undefined otherwise)."""
    rng = np.random.default_rng(seed)
    draws = (rng.integers(0, len(y), len(y)) for _ in range(b))
    return [i for i in draws if len(np.unique(y[i])) == 2]


def bootstrap_auc(y: np.ndarray, probs: np.ndarray, b: int = 2000, seed: int = 42,
                  level: float = 0.95) -> tuple[float, float, float]:
    """ROC-AUC with a percentile bootstrap CI: (auc, low, high)."""
    y = np.asarray(y).astype(int)
    stats = [roc_auc_score(y[i], probs[i]) for i in _bootstrap_indices(y, b, seed)]
    lo, hi = np.percentile(stats, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(roc_auc_score(y, probs)), float(lo), float(hi)


def paired_bootstrap_auc(y: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray, b: int = 2000,
                         seed: int = 42, level: float = 0.95) -> dict:
    """Paired bootstrap of AUC(a) - AUC(b): both models are scored on the same resampled participants.

    p_not_better is the share of resamples in which model a does not beat model b (delta <= 0).
    """
    y = np.asarray(y).astype(int)
    deltas = np.array([roc_auc_score(y[i], probs_a[i]) - roc_auc_score(y[i], probs_b[i])
                       for i in _bootstrap_indices(y, b, seed)])
    lo, hi = np.percentile(deltas, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return {"delta": float(roc_auc_score(y, probs_a) - roc_auc_score(y, probs_b)),
            "low": float(lo), "high": float(hi), "p_not_better": float(np.mean(deltas <= 0)),
            "n_resamples": len(deltas)}


def evaluate(y: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """Screening metrics at a fixed threshold, with 95% intervals."""
    y = np.asarray(y).astype(int)
    pred = (probs >= threshold).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    out = {
        "n": len(y), "positives": int(y.sum()), "threshold": threshold,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "f1_macro": f1_score(y, pred, average="macro"),
        "f1_macro_ci": bootstrap_f1(y, pred),
        "recall": tp / (tp + fn) if tp + fn else np.nan, "recall_ci": wilson(tp, tp + fn),
        "precision": tp / (tp + fp) if tp + fp else np.nan, "precision_ci": wilson(tp, tp + fp),
        "specificity": tn / (tn + fp) if tn + fp else np.nan, "specificity_ci": wilson(tn, tn + fp),
        "accuracy": (tp + tn) / len(y), "accuracy_ci": wilson(tp + tn, len(y)),
    }
    if len(np.unique(y)) == 2:
        out["roc_auc"] = roc_auc_score(y, probs)
        out["pr_auc"] = average_precision_score(y, probs)
    return out
