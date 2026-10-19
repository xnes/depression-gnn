"""Heterogeneous document graphs (dissertation Section 4.6), numpy only.

Two vocabularies share one layout:
    WordGraphBuilder    word nodes: 250 TF-IDF terms chosen by ANOVA F-score on the training split;
                        word-word edges: PPMI over a sliding window; doc-word edges: TF-IDF
    EmpathGraphBuilder  category nodes: Empath categories chosen by ANOVA F-score on the training
                        split; category-category edges: document-level PPMI; doc-category edges:
                        normalized Empath scores

Node order is [vocabulary | training documents | new documents]. Vocabulary nodes start as one-hot
vectors, document nodes as their TF-IDF or Empath vector; vocabulary self-loops carry PageRank
(Burdisso et al., node-weighted GCN). New documents are appended for inference only and never
change the fitted vocabulary, IDF weights, or vocabulary edges.

Ported from legacy/GAT_GCN_FeExtr_limpo (3).ipynb cell 8 and legacy/Empath_GCN_Hybrid.ipynb cell 6.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif


@dataclass
class Graph:
    x: np.ndarray            # (n_nodes, n_vocab) initial node features
    edge_index: np.ndarray   # (2, n_edges)
    edge_weight: np.ndarray  # (n_edges,) symmetric-normalized weights
    n_vocab: int
    n_train: int
    n_new: int

    @property
    def train_slice(self) -> slice:
        return slice(self.n_vocab, self.n_vocab + self.n_train)

    @property
    def new_slice(self) -> slice:
        return slice(self.n_vocab + self.n_train, self.n_vocab + self.n_train + self.n_new)


def ppmi(counts_ij: np.ndarray, counts_i: np.ndarray, total: int) -> np.ndarray:
    """Positive PMI from co-occurrence counts; zero where undefined."""
    with np.errstate(divide="ignore", invalid="ignore"):
        p_i = counts_i / total
        pmi = np.log((counts_ij / total) / np.outer(p_i, p_i))
    pmi[~np.isfinite(pmi)] = 0.0
    pmi = np.maximum(pmi, 0.0)
    np.fill_diagonal(pmi, 0.0)
    return pmi.astype(np.float32)


def pagerank(adjacency: np.ndarray, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """PageRank over a weighted, column-normalized adjacency matrix."""
    n = adjacency.shape[0]
    col = adjacency.sum(axis=0)
    a = np.divide(adjacency, col, out=np.zeros_like(adjacency, dtype=np.float64), where=col > 0)
    rank = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        new = damping * a @ rank + (1 - damping) / n
        if np.abs(new - rank).max() < tol:
            return new.astype(np.float32)
        rank = new
    return rank.astype(np.float32)


def assemble(vocab_adj: np.ndarray, docs_train: np.ndarray, docs_new: np.ndarray | None = None) -> Graph:
    """Build the normalized heterogeneous graph D^-1/2 A D^-1/2 over all nodes."""
    v = vocab_adj.shape[0]
    docs_new = np.zeros((0, v), np.float32) if docs_new is None else docs_new
    docs = np.vstack([docs_train, docs_new]).astype(np.float32)
    n_docs = docs.shape[0]
    a = np.zeros((v + n_docs, v + n_docs), dtype=np.float32)
    a[:v, :v] = vocab_adj
    a[:v, v:] = docs.T
    a[v:, :v] = docs
    a[v:, v:] = np.eye(n_docs, dtype=np.float32)
    deg = a.sum(axis=1)
    d = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    a = d[:, None] * a * d[None, :]
    rows, cols = np.nonzero(a)
    return Graph(
        x=np.vstack([np.eye(v, dtype=np.float32), docs]),
        edge_index=np.vstack([rows, cols]).astype(np.int64),
        edge_weight=a[rows, cols].astype(np.float32),
        n_vocab=v, n_train=docs_train.shape[0], n_new=docs_new.shape[0],
    )


class WordGraphBuilder:
    """InductGCN word graph: TF-IDF + ANOVA vocabulary, windowed PPMI, PageRank self-loops."""

    def __init__(self, vocab_size: int = 250, window_size: int = 20, damping: float = 0.85):
        self.vocab_size, self.window_size, self.damping = vocab_size, window_size, damping

    def fit(self, texts: list[str], labels: np.ndarray) -> "WordGraphBuilder":
        tmp = TfidfVectorizer(max_features=5000, stop_words="english", lowercase=True,
                              min_df=2, max_df=0.9, token_pattern=r"\b[a-z]{3,}\b")
        x = tmp.fit_transform(texts)
        k = min(self.vocab_size, x.shape[1])
        keep = SelectKBest(f_classif, k=k).fit(x, labels).get_support(indices=True)
        self.vocab = tmp.get_feature_names_out()[keep]
        self.vectorizer = TfidfVectorizer(vocabulary=self.vocab, lowercase=True,
                                          token_pattern=r"\b[a-z]{3,}\b").fit(texts)
        self.vocab_adj = self._vocab_adjacency(texts)
        self.docs_train = self.doc_features(texts)
        return self

    def doc_features(self, texts: list[str]) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray().astype(np.float32)

    def _vocab_adjacency(self, texts: list[str]) -> np.ndarray:
        analyzer = self.vectorizer.build_analyzer()
        index = self.vectorizer.vocabulary_
        n = len(self.vocab)
        win = np.zeros(n)
        cooc = np.zeros((n, n))
        total = 0
        for text in texts:
            ids = [index[w] for w in analyzer(text) if w in index]
            for pos in range(len(ids) - self.window_size + 1):
                window = sorted(set(ids[pos:pos + self.window_size]))
                if len(window) < 2:
                    continue
                win[window] += 1
                for i, j in combinations(window, 2):
                    cooc[i, j] += 1
                    cooc[j, i] += 1
                total += 1
        if total == 0:
            raise ValueError("No co-occurrence window found; texts are shorter than window_size.")
        adj = ppmi(cooc, win, total)
        np.fill_diagonal(adj, pagerank(adj, self.damping))
        return adj

    def graph(self, new_texts: list[str] | None = None) -> Graph:
        new = None if new_texts is None else self.doc_features(new_texts)
        return assemble(self.vocab_adj, self.docs_train, new)


class EmpathGraphBuilder:
    """EmpathGCN category graph: ANOVA-selected categories, document-level PPMI, PageRank."""

    def __init__(self, n_categories: int = 150, damping: float = 0.85):
        self.n_categories, self.damping = n_categories, damping

    def fit(self, empath_train: np.ndarray, categories: list[str], labels: np.ndarray) -> "EmpathGraphBuilder":
        """empath_train: (n_docs, n_all_categories) Empath scores over `categories`."""
        k = min(self.n_categories, empath_train.shape[1])
        with np.errstate(divide="ignore", invalid="ignore"):
            selector = SelectKBest(f_classif, k=k).fit(empath_train, labels)
        self.keep = np.sort(selector.get_support(indices=True))
        self.categories = [categories[i] for i in self.keep]
        self.docs_train = empath_train[:, self.keep].astype(np.float32)
        active = (self.docs_train > 0).astype(np.float64)
        adj = ppmi(active.T @ active, active.sum(axis=0), active.shape[0])
        np.fill_diagonal(adj, pagerank(adj, self.damping))
        self.vocab_adj = adj
        return self

    def doc_features(self, empath_new: np.ndarray) -> np.ndarray:
        return empath_new[:, self.keep].astype(np.float32)

    def graph(self, empath_new: np.ndarray | None = None) -> Graph:
        new = None if empath_new is None else self.doc_features(empath_new)
        return assemble(self.vocab_adj, self.docs_train, new)
