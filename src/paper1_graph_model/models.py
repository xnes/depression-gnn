"""Graph models (dissertation Section 4.7). Requires torch and torch_geometric.

All models share one call signature:
    model(x, edge_index, edge_weight, prof, docs) -> logits for the document nodes in `docs`
where `docs` is a slice over node positions and `prof` holds the per-document professional
features (or None when a model uses the graph alone).

Ported from legacy/GAT_GCN_FeExtr_limpo (3).ipynb (cells 20, 26) and
legacy/Empath_GCN_Hybrid.ipynb (cell 9). Architectures are unchanged; the `docs` slice replaces
the legacy `vocab_size`/`doc_start` arguments. GATConv ignores edge weights, as in the legacy code.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


def _head(in_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, 32), nn.BatchNorm1d(32), nn.ReLU(),
                         nn.Dropout(dropout), nn.Linear(32, 2))


class _ProfFusion(nn.Module):
    """Concatenate graph document embeddings with an MLP over professional features."""

    def __init__(self, hidden_dim: int, n_prof: int, dropout: float):
        super().__init__()
        self.mlp = None
        if n_prof > 0:
            self.mlp = nn.Sequential(nn.Linear(n_prof, 32), nn.BatchNorm1d(32), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(32, 16))
        self.fusion = _head(hidden_dim + (16 if n_prof > 0 else 0), dropout)

    def forward(self, h_docs: torch.Tensor, prof: torch.Tensor | None) -> torch.Tensor:
        if self.mlp is not None:
            h_docs = torch.cat([h_docs, self.mlp(prof)], dim=1)
        return self.fusion(h_docs)


class InducTGCN(nn.Module):
    """Burdisso et al. baseline: one projection, one propagation step, linear classifier."""

    def __init__(self, n_vocab: int, n_prof: int = 0, hidden_dim: int = 64, dropout: float = 0.5, **_):
        super().__init__()
        self.projection = nn.Linear(n_vocab, hidden_dim, bias=False)
        self.classifier = nn.Linear(hidden_dim, 2, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight, prof, docs):
        h = self.dropout(F.relu(self.projection(x)))
        a = torch.sparse_coo_tensor(edge_index, edge_weight, (x.shape[0], x.shape[0]))
        return self.classifier(torch.sparse.mm(a, h))[docs]


class RobustInductGAT(nn.Module):
    """Two GAT layers with skip connections and batch norm, fused with professional features."""

    def __init__(self, n_vocab: int, n_prof: int, hidden_dim: int = 32, heads: int = 2, dropout: float = 0.5):
        super().__init__()
        self.gat1 = GATConv(n_vocab, hidden_dim, heads=heads, dropout=dropout, add_self_loops=True)
        self.skip1 = nn.Linear(n_vocab, hidden_dim * heads)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.skip2 = nn.Linear(hidden_dim * heads, hidden_dim)
        self.bn1, self.bn2 = nn.BatchNorm1d(hidden_dim * heads), nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = _ProfFusion(hidden_dim, n_prof, dropout)

    def encode(self, x, edge_index, edge_weight, attention: bool = False):
        h1 = self.dropout(F.elu(self.bn1(self.gat1(x, edge_index) + self.skip1(x))))
        # PyG returns attention whenever this argument is a bool, so pass None to skip it.
        out = self.gat2(h1, edge_index, return_attention_weights=True if attention else None)
        h2_raw, att = (out if attention else (out, None))
        h2 = self.dropout(F.elu(self.bn2(h2_raw + self.skip2(h1))))
        return h2, att

    def forward(self, x, edge_index, edge_weight, prof, docs):
        h, _ = self.encode(x, edge_index, edge_weight)
        return self.head(h[docs], prof)


class EmpathGCN(nn.Module):
    """Two GCN layers with skip connection and batch norm, fused with professional features."""

    def __init__(self, n_vocab: int, n_prof: int, hidden_dim: int = 64, dropout: float = 0.5, **_):
        super().__init__()
        self.gcn1, self.gcn2 = GCNConv(n_vocab, hidden_dim), GCNConv(hidden_dim, hidden_dim)
        self.skip = nn.Linear(n_vocab, hidden_dim)
        self.bn1, self.bn2 = nn.BatchNorm1d(hidden_dim), nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = _ProfFusion(hidden_dim, n_prof, dropout)

    def forward(self, x, edge_index, edge_weight, prof, docs):
        h1 = self.dropout(F.elu(self.bn1(self.gcn1(x, edge_index, edge_weight) + self.skip(x))))
        h2 = self.dropout(F.elu(self.bn2(self.gcn2(h1, edge_index, edge_weight) + h1)))
        return self.head(h2[docs], prof)


class HybridGCNGAT(nn.Module):
    """GCN layer followed by a GAT layer (legacy HybridEmpathModel / HybridInductModel)."""

    def __init__(self, n_vocab: int, n_prof: int, hidden_dim: int = 32, heads: int = 2, dropout: float = 0.5):
        super().__init__()
        self.gcn1 = GCNConv(n_vocab, hidden_dim)
        self.skip = nn.Linear(n_vocab, hidden_dim)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout)
        self.bn1, self.bn2 = nn.BatchNorm1d(hidden_dim), nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = _ProfFusion(hidden_dim, n_prof, dropout)

    def encode(self, x, edge_index, edge_weight, attention: bool = False):
        h1 = self.dropout(F.elu(self.bn1(self.gcn1(x, edge_index, edge_weight) + self.skip(x))))
        out = self.gat2(h1, edge_index, return_attention_weights=True if attention else None)
        h2_raw, att = (out if attention else (out, None))
        return self.dropout(F.elu(self.bn2(h2_raw + h1))), att

    def forward(self, x, edge_index, edge_weight, prof, docs):
        h, _ = self.encode(x, edge_index, edge_weight)
        return self.head(h[docs], prof)


ARCHITECTURES = {
    "inductgcn": InducTGCN,
    "gat": RobustInductGAT,
    "gcn": EmpathGCN,
    "gcn_gat": HybridGCNGAT,
}
