"""Skeleton graph construction for NTU RGB+D (25 joints) layout."""
import numpy as np

# NTU RGB+D joint connectivity — neighbor_base is 1-indexed (from pyskl source)
_NTU_NEIGHBOR_BASE = [
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
    (20, 19), (22, 8), (23, 8), (24, 12), (25, 12),
]
_NTU_NUM_NODES = 25
_NTU_CENTER = 20  # joint 21 in 1-indexed


def _edge2mat(link, num_node):
    """link is a list of (i, j); sets A[j, i] = 1 (matches pyskl convention)."""
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def _normalize_digraph(A):
    """Column-wise D^{-1} normalization: AD where D_ii = 1/col_sum_i."""
    V = A.shape[1]
    col_sum = A.sum(axis=0)
    Dn = np.zeros((V, V))
    for i in range(V):
        if col_sum[i] > 0:
            Dn[i, i] = col_sum[i] ** -1
    return A @ Dn


def get_spatial_graph(layout: str = 'nturgb+d') -> np.ndarray:
    """Return the spatial adjacency matrix of shape (3, V, V).

    Subset 0: self-links (identity), 1: inward-normalised, 2: outward-normalised.
    Reproduces the 'spatial' mode from pyskl's Graph class exactly.
    """
    assert layout == 'nturgb+d', f"Only 'nturgb+d' layout is supported, got {layout!r}"
    V = _NTU_NUM_NODES
    inward = [(i - 1, j - 1) for i, j in _NTU_NEIGHBOR_BASE]
    outward = [(j, i) for i, j in inward]
    self_link = [(i, i) for i in range(V)]

    Iden = _edge2mat(self_link, V)
    In = _normalize_digraph(_edge2mat(inward, V))
    Out = _normalize_digraph(_edge2mat(outward, V))
    return np.stack([Iden, In, Out])  # (3, V, V)
