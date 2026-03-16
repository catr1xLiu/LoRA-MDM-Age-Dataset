"""
NTU RGB+D skeleton graph construction for ST-GCN++.

The graph models the 25-joint NTU RGB+D skeleton as a directed graph with
three subsets: self-links, inward edges, and outward edges. This 'spatial'
partition strategy (from ST-GCN) gives the GCN three distinct message-passing
pathways per layer.

Reference: PYSKL pyskl/utils/graph.py
"""

import numpy as np


# ---------------------------------------------------------------------------
# NTU RGB+D skeleton definition (25 joints, 1-indexed in the official format)
# ---------------------------------------------------------------------------
#
# Joint index mapping (1-indexed → 0-indexed):
#   1  → 0  : pelvis / hip center
#   2  → 1  : spine (lower)
#   3  → 2  : spine (middle)
#   4  → 3  : neck
#   5  → 4  : left shoulder
#   6  → 5  : left elbow
#   7  → 6  : left wrist
#   8  → 7  : left hand
#   9  → 8  : right shoulder
#   10 → 9  : right elbow
#   11 → 10 : right wrist
#   12 → 11 : right hand
#   13 → 12 : left hip
#   14 → 13 : left knee
#   15 → 14 : left ankle
#   16 → 15 : left foot
#   17 → 16 : right hip
#   18 → 17 : right knee
#   19 → 18 : right ankle
#   20 → 19 : right foot
#   21 → 20 : spine (upper / torso)
#   22 → 21 : left hand tip
#   23 → 22 : left thumb
#   24 → 23 : right hand tip
#   25 → 24 : right thumb

NUM_JOINTS = 25

# Inward edges: (child, parent) in 0-indexed form.
# Derived from the 1-indexed pairs in the NTU spec by subtracting 1 from each.
_NTU_INWARD = [
    (0, 1),   # pelvis ← spine_lower
    (1, 20),  # spine_lower ← torso
    (2, 20),  # spine_mid ← torso
    (3, 2),   # neck ← spine_mid
    (4, 20),  # l_shoulder ← torso
    (5, 4),   # l_elbow ← l_shoulder
    (6, 5),   # l_wrist ← l_elbow
    (7, 6),   # l_hand ← l_wrist
    (8, 20),  # r_shoulder ← torso
    (9, 8),   # r_elbow ← r_shoulder
    (10, 9),  # r_wrist ← r_elbow
    (11, 10), # r_hand ← r_wrist
    (12, 0),  # l_hip ← pelvis
    (13, 12), # l_knee ← l_hip
    (14, 13), # l_ankle ← l_knee
    (15, 14), # l_foot ← l_ankle
    (16, 0),  # r_hip ← pelvis
    (17, 16), # r_knee ← r_hip
    (18, 17), # r_ankle ← r_knee
    (19, 18), # r_foot ← r_ankle
    (21, 22), # l_hand_tip ← l_thumb (unusual — matches PYSKL exactly)
    (20, 20), # torso self-loop (center node)
    (22, 7),  # l_thumb ← l_hand
    (23, 24), # r_hand_tip ← r_thumb
    (24, 11), # r_thumb ← r_hand
]

# Center joint (torso / spine upper, 0-indexed)
_CENTER = 20


def _edge_to_adjacency(edges: list[tuple[int, int]], num_nodes: int) -> np.ndarray:
    """Convert an edge list to a binary adjacency matrix.

    Args:
        edges: List of (source, destination) pairs.
        num_nodes: Total number of nodes.

    Returns:
        A (num_nodes, num_nodes) matrix with A[j, i] = 1 for edge (i→j).
    """
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for src, dst in edges:
        A[dst, src] = 1.0
    return A


def _normalize_digraph(A: np.ndarray) -> np.ndarray:
    """Right-multiply A by the inverse degree matrix: A_norm = A @ D^{-1}.

    This normalises each column so that messages flowing into a node are
    weighted by the inverse out-degree of the source node.

    Args:
        A: Square adjacency matrix of shape (N, N).

    Returns:
        Degree-normalised matrix of the same shape.
    """
    degree = A.sum(axis=0)  # out-degree of each node (column sum)
    N = A.shape[0]
    D_inv = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        if degree[i] > 0:
            D_inv[i, i] = 1.0 / degree[i]
    return A @ D_inv


class NTUGraph:
    """Spatial-partition adjacency matrix for the NTU RGB+D 25-joint skeleton.

    Produces three adjacency subsets following the spatial partition strategy
    from ST-GCN (Yan et al., AAAI 2018):

        A[0]: Identity (self-connections)
        A[1]: Inward edges, degree-normalised
        A[2]: Outward edges, degree-normalised

    The resulting tensor has shape (3, 25, 25) and is used as the initial
    (and learnable) graph in ST-GCN++.
    """

    def __init__(self) -> None:
        self_links = [(i, i) for i in range(NUM_JOINTS)]
        inward = _NTU_INWARD
        outward = [(dst, src) for (src, dst) in inward]

        # Subset 0: identity
        A_identity = _edge_to_adjacency(self_links, NUM_JOINTS)

        # Subset 1: inward, normalised
        A_inward = _normalize_digraph(_edge_to_adjacency(inward, NUM_JOINTS))

        # Subset 2: outward, normalised
        A_outward = _normalize_digraph(_edge_to_adjacency(outward, NUM_JOINTS))

        # Stack into a single (3, 25, 25) tensor
        self.A = np.stack([A_identity, A_inward, A_outward], axis=0)

    @property
    def num_subsets(self) -> int:
        return self.A.shape[0]

    @property
    def num_joints(self) -> int:
        return NUM_JOINTS
