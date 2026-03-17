"""
ST-GCN++ model: backbone and classifier head.

Architecture overview
---------------------
The full model is composed of two independent classes:

    STGCNBackbone   — Spatial-temporal GCN feature extractor
    GCNClassifier   — Global-average-pool + linear classification head

Inside the backbone each layer is an STGCNBlock which chains:

    UnitGCN  →  MSTCN  (+residual)

UnitGCN is an adaptive graph convolution (mode 'init': the adjacency matrix
is a fully learnable nn.Parameter) with an optional residual inside the graph
conv itself. MSTCN is a multi-branch temporal convolution that captures
patterns at different time-scales and dilation rates.

All design choices and hyper-parameters are taken directly from the PYSKL
ST-GCN++ configuration for NTU RGB+D 120 with official 3-D skeletons:
    configs/stgcn++/stgcn++_ntu120_xsub_3dkp/{j,b}.py

Reference: PYSKL pyskl/models/gcns/{stgcn,utils/gcn,utils/tcn}.py
"""

import torch
import torch.nn as nn
import numpy as np

from .graph import NTUGraph

# Small epsilon used to avoid floating-point boundary issues when computing
# integer channel counts after channel inflation.
_EPS = 1e-4


# ============================================================================
# Low-level building blocks
# ============================================================================

class UnitTCN(nn.Module):
    """Single-branch temporal convolution (Conv2d along the time axis).

    The spatial dimension V is treated like a batch dimension, so the kernel
    is always 1 in the V direction.  Dilation is supported to increase the
    effective receptive field without extra parameters.

    Args:
        in_channels:  Input feature channels.
        out_channels: Output feature channels.
        kernel_size:  Temporal kernel size. Default: 9.
        stride:       Temporal stride. Default: 1.
        dilation:     Temporal dilation. Default: 1.
        norm:         Whether to apply BatchNorm after the conv. Default: True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 9,
        stride: int = 1,
        dilation: int = 1,
        norm: bool = True,
    ) -> None:
        super().__init__()
        # Padding is chosen so that output length = ceil(input / stride) when
        # dilation is taken into account.
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels) if norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class MSTCN(nn.Module):
    """Multi-Scale Temporal Convolution.

    Six parallel branches capture patterns at different temporal resolutions:
        (3, dil=1)  (3, dil=2)  (3, dil=3)  (3, dil=4)  MaxPool  1×1

    Each branch first projects to a fraction of the output channels, then the
    concatenated outputs are fused by a transform layer (BN→ReLU→1×1 conv).

    Channel split: the first branch receives the 'remainder' channels so that
    the total always equals out_channels exactly:
        rem  = out_channels - (out_channels // 6) * 5
        rest = out_channels // 6

    Args:
        in_channels:  Input feature channels.
        out_channels: Output feature channels (== input channels in practice).
        stride:       Temporal stride applied by every branch. Default: 1.
        dropout:      Dropout probability after the final BN. Default: 0.0.
    """

    _MS_CFG = [(3, 1), (3, 2), (3, 3), (3, 4), ("max", 3), "1x1"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.act = nn.ReLU()
        num_branches = len(self._MS_CFG)

        # Channel sizes per branch
        mid = out_channels // num_branches
        rem = out_channels - mid * (num_branches - 1)  # first branch is larger

        branch_channels = [rem] + [mid] * (num_branches - 1)
        self.branches = nn.ModuleList()

        for i, cfg in enumerate(self._MS_CFG):
            branch_c = branch_channels[i]

            if cfg == "1x1":
                self.branches.append(
                    nn.Conv2d(in_channels, branch_c, 1, stride=(stride, 1))
                )

            elif isinstance(cfg, tuple) and cfg[0] == "max":
                pool_k = cfg[1]
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, 1),
                        nn.BatchNorm2d(branch_c),
                        self.act,
                        nn.MaxPool2d(
                            kernel_size=(pool_k, 1),
                            stride=(stride, 1),
                            padding=(1, 0),
                        ),
                    )
                )

            else:
                # (kernel_size, dilation)
                k, d = cfg
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, 1),
                        nn.BatchNorm2d(branch_c),
                        self.act,
                        UnitTCN(branch_c, branch_c, kernel_size=k, stride=stride,
                                dilation=d, norm=False),
                    )
                )

        # Fusion: BN → ReLU → 1×1 conv
        self.transform = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            self.act,
            nn.Conv2d(out_channels, out_channels, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outs = [branch(x) for branch in self.branches]
        feat = torch.cat(branch_outs, dim=1)
        feat = self.transform(feat)
        return self.drop(self.bn(feat))


class UnitGCN(nn.Module):
    """Adaptive Graph Convolution (ST-GCN++ variant).

    The adjacency matrix A is fully learnable (``adaptive='init'``): it is
    stored as ``nn.Parameter`` and updated end-to-end.  An optional residual
    skip inside the graph conv handles channel mismatches.

    Convolution order (``conv_pos='pre'``):
        1.  Conv2d(C_in → C_out * K, 1×1)    — mix channels per node
        2.  Reshape to (N, K, C_out, T, V)
        3.  Einsum with A: (N,K,C,T,V) × (K,V,W) → (N,C,T,W)

    This is mathematically equivalent to K independent GCN layers sharing
    temporal processing but having distinct spatial aggregation graphs.

    Args:
        in_channels:  Input feature channels.
        out_channels: Output feature channels.
        A:            Initial adjacency matrix of shape (K, V, V).
        with_res:     Add a residual path inside the GCN. Default: True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        with_res: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = A.size(0)
        self.with_res = with_res

        # Learnable adjacency matrix (fully trainable from the structural init)
        self.A = nn.Parameter(A.clone())

        # Pre-graph convolution: expand to K output subsets at once
        self.conv = nn.Conv2d(in_channels, out_channels * self.num_subsets, 1)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

        if with_res:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.down = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C_in, T, V)

        Returns:
            (N, C_out, T, V)
        """
        N, C, T, V = x.shape
        res = self.down(x) if self.with_res else 0

        # (N, C_out*K, T, V) → (N, K, C_out, T, V)
        y = self.conv(x).view(N, self.num_subsets, self.out_channels, T, V)

        # Graph aggregation: sum over source nodes weighted by A
        # y[n,k,c,t,v] = sum_w  y[n,k,c,t,w] * A[k,w,v]
        y = torch.einsum("nkctv,kvw->nctw", y, self.A).contiguous()

        return self.act(self.bn(y) + res)


# ============================================================================
# Core block
# ============================================================================

class STGCNBlock(nn.Module):
    """One Spatial-Temporal GCN block: GCN → MSTCN (+ outer residual).

    The outer residual connects the block input to the MSTCN output.  It is a
    learned 1×1 temporal conv when dimensions differ, and identity otherwise.

    Args:
        in_channels:  Input feature channels.
        out_channels: Output feature channels.
        A:            Adjacency matrix (K, V, V), cloned per block.
        stride:       Temporal stride applied in the TCN. Default: 1.
        residual:     Whether to use the outer residual. Default: True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        stride: int = 1,
        residual: bool = True,
    ) -> None:
        super().__init__()

        self.gcn = UnitGCN(in_channels, out_channels, A, with_res=True)
        self.tcn = MSTCN(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda _x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            # 1×1 temporal conv to match dimensions
            self.residual = UnitTCN(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C_in, T, V)

        Returns:
            (N, C_out, T_out, V)  where T_out = T // stride
        """
        res = self.residual(x)
        return self.relu(self.tcn(self.gcn(x)) + res)


# ============================================================================
# Backbone
# ============================================================================

class STGCNBackbone(nn.Module):
    """ST-GCN++ backbone: 10 stacked STGCNBlocks with channel inflation.

    The network processes skeleton sequences of shape (N, M, T, V, C):
        N  — batch size
        M  — number of persons per clip (always 2 for NTU)
        T  — temporal length (100 input frames)
        V  — number of joints (25)
        C  — coordinate channels (3 for 3-D skeletons)

    A data batch-normalisation layer (BatchNorm1d) is applied first over
    (V × C) to standardise raw skeleton coordinates before the GCN layers.

    Channel schedule (base_channels=64, ch_ratio=2):
        Stages 1-4 : 64 channels
        Stage 5    : 64 → 128 (stride 2, temporal downsampling)
        Stages 6-7 : 128 channels
        Stage 8    : 128 → 256 (stride 2, temporal downsampling)
        Stages 9-10: 256 channels

    Output shape: (N, M, 256, T/4, V)  →  T=100 becomes 25.

    Args:
        in_channels:     Coordinate dimension. Default: 3 (3-D skeleton).
        base_channels:   Initial channel width. Default: 64.
        ch_ratio:        Channel multiplier at inflation stages. Default: 2.
        num_stages:      Number of GCN blocks. Default: 10.
        inflate_stages:  Block indices (1-based) where channels double.
        down_stages:     Block indices (1-based) where temporal stride = 2.
        graph:           Pre-built NTUGraph. If None a new one is created.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        ch_ratio: int = 2,
        num_stages: int = 10,
        inflate_stages: list[int] | None = None,
        down_stages: list[int] | None = None,
        graph: NTUGraph | None = None,
    ) -> None:
        super().__init__()

        if inflate_stages is None:
            inflate_stages = [5, 8]
        if down_stages is None:
            down_stages = [5, 8]
        if graph is None:
            graph = NTUGraph()

        A = torch.tensor(graph.A, dtype=torch.float32)
        V = graph.num_joints

        # Data batch-normalisation over all (V × C) features at each time step.
        # Applied in VC mode: shape (N*M, V*C, T).
        self.data_bn = nn.BatchNorm1d(in_channels * V)

        # Build the sequence of STGCNBlocks
        modules = []
        _current_channels = base_channels
        inflate_count = 0

        # Block 1: project from in_channels to base_channels (no residual)
        if in_channels != base_channels:
            modules.append(
                STGCNBlock(in_channels, base_channels, A.clone(),
                           stride=1, residual=False)
            )

        for stage_idx in range(2, num_stages + 1):
            stride = 2 if stage_idx in down_stages else 1
            ch_in = _current_channels

            if stage_idx in inflate_stages:
                inflate_count += 1

            ch_out = int(base_channels * ch_ratio ** inflate_count + _EPS)
            _current_channels = ch_out

            modules.append(STGCNBlock(ch_in, ch_out, A.clone(), stride=stride))

        self.gcn = nn.ModuleList(modules)
        self.in_channels = in_channels
        self._V = V

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, M, T, V, C)

        Returns:
            (N, M, 256, T//4, V)
        """
        N, M, T, V, C = x.shape

        # --- Data batch-norm ---
        # Permute to (N, M, V, C, T) so V and C are contiguous for reshape
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        # Merge persons into batch, flatten V×C
        x = self.data_bn(x.view(N * M, V * C, T))
        # Restore and convert to (N*M, C, T, V) for the GCN layers
        x = (
            x.view(N, M, V, C, T)
             .permute(0, 1, 3, 4, 2)   # (N, M, C, T, V)
             .contiguous()
             .view(N * M, C, T, V)
        )

        # --- GCN blocks ---
        for block in self.gcn:
            x = block(x)

        # Restore person dimension: (N, M, C_out, T_out, V)
        return x.view(N, M, *x.shape[1:])


# ============================================================================
# Classifier head
# ============================================================================

class GCNClassifier(nn.Module):
    """Classification head for skeleton action recognition.

    Performs global average pooling over the temporal and joint dimensions,
    averages over the person dimension, then applies a single linear layer.

    Args:
        in_channels: Feature channels from the backbone (256).
        num_classes: Number of action classes (120 for NTU-120).
        dropout:     Dropout probability. Default: 0 (off).
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 120,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Named fc_cls to match PYSKL checkpoint keys (cls_head.fc_cls.*)
        self.fc_cls = nn.Linear(in_channels, num_classes)
        self.in_channels = in_channels

        # Initialise classifier weights with small normal distribution
        nn.init.normal_(self.fc_cls.weight, std=0.01)
        nn.init.zeros_(self.fc_cls.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, M, C, T, V) — backbone output

        Returns:
            (N, num_classes) — raw logits
        """
        N, M, C, T, V = x.shape

        # Pool over (T, V): (N*M, C, T, V) → (N*M, C, 1, 1) → (N*M, C)
        x = self.pool(x.view(N * M, C, T, V)).view(N, M, C)

        # Average over persons: (N, M, C) → (N, C)
        x = x.mean(dim=1)

        return self.fc_cls(self.drop(x))


# ============================================================================
# Bottleneck classifier head for age classification (z_age extraction)
# ============================================================================


class AgeClassifierHead(nn.Module):
    """Bottleneck classifier head for 3-class age classification.

    Pool → Dropout → Linear(256→z_dim) → ReLU → Linear(z_dim→num_classes)

    The z_dim-dimensional intermediate representation is the z_age embedding
    used for LoRA-MDM conditioning.  Use ``get_z()`` to extract it without
    running the final classification layer.

    Args:
        in_channels: Feature channels from the backbone (256).
        z_dim:       Bottleneck / z_age dimensionality. Default: 32.
        num_classes: Age group classes (3: Young / Adult / Elderly).
        dropout:     Dropout probability applied before the bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 256,
        z_dim: int = 32,
        num_classes: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc_z = nn.Linear(in_channels, z_dim)
        self.relu = nn.ReLU()
        self.fc_cls = nn.Linear(z_dim, num_classes)

        nn.init.normal_(self.fc_z.weight, std=0.01)
        nn.init.zeros_(self.fc_z.bias)
        nn.init.normal_(self.fc_cls.weight, std=0.01)
        nn.init.zeros_(self.fc_cls.bias)

    def _pool_and_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Pool over (T, V) and average over persons → (N, C)."""
        N, M, C, T, V = x.shape
        x = self.pool(x.view(N * M, C, T, V)).view(N, M, C)
        return x.mean(dim=1)  # (N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, M, C, T, V) — backbone output

        Returns:
            (N, num_classes) — raw logits
        """
        x = self._pool_and_mean(x)         # (N, 256)
        z = self.fc_z(self.drop(x))        # (N, z_dim)
        return self.fc_cls(self.relu(z))   # (N, num_classes)

    def get_z(self, x: torch.Tensor) -> torch.Tensor:
        """Extract z_age embedding without running the classification layer.

        Args:
            x: (N, M, C, T, V) — backbone output

        Returns:
            (N, z_dim) — z_age bottleneck embeddings
        """
        x = self._pool_and_mean(x)   # (N, 256)
        return self.fc_z(self.drop(x))  # (N, z_dim)


# ============================================================================
# Convenience wrapper (backbone + head in one nn.Module)
# ============================================================================

class STGCNpp(nn.Module):
    """Full ST-GCN++ model: backbone + classification head.

    This thin wrapper exists mainly for clean checkpoint loading via
    ``load_state_dict``.  For fine-tuning it is easy to replace just one of
    the two sub-modules.

    Args:
        in_channels: Coordinate channels (3 for NTU 3-D skeleton).
        num_classes: Action classes (120 for NTU-120).
        graph:       Optional pre-built NTUGraph.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 120,
        graph: NTUGraph | None = None,
    ) -> None:
        super().__init__()
        self.backbone = STGCNBackbone(in_channels=in_channels, graph=graph)
        self.head = GCNClassifier(in_channels=256, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, M, T, V, C)

        Returns:
            (N, num_classes)
        """
        features = self.backbone(x)
        return self.head(features)
