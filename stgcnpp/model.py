"""Clean STGCN++ implementation — no OpenMMLab / mmcv dependencies.

Architecture matches the pyskl checkpoint trained with:
  gcn_adaptive='init', gcn_with_res=True, tcn_type='mstcn',
  graph layout='nturgb+d' mode='spatial', num_classes=120.
"""
import copy
import math
import torch
import torch.nn as nn

from graph import get_spatial_graph

EPS = 1e-4


# ---------------------------------------------------------------------------
# Temporal convolution blocks
# ---------------------------------------------------------------------------

class UnitTCN(nn.Module):
    """Single-branch temporal convolution (kernel along time axis)."""

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1,
                 dilation=1, dropout=0., norm=True):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.drop = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        return self.drop(self.bn(self.conv(x)))


class MSTCN(nn.Module):
    """Multi-scale temporal convolution with 6 branches."""

    _DEFAULT_CFG = [(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1']

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.,
                 ms_cfg=None):
        super().__init__()
        if ms_cfg is None:
            ms_cfg = self._DEFAULT_CFG
        self.act = nn.ReLU()
        num_branches = len(ms_cfg)
        mid_ch = out_channels // num_branches
        rem_ch = out_channels - mid_ch * (num_branches - 1)

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_ch if i == 0 else mid_ch
            if cfg == '1x1':
                branches.append(
                    nn.Conv2d(in_channels, branch_c, 1, stride=(stride, 1))
                )
            elif cfg[0] == 'max':
                branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, branch_c, 1),
                    nn.BatchNorm2d(branch_c),
                    self.act,
                    nn.MaxPool2d((cfg[1], 1), stride=(stride, 1), padding=(1, 0)),
                ))
            else:
                k, d = cfg
                branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, branch_c, 1),
                    nn.BatchNorm2d(branch_c),
                    self.act,
                    UnitTCN(branch_c, branch_c, kernel_size=k, stride=stride,
                            dilation=d, norm=False),
                ))
        self.branches = nn.ModuleList(branches)

        tin = mid_ch * (num_branches - 1) + rem_ch
        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin), self.act,
            nn.Conv2d(tin, out_channels, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def forward(self, x):
        feat = torch.cat([b(x) for b in self.branches], dim=1)
        feat = self.transform(feat)
        return self.drop(self.bn(feat))


# ---------------------------------------------------------------------------
# Graph convolution block
# ---------------------------------------------------------------------------

class UnitGCN(nn.Module):
    """Adaptive graph convolution (pre-multiply variant).

    adaptive='init'  → A is a learnable Parameter initialised from the
                        pre-computed spatial graph (no offset/importance mask).
    with_res=True    → residual shortcut with optional projection.
    conv_pos='pre'   → conv → reshape → einsum (default STGCN++ setting).
    """

    def __init__(self, in_channels, out_channels, A,
                 adaptive='init', with_res=False):
        super().__init__()
        assert adaptive in (None, 'init', 'offset', 'importance')
        self.adaptive = adaptive
        self.with_res = with_res
        self.num_subsets = A.size(0)

        # Graph adjacency
        if adaptive == 'init':
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer('A', A)

        if adaptive == 'offset':
            self.PA = nn.Parameter(torch.zeros_like(A))
            nn.init.uniform_(self.PA, -1e-6, 1e-6)
        elif adaptive == 'importance':
            self.PA = nn.Parameter(torch.ones_like(A))

        # Pre-multiply conv: projects to K * out_channels then splits
        self.conv = nn.Conv2d(in_channels, out_channels * self.num_subsets, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

        # Residual
        if with_res:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.down = nn.Identity()

    def forward(self, x, A=None):
        n, c, t, v = x.shape
        res = self.down(x) if self.with_res else 0

        # Resolve effective adjacency
        if self.adaptive == 'offset':
            A_eff = self.A + self.PA
        elif self.adaptive == 'importance':
            A_eff = self.A * self.PA
        else:
            A_eff = self.A  # 'init' or None

        x = self.conv(x)                                      # (N, K*C', T, V)
        x = x.view(n, self.num_subsets, -1, t, v)            # (N, K, C', T, V)
        x = torch.einsum('nkctv,kvw->nctw', x, A_eff).contiguous()  # (N, C', T, V)
        return self.act(self.bn(x) + res)


# ---------------------------------------------------------------------------
# STGCN++ block
# ---------------------------------------------------------------------------

class STGCNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True,
                 gcn_adaptive='init', gcn_with_res=False, tcn_type='mstcn',
                 tcn_dropout=0.):
        super().__init__()
        self.gcn = UnitGCN(in_channels, out_channels, A,
                           adaptive=gcn_adaptive, with_res=gcn_with_res)

        if tcn_type == 'unit_tcn':
            self.tcn = UnitTCN(out_channels, out_channels, 9, stride=stride,
                               dropout=tcn_dropout)
        else:  # mstcn
            self.tcn = MSTCN(out_channels, out_channels, stride=stride,
                             dropout=tcn_dropout)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = UnitTCN(in_channels, out_channels,
                                    kernel_size=1, stride=stride, norm=True)

    def forward(self, x):
        res = self.residual(x)
        return self.relu(self.tcn(self.gcn(x)) + res)


# ---------------------------------------------------------------------------
# STGCN++ backbone
# ---------------------------------------------------------------------------

class STGCN(nn.Module):
    """STGCN++ backbone.

    Default hyper-parameters reproduce the NTU RGB+D 120 joint checkpoint.
    Input shape: (N, M, T, V, C)  — batch, persons, frames, joints, channels.
    Output shape: (N, M, 256, T', V)
    """

    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 ch_ratio=2,
                 num_stages=10,
                 inflate_stages=(5, 8),
                 down_stages=(5, 8),
                 data_bn_type='VC',
                 num_person=2,
                 gcn_adaptive='init',
                 gcn_with_res=True,
                 tcn_type='mstcn'):
        super().__init__()
        A = torch.tensor(get_spatial_graph('nturgb+d'), dtype=torch.float32)
        V = A.size(1)

        self.data_bn_type = data_bn_type
        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * V)
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * V)
        else:
            self.data_bn = nn.Identity()

        block_kwargs = dict(gcn_adaptive=gcn_adaptive, gcn_with_res=gcn_with_res,
                            tcn_type=tcn_type)

        modules = []
        # Stage 1: channel expansion from in_channels to base_channels
        if in_channels != base_channels:
            modules.append(STGCNBlock(
                in_channels, base_channels, A.clone(), stride=1,
                residual=False, **block_kwargs
            ))

        inflate_times = 0
        ch = base_channels
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_ch = ch
            if i in inflate_stages:
                inflate_times += 1
            out_ch = int(base_channels * ch_ratio ** inflate_times + EPS)
            ch = out_ch
            modules.append(STGCNBlock(in_ch, out_ch, A.clone(), stride=stride,
                                      **block_kwargs))

        if in_channels == base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)

    def forward(self, x):
        N, M, T, V, C = x.shape
        x = x.permute(0, 1, 3, 4, 2).contiguous()   # (N, M, V, C, T)
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for block in self.gcn:
            x = block(x)

        return x.reshape(N, M, *x.shape[1:])  # (N, M, C', T', V)


# ---------------------------------------------------------------------------
# Classification head
# ---------------------------------------------------------------------------

class GCNHead(nn.Module):
    """Global average pool over (T, V) then linear classification."""

    def __init__(self, num_classes, in_channels, dropout=0.):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_channels, num_classes)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # x: (N, M, C, T, V)
        N, M, C, T, V = x.shape
        x = x.reshape(N * M, C, T, V)
        x = self.pool(x).reshape(N, M, C).mean(dim=1)   # (N, C)
        x = self.drop(x)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class STGCNPP(nn.Module):
    """STGCN++ model: backbone + classification head.

    Args:
        num_classes: number of action classes.
        pretrained: path to a pyskl/mmaction2 checkpoint (.pth).
                    Pass None to start from scratch.
    """

    def __init__(self, num_classes: int = 120, pretrained: str | None = None):
        super().__init__()
        self.backbone = STGCN(
            in_channels=3,
            base_channels=64,
            ch_ratio=2,
            num_stages=10,
            inflate_stages=(5, 8),
            down_stages=(5, 8),
            data_bn_type='VC',
            gcn_adaptive='init',
            gcn_with_res=True,
            tcn_type='mstcn',
        )
        self.cls_head = GCNHead(num_classes=num_classes, in_channels=256, dropout=0.)

        if pretrained is not None:
            self.load_pyskl_checkpoint(pretrained)

    def forward(self, x):
        """x: (N, M, T, V, C) skeleton tensor."""
        feat = self.backbone(x)
        return self.cls_head(feat)

    def load_pyskl_checkpoint(self, path: str):
        """Load a pyskl / mmaction2 RecognizerGCN checkpoint.

        The checkpoint state_dict uses 'backbone.*' and 'cls_head.*' prefixes.
        The head weights are loaded but can be discarded for fine-tuning
        (set strict=False or replace cls_head after loading).
        """
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        # Support both raw state_dict and mmcv-style {'state_dict': ...}
        state = ckpt.get('state_dict', ckpt)

        # Remap 'cls_head.fc_cls.*' → 'cls_head.fc.*'
        remapped = {}
        for k, v in state.items():
            new_k = k.replace('cls_head.fc_cls.', 'cls_head.fc.')
            remapped[new_k] = v

        missing, unexpected = self.load_state_dict(remapped, strict=False)
        if missing:
            print(f'[STGCNPP] Missing keys ({len(missing)}): {missing[:5]} ...')
        if unexpected:
            print(f'[STGCNPP] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...')
        print(f'[STGCNPP] Loaded checkpoint from {path}')

    def freeze_backbone(self):
        """Freeze backbone for head-only fine-tuning."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def reset_head(self, num_classes: int):
        """Replace classification head for fine-tuning on a new dataset."""
        in_ch = self.cls_head.fc.in_features
        self.cls_head = GCNHead(num_classes=num_classes, in_channels=in_ch, dropout=0.)
