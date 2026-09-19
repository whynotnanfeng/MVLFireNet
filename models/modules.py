# -*- coding: utf-8 -*-
"""
Building blocks for MVLFireNet.

Three modules correspond directly to the contributions of the paper:

* :class:`MSAAttention` -- Multi-Scale Spatial-Aware attention (MSA)
* :class:`CMF`          -- Cross-Modulation Fusion (CMF)
* :class:`MVLEBranch`   -- Multi-Granularity Vision-Language Enhancement (MVLE),
                           implemented in :mod:`models.mvle`

The remaining blocks (ELAN-style ``MMBlock``, ``SPPF``, ``MGFFN``) are standard
detector components used by the backbone and the RT-DETR decoder head.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def autopad(k, p=None, d=1):
    """Pad to 'same' shape."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Convolution + BatchNorm + activation."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# ELAN-style backbone blocks
# ---------------------------------------------------------------------------

class MMBasic(nn.Module):
    """Residual pair of 3x3 convolutions."""

    def __init__(self, ch_in, ch_out, e=0.5, k=3):
        super().__init__()
        c = int(ch_out * e)
        self.cv1 = Conv(ch_in, c, k, 1)
        self.cv2 = Conv(c, ch_out, k, 1)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class MMEnhance(nn.Module):
    """Gated variant of :class:`MMBasic` with a two-layer inner block."""

    def __init__(self, ch_in, ch_out, e=0.5, k=3):
        super().__init__()
        self.c = int(ch_out * e)
        self.conv1 = Conv(ch_in, self.c, k=1)
        self.conv2 = Conv(ch_in, self.c, k=1)
        self.conv3 = Conv(self.c * 2, ch_out, k=1)
        self.b1 = MMBasic(self.c, self.c, k=k)
        self.b2 = MMBasic(self.c, self.c, k=k)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.b1(self.b2(self.conv2(x)))
        return self.conv3(torch.cat((x1, x2), dim=1))


class MMBlock(nn.Module):
    """ELAN aggregation block: split, transform one branch, concatenate."""

    def __init__(self, ch_in, ch_out, e=0.5, k=3, enhance=False):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.c = int(ch_out * e)
        self.enhance = enhance
        self.conv1 = Conv(self.ch_in, self.c * 2, k=1)
        self.conv2 = Conv(self.c * 3, ch_out, k=1)

        if self.enhance:
            self.en1 = MMEnhance(self.c, self.c, k=k, e=1)
        else:
            self.en1 = MMBasic(self.c, self.c, k=k)

    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = x.split((self.c, self.c), 1)
        x3 = self.en1(x2)
        return self.conv2(torch.cat((x1, x2, x3), 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""

    def __init__(self, c1, c2, k=5, n=3, shortcut=False):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: Tensor) -> Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        y = self.cv2(torch.cat(y, 1))
        return y + x if self.add else y


class MGFFN(nn.Module):
    """Gated feed-forward network used inside the decoder layers."""

    def __init__(self, features, hidden=None, kernel_size=9):
        super().__init__()
        out_features = features
        hidden = int(2 * hidden / 3)
        self.fc1 = nn.Conv2d(features, hidden * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=kernel_size, stride=1,
                                padding=kernel_size // 2, bias=True, groups=hidden)
        self.act = nn.Mish()
        self.fc2 = nn.Conv2d(hidden, out_features, kernel_size=1)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x) + x) * v
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Multi-Scale Spatial-Aware attention (MSA)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head self-attention over flattened spatial tokens.

    The query/key projections are 1x1 convolutions, so this block carries no
    explicit two-dimensional spatial prior. :class:`MSAAttention` adds it back.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5

        self.qkv = Conv(dim, dim + self.key_dim * num_heads * 2, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)


class MSAAttention(nn.Module):
    """Attention block augmented with a multi-scale dilated depthwise branch.

    Self-attention treats a feature map as a flat sequence and therefore discards
    the 2D arrangement of local features. Fire spots and smoke edges survive
    mainly as high-frequency spatial detail, so this branch re-injects it with
    three parallel dilated depthwise convolutions (dilation 1, 3, 5 -> receptive
    fields 3x3, 7x7, 11x11), each scaled by an independent learnable weight.

    The depthwise branch carries no activation: it stays linear so the learned
    weights alone control how much each scale contributes.
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))

        def dw(dilation):
            # Padding equals the dilation so the spatial size is preserved.
            return nn.Sequential(
                nn.Conv2d(c, c, 3, 1, padding=dilation, dilation=dilation,
                          groups=c, bias=False),
                nn.BatchNorm2d(c),
            )

        self.dw3, self.dw7, self.dw11 = dw(1), dw(3), dw(5)
        self.alpha3 = nn.Parameter(torch.tensor(1.0))
        self.alpha7 = nn.Parameter(torch.tensor(1.0))
        self.alpha11 = nn.Parameter(torch.tensor(1.0))
        self.add = shortcut

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x + (self.alpha3 * self.dw3(x)
                    + self.alpha7 * self.dw7(x)
                    + self.alpha11 * self.dw11(x))


class MSABlock(nn.Module):
    """ELAN wrapper around :class:`MSAAttention`, used at neck level P5."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)
        self.m = nn.Sequential(*(
            MSAAttention(self.c, attn_ratio=0.5, num_heads=max(self.c // 32, 2))
            for _ in range(n)
        ))

    def forward(self, x: Tensor) -> Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((a, self.m(b)), 1))


# ---------------------------------------------------------------------------
# Cross-Modulation Fusion (CMF)
# ---------------------------------------------------------------------------

class CMF(nn.Module):
    """Bidirectional conditional affine modulation for pyramid fusion.

    Concatenation treats features from adjacent pyramid levels as interchangeable,
    but they live in different representation spaces: shallow maps carry spatial
    detail mixed with background noise, deep maps carry semantics but have lost
    resolution. CMF compresses both inputs, then modulates each with scale and
    bias factors predicted from the other, so deep semantics actively suppress
    background in the shallow map while shallow detail restores positional
    accuracy in the deep map. The two modulated maps are concatenated for the
    next stage.

    Scale and bias factors are shared across groups of channels, which cuts the
    parameter cost of the fusion and regularizes it. Gamma is initialized to 1
    and beta to 0, so at initialization the module is exactly equivalent to a
    compressed concatenation.

    Args:
        c_x: channels of the higher-level (deep) input.
        c_y: channels of the lower-level (shallow) input.
        ratio: fraction of channels retained per branch after compression.
        grouped: share each gamma/beta across several channels when True.
    """

    def __init__(self, c_x: int, c_y: int, ratio: float = 0.5, c_mid: int = None,
                 grouped: bool = True):
        super().__init__()
        self.c_ox = max(int(c_x * ratio), 8)
        self.c_oy = max(int(c_y * ratio), 8)
        self.c_out = self.c_ox + self.c_oy
        if c_mid is None:
            c_mid = max(min(self.c_ox, self.c_oy) // 4, 8)

        self.ng_y2x = max(self.c_ox // 2, 1) if grouped else self.c_ox
        self.ng_x2y = max(self.c_oy // 2, 1) if grouped else self.c_oy

        # Predictors for the modulation factors.
        self.y_compress = nn.Sequential(
            nn.Conv2d(self.c_oy, c_mid, 1, bias=False), nn.BatchNorm2d(c_mid), nn.SiLU())
        self.y_to_x_gamma = nn.Conv2d(c_mid, self.ng_y2x, 1)
        self.y_to_x_beta = nn.Conv2d(c_mid, self.ng_y2x, 1)

        self.x_compress = nn.Sequential(
            nn.Conv2d(self.c_ox, c_mid, 1, bias=False), nn.BatchNorm2d(c_mid), nn.SiLU())
        self.x_to_y_gamma = nn.Conv2d(c_mid, self.ng_x2y, 1)
        self.x_to_y_beta = nn.Conv2d(c_mid, self.ng_x2y, 1)

        # Input compression.
        self.x_compress_in = nn.Sequential(
            nn.Conv2d(c_x, self.c_ox, 1, bias=False), nn.BatchNorm2d(self.c_ox), nn.SiLU())
        self.y_compress_in = nn.Sequential(
            nn.Conv2d(c_y, self.c_oy, 1, bias=False), nn.BatchNorm2d(self.c_oy), nn.SiLU())

        self._init_identity()

    def _init_identity(self):
        """Start from the identity: gamma = 1, beta = 0."""
        for proj in (self.y_to_x_gamma, self.x_to_y_gamma):
            nn.init.ones_(proj.weight)
            nn.init.zeros_(proj.bias)
        for proj in (self.y_to_x_beta, self.x_to_y_beta):
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    @staticmethod
    def _group_film(comp, cond, compress, gamma_proj, beta_proj, num_groups):
        """Apply grouped affine modulation: (B,G,H,W) is broadcast back to (B,C,H,W)."""
        c = compress(cond)
        g = gamma_proj(c).clamp(-2.0, 4.0)
        b = beta_proj(c).clamp(-5.0, 5.0)
        C = comp.shape[1]
        num_groups = min(num_groups, C)
        reps = C // num_groups
        g = g.repeat(1, reps, 1, 1)[:, :C]
        b = b.repeat(1, reps, 1, 1)[:, :C]
        return g * comp + b

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Fuse a deep map ``x`` and a shallow map ``y``."""
        orig_dtype = x.dtype
        if x.shape[-2:] != y.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode='nearest')

        x_comp = self.x_compress_in(x.float())
        y_comp = self.y_compress_in(y.float())

        x_mod = self._group_film(x_comp, y_comp, self.y_compress,
                                 self.y_to_x_gamma, self.y_to_x_beta, self.ng_y2x)
        y_mod = self._group_film(y_comp, x_comp, self.x_compress,
                                 self.x_to_y_gamma, self.x_to_y_beta, self.ng_x2y)

        return torch.cat([x_mod, y_mod], dim=1).to(orig_dtype)