# -*- coding: utf-8 -*-
"""
Feature pyramid neck.

The neck is a plain top-down FPN. Its two paper contributions live here:

* :class:`~models.modules.MSABlock` at P5, and
* :class:`~models.modules.CMF` in place of the concatenation at P4 and P3.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import ELANBlock, CMF, MSABlock


class FPNNeck(nn.Module):
    """Top-down feature pyramid with MSA at P5 and CMF at the fusion nodes.

    Channel flow for ``in_channels=[128,128,256]``, ``out_channels=[64,128,256]``:

    ======  ==================================  ==============================
    Level   Operation                          Output
    ======  ==================================  ==============================
    P5      ``MSABlock(C5)``                   ``[256, 20, 20]``
    P4      ``CMF(C4, P5 up) -> ELANBlock``      ``[128, 40, 40]``
    P3      ``CMF(C3, P4 up) -> ELANBlock``      ``[64,  80, 80]``
    ======  ==================================  ==============================

    Args:
        in_channels: channels of the backbone outputs C3, C4, C5.
        out_channels: output channels of P3, P4, P5.
        use_msa: attach :class:`MSABlock` at P5. Disabled for the baseline
            configuration of the ablation study.
        use_cmf: use :class:`CMF` for the P4 and P3 fusion nodes. When False the
            nodes fall back to plain concatenation (baseline configuration).
    """

    def __init__(self, in_channels, out_channels, use_msa: bool = True,
                 use_cmf: bool = True):
        super().__init__()
        c3i, c4i, c5i = in_channels
        c3o, c4o, c5o = out_channels

        # ── fusion nodes ──
        if use_cmf:
            self.fuse_p4 = CMF(c4i, c5o)
            self.fuse_p3 = CMF(c3i, c4o)
            fuse_p4_ch, fuse_p3_ch = self.fuse_p4.c_out, self.fuse_p3.c_out
        else:
            self.fuse_p4 = self.fuse_p3 = None
            fuse_p4_ch, fuse_p3_ch = c4i + c5o, c3i + c4o

        # ── processing blocks ──
        self.p5_block = MSABlock(c5i, c5o) if use_msa else ELANBlock(c5i, c5o)
        self.p4_block = ELANBlock(fuse_p4_ch, c4o)
        self.p3_block = ELANBlock(fuse_p3_ch, c3o)

    @staticmethod
    def _fuse(fusion, x, y):
        """Apply CMF, or concatenate when the fusion node is disabled."""
        if fusion is None:
            return torch.cat([x, y], dim=1)
        return fusion(x, y)

    def forward(self, feats):
        c3, c4, c5 = feats

        p5 = self.p5_block(c5)
        p5_up = F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p4 = self.p4_block(self._fuse(self.fuse_p4, c4, p5_up))

        p4_up = F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p3 = self.p3_block(self._fuse(self.fuse_p3, c3, p4_up))

        return p3, p4, p5