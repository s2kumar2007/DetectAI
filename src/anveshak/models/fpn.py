"""FPN producing P2-P6 (or P3-P6 when ``use_p2`` is false)."""
from __future__ import annotations

from collections import OrderedDict
from typing import List, Sequence

import torch
import torch.nn as nn
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from .backbone import gn_factory


class FPN(nn.Module):
    def __init__(self, in_channels: Sequence[int], out_channels: int = 256, use_p2: bool = True, groups: int = 32):
        super().__init__()
        self.use_p2 = use_p2
        first = 0 if use_p2 else 1
        self.first = first
        self.fpn = FeaturePyramidNetwork(
            list(in_channels[first:]), out_channels, extra_blocks=LastLevelMaxPool(), norm_layer=gn_factory(groups)
        )
        self.out_channels = out_channels
        self.strides: List[int] = [4 * 2**i for i in range(first, 4)] + [64]

    @property
    def num_levels(self) -> int:
        return len(self.strides)

    def forward(self, cs: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        feats = OrderedDict((str(i), c) for i, c in enumerate(cs[self.first :]))
        return list(self.fpn(feats).values())
