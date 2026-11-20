"""ResNet backbone with GroupNorm, RANDOM INIT ONLY (weights=None). No code path here can load pretrained weights."""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet


def valid_groups(channels: int, groups: int) -> int:
    """Largest g <= groups that divides ``channels``."""
    g = min(groups, channels)
    while channels % g:
        g -= 1
    return g


def gn(channels: int, groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(valid_groups(channels, groups), channels)


def gn_factory(groups: int) -> Callable[[int], nn.Module]:
    return lambda c: nn.GroupNorm(valid_groups(c, groups), c)


class ResNetBody(nn.Module):
    """Stem + 4 stages; returns the C2..C5 feature maps."""

    def __init__(self, net: ResNet, out_channels: Sequence[int]):
        super().__init__()
        self.conv1, self.bn1, self.relu, self.maxpool = net.conv1, net.bn1, net.relu, net.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.out_channels = list(out_channels)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]

    def stage_modules(self) -> List[List[nn.Module]]:
        """[[stem], [layer1], [layer2], [layer3], [layer4]] for freezing."""
        return [[self.conv1, self.bn1], [self.layer1], [self.layer2], [self.layer3], [self.layer4]]

    def freeze_stages(self, n: int) -> None:
        """n=0 none; 1 stem; 2 stem+layer1; 3 +layer2; 4 +layer3; 5 all."""
        for mods in self.stage_modules()[: max(n, 0)]:
            for m in mods:
                for p in m.parameters():
                    p.requires_grad_(False)


def _init_weights(net: nn.Module) -> None:
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    for m in net.modules():  # zero-init last norm of each residual branch (stabilises from-scratch GN training)
        if isinstance(m, Bottleneck):
            nn.init.zeros_(m.bn3.weight)
        elif isinstance(m, BasicBlock):
            nn.init.zeros_(m.bn2.weight)


def build_backbone(name: str = "resnet50", layers: Optional[Sequence[int]] = None, groups: int = 32) -> ResNetBody:
    """GroupNorm ResNet, weights=None, kaiming init. ``layers`` overrides the per-stage block counts (smoke tests)."""
    norm = gn_factory(groups)
    if name == "resnet50":
        block, default_layers = Bottleneck, [3, 4, 6, 3]
    elif name == "resnet18":
        block, default_layers = BasicBlock, [2, 2, 2, 2]
    else:
        raise ValueError(f"unsupported backbone {name!r} (resnet50 | resnet18)")
    if layers is None:
        factory = torchvision.models.resnet50 if name == "resnet50" else torchvision.models.resnet18
        net = factory(weights=None, norm_layer=norm)  # weights=None: NEVER pretrained
    else:
        net = ResNet(block, list(layers), norm_layer=norm)
    _init_weights(net)
    exp = block.expansion
    return ResNetBody(net, [64 * exp, 128 * exp, 256 * exp, 512 * exp])
