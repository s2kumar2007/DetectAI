"""The full open-world detector: GN-ResNet -> FPN(P2-P6) -> OLN RPN -> cosine RoI head (64 masked slots)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .backbone import build_backbone
from .fpn import FPN
from .roi_head import RoIHead
from .rpn import OLNRPN

# Conventional normalisation constants (plain numbers - not learned / pretrained weights).
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

Targets = Sequence[Dict[str, Tensor]]


def arch_signature(cfg) -> dict:
    """The parts of the config that define tensor shapes: checkpoints must match on these across stages."""
    m = cfg.model
    return {
        "backbone": m.backbone.name, "layers": list(m.backbone.layers) if m.backbone.layers else None,
        "groups": int(m.backbone.groups), "fpn_out": int(m.fpn.out_channels), "use_p2": bool(m.fpn.use_p2),
        "fc_dim": int(m.roi_head.fc_dim), "embed_dim": int(m.roi_head.embed_dim), "num_slots": int(m.roi_head.num_slots),
        "roi_size": int(m.roi_head.roi_size), "rpn_convs": int(m.rpn.num_convs),
    }


class OWODDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg.model
        self.cfg = cfg
        self.size_divisible = int(cfg.input.size_divisible)
        self.backbone = build_backbone(m.backbone.name, m.backbone.layers, int(m.backbone.groups))
        self.fpn = FPN(self.backbone.out_channels, int(m.fpn.out_channels), bool(m.fpn.use_p2), int(m.backbone.groups))
        self.rpn = OLNRPN(self.fpn.out_channels, self.fpn.strides, m.rpn, int(m.backbone.groups))
        self.roi_head = RoIHead(
            self.fpn.out_channels, m.roi_head, m.mining, num_pooled_levels=self.fpn.num_levels - 1,
            use_maha=bool(m.ablation.prob_mahalanobis), maha_momentum=float(m.ablation.maha_momentum),
        )
        self.register_buffer("pixel_mean", torch.tensor(_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(_STD).view(1, 3, 1, 1), persistent=False)
        self.mining_on = False  # toggled by the trainer after ``mining.start_frac`` of the schedule

    # ------------------------------------------------------------------ slots
    @property
    def num_slots(self) -> int:
        return self.roi_head.cls.num_slots

    @property
    def active_slots(self) -> List[int]:
        return self.roi_head.cls.active.nonzero().flatten().tolist()

    def activate_slots(self, slots: Sequence[int]) -> None:
        self.roi_head.cls.activate(slots)

    # ------------------------------------------------------------------ pre-processing
    def preprocess(self, images: Sequence[Tensor]) -> Tuple[Tensor, List[Tuple[int, int]]]:
        """uint8 (or 0-1 float) CHW images -> normalised, bottom/right zero-padded batch + true (h, w) sizes."""
        dev = self.pixel_mean.device
        sizes = [(int(im.shape[-2]), int(im.shape[-1])) for im in images]
        d = self.size_divisible
        H = -(-max(s[0] for s in sizes) // d) * d
        W = -(-max(s[1] for s in sizes) // d) * d
        batch = torch.zeros(len(images), 3, H, W, device=dev)
        for i, im in enumerate(images):
            x = im.to(dev)
            x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
            batch[i, :, : x.shape[1], : x.shape[2]] = (x[None] - self.pixel_mean) / self.pixel_std
        return batch, sizes

    def extract(self, x: Tensor) -> List[Tensor]:
        return self.fpn(self.backbone(x))

    # ------------------------------------------------------------------ training
    def forward_train_x(self, x: Tensor, sizes, targets: Targets):
        feats = self.extract(x)
        props, rpn_losses = self.rpn(feats, sizes, targets)
        shapes = [tuple(x.shape[-2:])] * x.shape[0]
        roi_losses, aux = self.roi_head.loss(
            feats, [p[0] for p in props], [p[1] for p in props], targets, shapes, self.mining_on
        )
        aux["feats"], aux["image_shapes"], aux["proposals"] = feats, shapes, props
        return {**rpn_losses, **roi_losses}, aux

    def forward_train(self, images: Sequence[Tensor], targets: Targets):
        x, sizes = self.preprocess(images)
        dev = x.device
        targets = [{"boxes": t["boxes"].to(dev), "labels": t["labels"].to(dev)} for t in targets]
        return self.forward_train_x(x, sizes, targets)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_raw_x(self, x: Tensor, sizes) -> List[Dict[str, Tensor]]:
        feats = self.extract(x)
        props, _ = self.rpn(feats, sizes)
        shapes = [tuple(x.shape[-2:])] * x.shape[0]
        return self.roi_head.predict(feats, props, sizes, shapes)

    @torch.no_grad()
    def predict_raw(self, images: Sequence[Tensor]) -> List[Dict[str, Tensor]]:
        """Per image: boxes (refined), obj (S_obj), probs (N,num_slots), cos_max, rpn_scores. Call in eval mode."""
        x, sizes = self.preprocess(images)
        return self.predict_raw_x(x, sizes)

    @torch.no_grad()
    def embed_gt_boxes(self, images: Sequence[Tensor], boxes: Sequence[Tensor]) -> Tensor:
        x, sizes = self.preprocess(images)
        feats = self.extract(x)
        shapes = [tuple(x.shape[-2:])] * x.shape[0]
        return self.roi_head.embed_boxes(feats, [b.to(x.device).float() for b in boxes], shapes)


def build_model(cfg) -> OWODDetector:
    """Build the detector with RANDOM initialisation (no weights are ever loaded here)."""
    return OWODDetector(cfg)
