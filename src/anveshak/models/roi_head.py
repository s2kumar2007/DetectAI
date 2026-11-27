"""RoI head: RoIAlign -> 2 FC -> L2-normalised embedding -> cosine classifier over 64 pre-allocated prototype slots
(inactive slots masked to -inf) + class-agnostic box regression + RoI objectness / IoU-quality head.

Also holds the unknown-mining loss (soft unknown targets + known/unknown margin) and the optional PROB-style running
Mahalanobis objectness ablation (OFF by default).
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import MultiScaleRoIAlign, box_iou

from .box_ops import clip_boxes_to, decode_boxes, encode_boxes


class CosineClassifier(nn.Module):
    """Cosine-similarity classifier with learnable temperature and an ``active`` slot mask."""

    def __init__(self, num_slots: int, dim: int, init_scale: float = 10.0, max_scale: float = 50.0):
        super().__init__()
        self.num_slots, self.dim, self.max_scale = num_slots, dim, max_scale
        self.prototypes = nn.Parameter(F.normalize(torch.randn(num_slots, dim), dim=1))
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale)))
        self.register_buffer("active", torch.zeros(num_slots, dtype=torch.bool))

    @property
    def scale(self) -> Tensor:
        return self.log_scale.exp().clamp(max=self.max_scale)

    def activate(self, slots: Sequence[int]) -> None:
        for s in slots:
            if not 0 <= int(s) < self.num_slots:
                raise ValueError(f"slot {s} outside [0,{self.num_slots})")
            self.active[int(s)] = True

    def deactivate(self, slots: Sequence[int]) -> None:
        for s in slots:
            self.active[int(s)] = False

    @torch.no_grad()
    def set_prototype(self, slot: int, vec: Tensor) -> None:
        self.prototypes[slot] = F.normalize(vec.float(), dim=0).to(self.prototypes.dtype)

    def cosines(self, emb: Tensor) -> Tensor:
        """Raw cosine similarity to ALL slots, (N, num_slots)."""
        return F.normalize(emb.float(), dim=1) @ F.normalize(self.prototypes.float(), dim=1).t()

    def forward(self, emb: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (logits with inactive slots = -inf, raw cosines)."""
        cos = self.cosines(emb)
        logits = (self.scale * cos).masked_fill(~self.active[None, :], float("-inf"))
        return logits, cos


class RunningMahalanobis(nn.Module):
    """PROB-style running Gaussian over positive-RoI embeddings; objectness = exp(-d^2 / (2D)). Ablation only."""

    def __init__(self, dim: int, momentum: float = 0.01):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("cov", torch.eye(dim))
        self.register_buffer("seen", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, emb: Tensor) -> None:
        emb = emb.detach().float()
        if emb.shape[0] < 2:
            return
        m = emb.mean(0)
        c = ((emb - m).t() @ (emb - m)) / (emb.shape[0] - 1)
        a = 1.0 if int(self.seen) == 0 else self.momentum
        self.mean.mul_(1 - a).add_(a * m)
        self.cov.mul_(1 - a).add_(a * c)
        self.seen += 1

    @torch.no_grad()
    def objectness(self, emb: Tensor) -> Tensor:
        d = emb.shape[1]
        inv = torch.linalg.pinv(self.cov + 1e-3 * torch.eye(d, device=emb.device))
        diff = emb.float() - self.mean
        d2 = (diff @ inv * diff).sum(1)
        return torch.exp(-0.5 * d2 / d)


class RoIHead(nn.Module):
    def __init__(self, in_ch: int, cfg, mining, num_pooled_levels: int, use_maha: bool = False, maha_momentum: float = 0.01):
        super().__init__()
        self.cfg, self.mining = cfg, mining
        self.names = [str(i) for i in range(num_pooled_levels)]
        self.pool = MultiScaleRoIAlign(self.names, output_size=int(cfg.roi_size), sampling_ratio=2)
        flat = in_ch * int(cfg.roi_size) ** 2
        self.fc1 = nn.Linear(flat, int(cfg.fc_dim))
        self.fc2 = nn.Linear(int(cfg.fc_dim), int(cfg.fc_dim))
        self.embed = nn.Linear(int(cfg.fc_dim), int(cfg.embed_dim))
        self.box = nn.Linear(int(cfg.fc_dim), 4)
        self.obj = nn.Linear(int(cfg.fc_dim), 1)
        self.cls = CosineClassifier(int(cfg.num_slots), int(cfg.embed_dim), float(cfg.init_scale), float(cfg.max_scale))
        self.register_buffer("class_weights", torch.ones(int(cfg.num_slots), dtype=torch.float32), persistent=False)
        self.maha = RunningMahalanobis(int(cfg.embed_dim), maha_momentum) if use_maha else None
        for m in (self.fc1, self.fc2, self.embed):
            nn.init.kaiming_uniform_(m.weight, a=1)
            nn.init.zeros_(m.bias)
        nn.init.normal_(self.box.weight, std=0.001)
        nn.init.zeros_(self.box.bias)
        nn.init.normal_(self.obj.weight, std=0.01)
        nn.init.zeros_(self.obj.bias)

    # ------------------------------------------------------------------ shared pieces
    def trunk(self, feats: Sequence[Tensor], rois: Sequence[Tensor], image_shapes: Sequence[Tuple[int, int]]) -> Tensor:
        pooled_in = OrderedDict((n, f) for n, f in zip(self.names, feats))
        x = self.pool(pooled_in, [r.float() for r in rois], list(image_shapes))
        x = F.relu(self.fc1(x.flatten(1)))
        return F.relu(self.fc2(x))

    def outputs(self, h: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        emb = F.normalize(self.embed(h).float(), dim=1)
        return emb, self.box(h).float(), self.obj(h).float().squeeze(-1)

    @torch.no_grad()
    def embed_boxes(self, feats, boxes: Sequence[Tensor], image_shapes) -> Tensor:
        h = self.trunk(feats, boxes, image_shapes)
        return F.normalize(self.embed(h).float(), dim=1)

    @torch.no_grad()
    def logits_for(self, feats, rois: Sequence[Tensor], image_shapes) -> Tensor:
        """Masked cosine logits for given RoIs (used for logit distillation by the frozen teacher)."""
        emb, _, _ = self.outputs(self.trunk(feats, rois, image_shapes))
        return self.cls(emb)[0]

    # ------------------------------------------------------------------ training
    def _sample(self, proposals, scores, targets):
        c = self.cfg
        bs, pos_frac = int(c.batch_size_per_image), float(c.pos_fraction)
        rois, labels, tgt_boxes, iou_t, is_pos, img_idx = [], [], [], [], [], []
        for i, (props, sc, tgt) in enumerate(zip(proposals, scores, targets)):
            dev = props.device
            gt, gl = tgt["boxes"].float(), tgt["labels"].long()
            if gt.shape[0]:
                props = torch.cat([props, gt])
                sc = torch.cat([sc, torch.ones(gt.shape[0], device=dev)])
                max_iou, arg = box_iou(props, gt).max(1)
            else:
                max_iou = torch.zeros(props.shape[0], device=dev)
                arg = torch.zeros(props.shape[0], dtype=torch.long, device=dev)
            pos = (max_iou >= float(c.pos_iou)).nonzero().flatten()
            neg = (max_iou < float(c.neg_iou)).nonzero().flatten()
            n_pos = min(pos.numel(), int(bs * pos_frac))
            pos_sel = pos[torch.randperm(pos.numel(), device=dev)[:n_pos]]
            n_neg = min(neg.numel(), bs - n_pos)
            order = neg[sc[neg].argsort(descending=True)]  # half hard (high RPN score), half random
            n_hard = n_neg // 2
            rest = order[n_hard:]
            neg_sel = torch.cat([order[:n_hard], rest[torch.randperm(rest.numel(), device=dev)[: n_neg - n_hard]]])
            sel = torch.cat([pos_sel, neg_sel])
            lab = torch.full((sel.numel(),), -1, dtype=torch.long, device=dev)
            if gt.shape[0] and n_pos:
                lab[:n_pos] = gl[arg[pos_sel]]
            rois.append(props[sel])
            labels.append(lab)
            tgt_boxes.append(gt[arg[sel]] if gt.shape[0] else torch.zeros(sel.numel(), 4, device=dev))
            iou_t.append(max_iou[sel])
            is_pos.append(torch.arange(sel.numel(), device=dev) < n_pos)
            img_idx.append(torch.full((sel.numel(),), i, dtype=torch.long, device=dev))
        return rois, torch.cat(labels), torch.cat(tgt_boxes), torch.cat(iou_t), torch.cat(is_pos), torch.cat(img_idx)

    def loss(self, feats, proposals, scores, targets, image_shapes, mining_on: bool):
        c, m = self.cfg, self.mining
        rois_l, labels, tgt_boxes, iou_t, is_pos, img_idx = self._sample(proposals, scores, targets)
        rois = torch.cat(rois_l)
        h = self.trunk(feats, rois_l, image_shapes)
        emb, deltas, obj_logit = self.outputs(h)
        logits, cos = self.cls(emb)
        zero = obj_logit.sum() * 0.0
        active = self.cls.active

        # --- classification (positives only; unlabelled / background RoIs never get a known label)
        if is_pos.any():
            loss_cls = F.cross_entropy(logits[is_pos], labels[is_pos], weight=self.class_weights.to(logits.device))
            loss_box = F.smooth_l1_loss(deltas[is_pos], encode_boxes(rois[is_pos], tgt_boxes[is_pos]), beta=1 / 9, reduction="sum") / int(is_pos.sum())
            cos_own = cos[is_pos].gather(1, labels[is_pos].unsqueeze(1)).squeeze(1)
            margin_known = F.relu(float(m.margin_known) - cos_own).mean()
            if self.maha is not None:
                self.maha.update(emb[is_pos])
        else:
            loss_cls, loss_box, margin_known = zero, zero, zero

        # --- unknown mining among negatives: high objectness, no GT overlap, low known similarity
        # negatives are only *probably* background (unlabelled objects live there): down-weight them
        neg_w = float(c.get("roi_neg_weight", c.get("neg_obj_weight", 0.1)))
        neg_w = max(0.0, min(1.0, neg_w))
        obj_w = torch.where(is_pos, torch.ones_like(obj_logit), torch.full_like(obj_logit, neg_w))
        obj_t = torch.where(is_pos, iou_t, torch.zeros_like(iou_t))
        mined = torch.zeros_like(is_pos)
        margin_unknown = zero

        # Negative sampling hygiene: down-weight negatives on regions the model itself scores as high-objectness but match no GT
        if bool(c.get("ignore_high_obj_negatives", True)):
            with torch.no_grad():
                obj_p = torch.sigmoid(obj_logit)
                thr = float(c.get("high_obj_thresh", 0.3))
                if not mining_on:
                    thr = max(thr, 0.6)
                if bool(active.any()):
                    max_cos = cos[:, active].max(1).values
                    high_obj = (~is_pos) & (obj_p >= thr) & (max_cos <= float(m.get("cos_max", 0.8)))
                else:
                    high_obj = (~is_pos) & (obj_p >= thr)
                obj_w = torch.where(high_obj, torch.zeros_like(obj_w), obj_w)

        if bool(m.enable) and mining_on and bool(active.any()):
            with torch.no_grad():
                obj_p = torch.sigmoid(obj_logit)
                max_cos = cos[:, active].max(1).values
                cand = (~is_pos) & (obj_p >= float(m.obj_min)) & (max_cos <= float(m.cos_max))
                for i in range(len(rois_l)):
                    idx = ((img_idx == i) & cand).nonzero().flatten()
                    if idx.numel():
                        top = idx[obj_p[idx].argsort(descending=True)[: int(m.topk)]]
                        mined[top] = True
                if mined.any():  # ignore negatives that overlap a mined box (they are the same unlabelled object)
                    for i in range(len(rois_l)):
                        mi = (mined & (img_idx == i)).nonzero().flatten()
                        if mi.numel():
                            ni = ((img_idx == i) & ~is_pos & ~mined).nonzero().flatten()
                            if ni.numel():
                                near = box_iou(rois[ni], rois[mi]).max(1).values >= float(m.suppress_iou)
                                obj_w[ni[near]] = 0.0
                    obj_t = torch.where(mined, torch.full_like(obj_t, float(m.soft_target)), obj_t)
                    obj_w = torch.where(mined, torch.full_like(obj_w, float(m.get("weight", 1.0))), obj_w)
            if mined.any():
                margin_unknown = F.relu(cos[mined][:, active].max(1).values - float(m.margin_unknown)).mean()
        loss_obj = F.binary_cross_entropy_with_logits(obj_logit, obj_t, weight=obj_w, reduction="sum") / obj_w.sum().clamp(min=1.0)

        losses = {
            "roi_cls": float(c.cls_weight) * loss_cls,
            "roi_box": float(c.box_weight) * loss_box,
            "roi_obj": float(c.obj_weight) * loss_obj,
            "roi_margin": float(m.weight) * (margin_known + margin_unknown),
        }
        aux = {
            "rois": rois_l, "logits": logits, "labels": labels, "is_pos": is_pos, "mined": mined,
            "obj_t": obj_t, "obj_w": obj_w, "obj_logit": obj_logit,
            "n_pos": int(is_pos.sum()), "n_mined": int(mined.sum()),
        }
        return losses, aux

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict(self, feats, proposals: Sequence[Tuple[Tensor, Tensor]], image_sizes, image_shapes) -> List[Dict[str, Tensor]]:
        rois = [p[0] for p in proposals]
        counts = [r.shape[0] for r in rois]
        if sum(counts) == 0:
            return [self._empty(r.device) for r in rois]
        h = self.trunk(feats, rois, image_shapes)
        emb, deltas, obj_logit = self.outputs(h)
        logits, cos = self.cls(emb)
        probs = torch.softmax(logits, dim=1) if bool(self.cls.active.any()) else torch.zeros_like(logits)
        max_cos = cos[:, self.cls.active].max(1).values if bool(self.cls.active.any()) else torch.zeros(cos.shape[0], device=cos.device)
        obj = torch.sigmoid(obj_logit)
        if self.maha is not None:
            obj = torch.sqrt(obj * self.maha.objectness(emb))
        boxes_all = decode_boxes(deltas, torch.cat(rois))
        out = []
        start = 0
        for n, size in zip(counts, image_sizes):
            sl = slice(start, start + n)
            out.append({
                "boxes": clip_boxes_to(boxes_all[sl], size), "obj": obj[sl], "probs": probs[sl],
                "cos_max": max_cos[sl], "rpn_scores": proposals[len(out)][1],
            })
            start += n
        return out

    def _empty(self, device) -> Dict[str, Tensor]:
        S = int(self.cfg.num_slots)
        return {"boxes": torch.zeros(0, 4, device=device), "obj": torch.zeros(0, device=device),
                "probs": torch.zeros(0, S, device=device), "cos_max": torch.zeros(0, device=device),
                "rpn_scores": torch.zeros(0, device=device)}
