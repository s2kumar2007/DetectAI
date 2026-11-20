"""OLN-style class-agnostic localisation-quality RPN (anchor-free).

For every FPN location we predict ltrb distances, a *centerness* and a *box-IoU quality*; there is no binary fg/bg
classification. Proposal score = sqrt(centerness * iou_quality).

Unlabelled objects must not be trained as background (open world): positives are locations inside a GT box, negatives
are sampled ONLY from locations that lie outside every GT box dilated by ``neg_dilate``; the ambiguous band in between
is ignored.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import generalized_box_iou_loss, nms

from .backbone import gn
from .box_ops import clip_boxes_to, elementwise_iou


class OLNRPN(nn.Module):
    def __init__(self, in_ch: int, strides: Sequence[int], cfg, groups: int = 32):
        super().__init__()
        self.cfg = cfg
        self.strides = list(strides)
        tower: List[nn.Module] = []
        for _ in range(int(cfg.num_convs)):
            tower += [nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False), gn(in_ch, groups), nn.ReLU(inplace=True)]
        self.tower = nn.Sequential(*tower)
        self.reg_conv = nn.Conv2d(in_ch, 4, 3, padding=1)
        self.ctr_conv = nn.Conv2d(in_ch, 1, 3, padding=1)
        self.iou_conv = nn.Conv2d(in_ch, 1, 3, padding=1)
        self.scales = nn.Parameter(torch.ones(len(self.strides)))
        for m in self.tower.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
        for m in (self.reg_conv, self.ctr_conv, self.iou_conv):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)
        hi = [16.0 * s for s in self.strides]
        hi[-1] = math.inf
        lo = [0.0] + hi[:-1]
        self.register_buffer("range_lo", torch.tensor(lo), persistent=False)
        self.register_buffer("range_hi", torch.tensor(hi), persistent=False)

    # ------------------------------------------------------------------ head
    def _head(self, feats: Sequence[Tensor]):
        regs, ctrs, ious, pts, lvls, slices = [], [], [], [], [], []
        start = 0
        for li, (f, s) in enumerate(zip(feats, self.strides)):
            t = self.tower(f)
            reg = torch.exp((self.reg_conv(t).float() * self.scales[li]).clamp(-4.0, 4.0)) * (8.0 * s)
            ctr = self.ctr_conv(t).float()
            iou = self.iou_conv(t).float()
            h, w = f.shape[-2:]
            ys = (torch.arange(h, device=f.device, dtype=torch.float32) + 0.5) * s
            xs = (torch.arange(w, device=f.device, dtype=torch.float32) + 0.5) * s
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            pts.append(torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1))
            lvls.append(torch.full((h * w,), li, device=f.device, dtype=torch.long))
            regs.append(reg.flatten(2).transpose(1, 2))
            ctrs.append(ctr.flatten(1))
            ious.append(iou.flatten(1))
            slices.append((start, start + h * w))
            start += h * w
        reg = torch.cat(regs, dim=1)
        pts_t = torch.cat(pts, dim=0)
        boxes = torch.cat([pts_t[None] - reg[..., :2], pts_t[None] + reg[..., 2:]], dim=-1)
        return boxes, reg, torch.cat(ctrs, dim=1), torch.cat(ious, dim=1), pts_t, torch.cat(lvls), slices

    # ------------------------------------------------------------------ targets
    @torch.no_grad()
    def _assign(self, pts: Tensor, lvl: Tensor, gt: Tensor, n_level0: int):
        """Return (pos mask, assigned gt index, neg mask) for one image."""
        P, G = pts.shape[0], gt.shape[0]
        dev = pts.device
        best_area = torch.full((P,), math.inf, device=dev)
        gt_idx = torch.zeros(P, dtype=torch.long, device=dev)
        in_dil_any = torch.zeros(P, dtype=torch.bool, device=dev)
        lo, hi = self.range_lo[lvl][:, None], self.range_hi[lvl][:, None]
        chunk = int(self.cfg.get("gt_chunk", 64))
        dil = float(self.cfg.neg_dilate)
        for s in range(0, G, chunk):
            g = gt[s : s + chunk]
            px, py = pts[:, 0:1], pts[:, 1:2]
            ltrb = torch.stack([px - g[None, :, 0], py - g[None, :, 1], g[None, :, 2] - px, g[None, :, 3] - py], dim=-1)
            inside = ltrb.amin(-1) > 0
            mr = ltrb.amax(-1)
            cand = inside & (mr >= lo) & (mr < hi)
            area = ((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]))[None].expand(P, -1)
            area = torch.where(cand, area, torch.full_like(area, math.inf))
            a_min, a_arg = area.min(1)
            better = a_min < best_area
            best_area = torch.where(better, a_min, best_area)
            gt_idx = torch.where(better, a_arg + s, gt_idx)
            cx, cy = (g[:, 0] + g[:, 2]) / 2, (g[:, 1] + g[:, 3]) / 2
            hw, hh = (g[:, 2] - g[:, 0]) * dil / 2, (g[:, 3] - g[:, 1]) * dil / 2
            in_dil = ((px - cx[None]).abs() < hw[None]) & ((py - cy[None]).abs() < hh[None])
            in_dil_any |= in_dil.any(1)
        pos = best_area < math.inf
        # every GT gets at least one positive (finest level, nearest to its centre) - needed for tiny objects
        counts = torch.bincount(gt_idx[pos], minlength=G)
        for g_i in (counts == 0).nonzero().flatten().tolist():
            c = torch.stack([(gt[g_i, 0] + gt[g_i, 2]) / 2, (gt[g_i, 1] + gt[g_i, 3]) / 2])
            j = int(((pts[:n_level0] - c[None]) ** 2).sum(1).argmin())
            if not pos[j]:
                pos[j] = True
                gt_idx[j] = g_i
        neg = (~in_dil_any) & (~pos)
        return pos, gt_idx, neg

    def _loss(self, boxes, ctr, iou, pts, lvl, n_level0, targets) -> Dict[str, Tensor]:
        c = self.cfg
        dev = ctr.device
        pb, pt, p_ctr, p_iou, n_ctr, n_iou = [], [], [], [], [], []
        for i, tgt in enumerate(targets):
            gt = tgt["boxes"].float()
            if gt.shape[0]:
                pos, gt_idx, neg = self._assign(pts, lvl, gt, n_level0)
            else:
                pos = torch.zeros(pts.shape[0], dtype=torch.bool, device=dev)
                gt_idx = torch.zeros(pts.shape[0], dtype=torch.long, device=dev)
                neg = torch.ones_like(pos)
            pidx = pos.nonzero().flatten()
            nidx = neg.nonzero().flatten()
            n_neg = min(nidx.numel(), max(int(c.neg_min), int(math.ceil(float(c.neg_per_pos) * pidx.numel()))))
            if n_neg < nidx.numel():
                nidx = nidx[torch.randperm(nidx.numel(), device=dev)[:n_neg]]
            if pidx.numel():
                pb.append(boxes[i, pidx])
                pt.append(torch.cat([pts[pidx], gt[gt_idx[pidx]]], dim=1))  # (x,y,gx1,gy1,gx2,gy2)
                p_ctr.append(ctr[i, pidx])
                p_iou.append(iou[i, pidx])
            n_ctr.append(ctr[i, nidx])
            n_iou.append(iou[i, nidx])
        zero = ctr.sum() * 0.0
        n_ctr_c, n_iou_c = torch.cat(n_ctr), torch.cat(n_iou)
        nw = float(c.neg_weight)
        neg_ctr_loss = F.binary_cross_entropy_with_logits(n_ctr_c, torch.zeros_like(n_ctr_c), reduction="sum") if n_ctr_c.numel() else zero
        neg_iou_loss = F.binary_cross_entropy_with_logits(n_iou_c, torch.zeros_like(n_iou_c), reduction="sum") if n_iou_c.numel() else zero
        n_neg_total = n_ctr_c.numel()
        if pb:
            pred = torch.cat(pb)
            meta = torch.cat(pt)
            tgt_box, xy = meta[:, 2:], meta[:, :2]
            l, t_, r, b = xy[:, 0] - tgt_box[:, 0], xy[:, 1] - tgt_box[:, 1], tgt_box[:, 2] - xy[:, 0], tgt_box[:, 3] - xy[:, 1]
            ctr_t = torch.sqrt((torch.minimum(l, r) / torch.maximum(l, r).clamp(min=1e-6)).clamp(0, 1)
                               * (torch.minimum(t_, b) / torch.maximum(t_, b).clamp(min=1e-6)).clamp(0, 1))
            iou_t = elementwise_iou(pred.detach(), tgt_box)
            giou = generalized_box_iou_loss(pred, tgt_box, reduction="none")
            loss_reg = (giou * ctr_t).sum() / ctr_t.sum().clamp(min=1e-6)
            pc, pi = torch.cat(p_ctr), torch.cat(p_iou)
            pos_ctr = F.binary_cross_entropy_with_logits(pc, ctr_t, reduction="sum")
            pos_iou = F.binary_cross_entropy_with_logits(pi, iou_t, reduction="sum")
            n_pos = pc.numel()
        else:
            loss_reg, pos_ctr, pos_iou, n_pos = zero, zero, zero, 0
        denom = max(n_pos + nw * n_neg_total, 1.0)
        return {
            "rpn_reg": float(c.reg_weight) * loss_reg,
            "rpn_ctr": float(c.ctr_weight) * (pos_ctr + nw * neg_ctr_loss) / denom,
            "rpn_iou": float(c.iou_weight) * (pos_iou + nw * neg_iou_loss) / denom,
        }

    # ------------------------------------------------------------------ proposals
    @torch.no_grad()
    def _propose(self, boxes, ctr, iou, slices, image_sizes, post_n: int) -> List[Tuple[Tensor, Tensor]]:
        scores = (torch.sigmoid(ctr) * torch.sigmoid(iou)).sqrt()
        out = []
        for i in range(boxes.shape[0]):
            bl, sl = [], []
            for a, b in slices:
                s = scores[i, a:b]
                k = min(int(self.cfg.pre_nms_top_n), s.numel())
                top, idx = s.topk(k)
                bl.append(boxes[i, a:b][idx])
                sl.append(top)
            bx = clip_boxes_to(torch.cat(bl), image_sizes[i])
            sc = torch.cat(sl)
            ms = float(self.cfg.min_box_size)
            ok = ((bx[:, 2] - bx[:, 0]) >= ms) & ((bx[:, 3] - bx[:, 1]) >= ms)
            bx, sc = bx[ok], sc[ok]
            keep = nms(bx, sc, float(self.cfg.nms_iou))[:post_n]
            out.append((bx[keep].detach(), sc[keep].detach()))
        return out

    def forward(
        self, feats: Sequence[Tensor], image_sizes: Sequence[Tuple[int, int]], targets: Optional[Sequence[dict]] = None
    ) -> Tuple[List[Tuple[Tensor, Tensor]], Dict[str, Tensor]]:
        boxes, _reg, ctr, iou, pts, lvl, slices = self._head(feats)
        losses: Dict[str, Tensor] = {}
        if self.training:
            assert targets is not None, "targets required in training"
            n0 = slices[0][1] - slices[0][0]
            losses = self._loss(boxes, ctr, iou, pts, lvl, n0, targets)
        post_n = int(self.cfg.post_nms_top_n_train if self.training else self.cfg.post_nms_top_n_test)
        props = self._propose(boxes.detach(), ctr.detach(), iou.detach(), slices, image_sizes, post_n)
        return props, losses
