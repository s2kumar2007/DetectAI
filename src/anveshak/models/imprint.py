"""Weight imprinting of new class prototypes from few-shot exemplars."""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Union

import torch
import torch.nn.functional as F

from ..utils.log import get_logger

log = get_logger("anveshak.imprint")


@torch.no_grad()
def imprint_new_classes(
    model,
    few_shot_loader: Iterable,
    slots: Union[Sequence[int], Mapping[str, int]],
    activate: bool = True,
    min_boxes: int = 1,
) -> Dict[int, int]:
    """Set each new prototype to the L2-normalised MEAN RoI embedding of its GT boxes.

    ``few_shot_loader`` yields ``(images, targets)`` where ``targets[i]["labels"]`` are *slot ids* and
    ``targets[i]["boxes"]`` are xyxy boxes in the (resized) image frame. The model (current or EMA copy) is used in eval
    mode under ``no_grad``. Returns {slot: number of exemplar boxes}. Raises if a slot has fewer than ``min_boxes``.
    """
    slot_ids = sorted(int(s) for s in (slots.values() if isinstance(slots, Mapping) else slots))
    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {s: 0 for s in slot_ids}
    was_training = model.training
    model.eval()
    try:
        for images, targets in few_shot_loader:
            boxes = [t["boxes"].float() for t in targets]
            labels = [t["labels"].long() for t in targets]
            if not any(len(b) for b in boxes):
                continue
            emb = model.embed_gt_boxes(images, boxes).float()
            lab = torch.cat(labels).to(emb.device)
            for s in slot_ids:
                m = lab == s
                if m.any():
                    sums[s] = sums.get(s, 0) + emb[m].sum(0)
                    counts[s] += int(m.sum())
    finally:
        model.train(was_training)
    missing = [s for s in slot_ids if counts[s] < min_boxes]
    if missing:
        raise ValueError(f"no exemplar boxes found for slots {missing}; cannot imprint")
    for s in slot_ids:
        mean = sums[s] / counts[s]
        model.roi_head.cls.set_prototype(s, F.normalize(mean, dim=0))
    if activate:
        model.roi_head.cls.activate(slot_ids)
    log.info("imprinted slots %s from boxes %s", slot_ids, [counts[s] for s in slot_ids])
    return counts
