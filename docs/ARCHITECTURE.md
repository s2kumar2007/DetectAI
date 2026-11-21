# Architecture

```
image (letterboxed square, uint8) -> normalise
  -> GroupNorm ResNet-50 (32 groups, weights=None, kaiming)           C2..C5 (strides 4..32)   [resnet18 for tests]
  -> FPN P2..P6 (P2 optional), GN                                     256 channels
  -> OLN RPN (anchor-free): per location ltrb + centerness + box-IoU quality   proposal score = sqrt(ctr * iou_q)
  -> RoIAlign 7x7 on P2..P5 -> FC1024 -> FC1024
        |-> embed Linear -> L2-normalised 256-d  -> cosine classifier over 64 prototype slots (learnable temperature, inactive = -inf)
        |-> class-agnostic box regression (4 deltas)
        |-> RoI objectness / IoU-quality (1 logit)  = S_obj
```
Code: `src/anveshak/models/{backbone,fpn,rpn,roi_head,detector,imprint,assertions}.py`.

## Slots
64 prototype rows live in `roi_head.cls.prototypes`; `roi_head.cls.active` (bool buffer, saved in the state dict) masks them.
Allocation (`data/slots.py`): Round 1 classes 0..39 (Set A first, then Set B's new classes), Round 2 classes 40..49, 50..63 reserved.
`class_slots.json` is written beside every checkpoint and embedded in it.

## Training stages (`src/anveshak/training`)
| | Stage 1 | Stage 2 | Stage 3 (Round 2) |
|---|---|---|---|
| data | Set A | Set B (+ replay of Set A) | Round 2 set (+ replay of A and B) |
| start | random init (asserted, hash printed) | Stage 1 EMA weights | verified Stage 2 EMA weights |
| new slots | 0.. | next free (30..39) | 40..49 |
| extras | unknown mining + margin loss | imprint, pseudo-labels, replay, feature/logit KD | same, short schedule |
| frozen | - | stem + layer1, old prototypes | stem + layer1-2, old prototypes |

`engine.py` is shared: AdamW, warmup + time-aware cosine LR, gradient accumulation, BF16 autocast (CUDA), channels_last, EMA,
checkpoints every N steps (`last.pt`) with resume, `--max_hours` clean stop. `checkpoint.pt` holds weights, EMA, config, slot map,
seed, git commit, initial-weights hash, split and (stage 2/3) the previous checkpoint's SHA-256.

## Inference (`src/anveshak/postprocess`)
1. letterbox -> 4 views (orig, h-flip, v-flip, rot90) in one batch (deterministic).
2. Per view: known detections (score = sqrt(S_obj * p_class), top-2 classes, class-aware NMS) and unknown candidates
   (`unknown_score = S_obj * (1 - clamp(cos_max/cos_ref)) * alpha > tau_unk`).
3. Un-map boxes, in-house Weighted Boxes Fusion per class (unknown = one label), NMS again.
4. Conflict suppression (known vs unknown IoU >= 0.5 -> higher score survives, ties earlier row) on the rounded rows, unknown cap, 1000-row cap.
5. CSV writer + validator (`csv_check.py`; the organisers' validator too when configured).

## Evaluation (`src/anveshak/eval`)
`local_scorer.py` (mAP_K, U-Rec/U-Prec/U-F1, OE), `conflict.py` (rule applied before scoring), `io.py`.
