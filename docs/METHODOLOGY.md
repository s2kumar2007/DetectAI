# Methodology

## Problem
Detect objects in remote-sensing tiles, name the known classes, label everything else `unknown`, add classes over time without
forgetting, from random initialisation only. Score: `OE = 100*2*mAP_K*U-F1/(mAP_K+U-F1)`.

## Open-world ingredients
- **Localisation-quality proposals (OLN style).** The RPN predicts centerness and box-IoU quality instead of fg/bg. Positives are
  locations inside a GT box; negatives come only from locations outside every GT box dilated 2x, with low weight; the band in
  between is ignored. The RoI objectness head is trained on positives only (`neg_obj_weight: 0`).
  *Smoke finding:* with background-weighted RoI negatives the held-out objects collapsed to objectness ~0.01 and no unknown could be detected.
- **Cosine prototypes.** Embeddings are L2-normalised and compared with 64 prototype slots (no background class), so a new class
  is one more prototype; unseen objects show up as low maximum cosine.
- **Unknown mining.** After 20 % of the schedule, the top-K non-overlapping RoIs with high objectness but low known similarity get soft
  objectness targets (never 0) and a margin loss keeps known embeddings close to their prototype and mined unknowns away from all prototypes.
- **Unknown gate.** `unknown_score = S_obj * (1 - S_known_cal) * alpha`; `alpha, tau_unk, cap, cos_ref` are tuned on a simulated-unknown
  split (6 known classes held out as fake unknowns, labels removed from training, kept in validation as `unknown`) by maximising OE.

## Continual learning (no replay of test data, no re-training from scratch)
Imprint new prototypes from exemplar RoI embeddings; teacher pseudo-labels keep old objects from becoming background; a balanced
exemplar buffer replays old data; FPN feature L2 + old-class logit KL distillation; frozen early stages and frozen old prototypes;
low LR. A retention report (old-class mAP before/after on held-out validation images) is saved by each stage.

## Reproducibility
Seeds everywhere, initial-weights hash, checkpoint SHA-256 chain (stage 2/3 record the previous checkpoint's hash), deterministic
inference, `verify_checkpoint.py` re-runs the model against the submitted CSV, git commit stored in every checkpoint.

## Smoke results (SYNTHETIC data, tiny ResNet-18 GN, CPU; not real-data accuracy)
| check | result |
|---|---|
| Stage 1, 600 its | loss 4.27 -> 1.78; validation mAP 1.0 on 6 easy classes (untrained model: 0.0) |
| unknown gate, neg_obj_weight 0.1 vs 0.0 | U-Rec 0 / OE 0 vs U-Rec 0.23 / OE 12.1 (calibrated) |
| Stage 2 ablation, 80 its | old-class mAP 0.93 -> 0.19 naive fine-tune; 0.62 (lr 1e-3) / 0.72 (lr 3e-4) with the full recipe |
| full CLI pipeline (`make smoke`) | all steps, validators and verification PASS; mAP_K 0.60 (R1) / 0.64 (R2); **U-F1 0.0** on the synthetic test tiles (tiny undertrained model) |
| tests | retention 1.00 (Stage 2), 0.76 -> 0.69 (Round 2) |

## Limitations
Real-data accuracy is unknown; the calibration run is a proxy model; the OLN-style objectness trades background precision for unknown recall;
interpretations of the brief (conflict rule, matching, id format) are listed in `ASSUMPTIONS.md` and should be checked against the organisers' validator.
