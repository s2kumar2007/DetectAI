# ASSUMPTIONS

Choices made where the brief was silent. Each is the simplest safe default and is
configurable where noted. Newest entries are appended per phase.

## General
- A1. Python >= 3.9, PyTorch >= 2.2. Scripts add `src/` to `sys.path`, so `pip install -e .` is optional.
- A2. Git identity: a global identity was already configured on the build machine, so no local one was set.
- A3. Config files are YAML; `configs/base.yaml` holds every default and is always merged first.
  Several config files can be passed (later ones win), plus `--set a.b.c=value` overrides.
- A4. Input normalisation uses the conventional mean/std constants (0.485.. / 0.229..). These are plain
  numbers, not learned weights, and do not constitute pretrained weights.

## Phase 1 - data layer
- A5. Format detection order: COCO JSON -> Pascal VOC XML -> CSV -> YOLO txt (first match under the root wins);
  `data.format` can force one. Several COCO/CSV/XML files under one root are merged.
- A6. `image_id` is the file stem by default (`data.image_id_mode: filename` switches to the full file name). The CSV
  `image_id` must match what the organisers' validator expects - confirm with `validate_predictions.py` on site.
- A7. YOLO class indices map through the class list (file order); without a class list, through `classes.txt` /
  `data.yaml`, else `class_<idx>`. YOLO polygon/OBB lines are converted to axis-aligned boxes. Images without a label
  file are kept as empty records (their objects are *unlabelled*, never assumed to be background-only for proposals).
- A8. COCO/VOC/CSV keep only images that are referenced by an annotation file; YOLO keeps every image under the root.
- A9. Numeric class labels in COCO/VOC/CSV (e.g. "3") are mapped through the class list when the label is not itself a code.
- A10. 16-bit / float TIFFs are min-max scaled to 8 bit and replicated to 3 channels.
- A11. The synthetic generator lives in `src/anveshak/data/synthetic.py` (importable by tests) with the CLI
  `scripts/make_synthetic_data.py`; `data/synthetic/` is gitignored except `.gitkeep`.

## Phase 2 - metric, CSV checker, split
- A12. Conflict rule is evaluated PAIRWISE on the original prediction set: a prediction is removed if it loses to any
  overlapping opposite-type prediction (higher score wins, ties -> earlier row), even if that opponent is itself removed
  by a third box. This is the literal reading of the brief; it is covered by a unit test.
- A13. Matching follows the VOC convention: a prediction is compared with its best-IoU GT; if that GT is already
  matched (or IoU < 0.5) the prediction is a false positive. `match_mode="greedy"` (best *free* GT) is available but not
  the default. Unknown matching uses the same procedure. Predictions are processed by descending score, ties by row order.
- A14. A GT row whose class is `unknown` OR not in the supplied known-class set is an unknown GT. With zero unknown GT
  (or zero unknown predictions) U-Rec / U-Prec are 0.0 and therefore OE = 0 - the brief does not define this case.
- A15. `known_classes` for mAP_K is the set of classes the model knows at that stage; predictions of unlisted, non-unknown
  classes are ignored by the scorer (the CSV checker rejects them in submissions).
- A16. CSV checker: BOM is rejected, duplicate = identical (image, class, score, box); images with no rows are a warning,
  not an error (a CSV cannot contain a row for an empty prediction set).
- A17. Validation split: held-out classes are random among classes with >= 3 images (seeded); each is forced into >= 2
  validation images; every other class with >= 2 images appears at least once in validation when budget allows.
  Few-shot Set B has no split (too small) - retention is measured on the Set A validation images.

## Phase 3 - model
- A18. The RPN is anchor-free (FCOS/OLN style): per-location ltrb + centerness + box-IoU quality; levels are assigned by
  max(ltrb) ranges (16 x stride per level). Positives = locations inside a GT box at the matching level (+ the nearest
  finest-level location for tiny GTs). Negatives = a few (neg_per_pos, low weight) locations OUTSIDE every GT dilated by
  `neg_dilate`; the band in between is ignored, so unlabelled objects are not trained as background.
- A19. RoI head has no background class. The cosine classifier is trained only on positive RoIs (IoU >= 0.5); background /
  objectness is carried by the RoI quality head (target = IoU for positives, 0 for far negatives, soft 0.5 for mined unknowns).
  Known-class detection score = sqrt(S_obj * p_class) (set in phase 5).
- A20. Unknown mining (Stage 1+): among non-positive RoIs, the top-K by RoI objectness with low max-cosine to active prototypes
  become soft-unknown targets (objectness target 0.5, never pushed to 0); a margin loss keeps known embeddings >= margin_known
  to their prototype and mined-unknown embeddings <= margin_unknown to every known prototype. Mining starts after
  `mining.start_frac` of the schedule (set by the trainer).
- A21. `assert_stage1_random_init` re-builds a fresh seeded model and requires bit-identical weights, so any loaded/modified
  weight is caught; it prints the SHA-256 of the initial weights. Stage 2/3 loads are validated by `assert_prev_checkpoint_ok`
  (created_by tag, expected previous stage, identical architecture signature).
- A22. Normalisation uses the common mean/std constants (plain numbers); GN groups = largest divisor <= 32 of the channel count.

## Phase 4 - Stage 1 training
- A23. Every image is letterboxed to a square `input.size` canvas (long side resized, zero-padded bottom/right; tiles smaller
  than `input.size` are up-scaled). Boxes in targets/predictions are in letterbox pixels and mapped back with the stored scale.
- A24. Augmentation randomness is a pure function of (seed, epoch, sample index). Mosaic is built from letterboxed parts, then
  flips / rot90 / scale-jitter / colour-jitter act on the result. Boxes < 40 % visible (30 % in mosaic) or < 2 px are dropped:
  they become *unlabelled* objects, never background labels.
- A25. Annotation classes missing from the class list are kept (appended after the listed ones, with a warning) rather than dropped.
- A26. Validation split priority: explicit `data.split_file` > split recorded in the previous checkpoint (Stage 2/3) > automatic
  `eval.val_frac` image split without hold-out (seeded by `seed`). Held-out classes get NO slot (never learned).
- A27. `train.max_iters` counts optimizer steps (each = `accum_steps` micro-batches). The cosine LR follows
  `progress = max(iter/max_iters, elapsed/budget)` so time-budgeted runs finish with a fully decayed LR. Resume restores
  weights, EMA, optimizer, iteration and elapsed time; data order restarts (seeded by seed + iteration), so resume is not bit-exact.
- A28. Stage 1 resume verifies that `last.pt` descends from this seed's random init (hash of the initial weights).
- A29. EMA decay warms up as min(decay, (1+u)/(10+u)); EMA weights are used for validation, imprinting, teachers and inference.
- A30. Training is seeded but NOT run with deterministic kernels (speed); determinism is enforced at inference (rule 6).
- A31. Smoke result (SYNTHETIC data only, tiny ResNet-18 GN [1,1,1,1], 600 its on CPU): loss 4.27 -> 1.78 and validation mAP 1.0
  on the 6 easy synthetic base classes (an untrained model scores 0.0). This says nothing about real-data accuracy.

## Phase 5 - unknown gate, postprocess, calibration
- A32. Smoke finding that changed a default: with RoI-objectness negatives at weight 0.1 (and 1.0) the held-out (unlabelled) objects
  were driven to objectness ~0.01 and no unknown could ever be detected (OE 0). With `roi_head.neg_obj_weight: 0.0` (pure OLN:
  quality head trained on positives only) the same tiny run recovers unknowns (SYNTHETIC val: U-Rec 0.23, OE 12.1 vs 4.2 at the
  un-tuned config defaults). Default is therefore 0.0; raise it only if real-data validation shows too many background unknowns.
- A33. S_known_max calibration = clamp(cos_max / cos_ref, 0, 1); `cos_ref` is tuned by calibrate_unknown.py together with alpha,
  tau_unk and the cap (candidates: 1.0 and percentiles 10/25/50/75 of the max-cosine of RoIs matching known GT).
- A34. Known score = sqrt(S_obj * p_class) (softmax over ACTIVE slots), top-2 classes per RoI, class-aware NMS; unknown candidates
  are class-agnostic NMS'd at `unk_nms_iou`. TTA: 4 views, boxes un-mapped, fused per class (unknown = one label) by the in-house
  WBF (a view never contributes twice to one cluster; score x min(n,views)/views), then NMS again.
- A35. Conflict suppression is applied per image on the ROUNDED rows (coords 2 dp, score 6 dp) in final row order using the same
  pairwise rule as the scorer, so no known/unknown pair >= IoU 0.5 survives and the official scorer's result is identical under
  either reading of the rule. Unknown cap is applied after conflict suppression. Rows are written sorted by score per image.
- A36. Calibration needs a checkpoint trained WITHOUT the held-out classes (a separate short "calibration run" with
  `data.split_file`); the resulting calibration.json is then applied to the final lineage. This transfer is an approximation.
- A37. The organisers' validator is invoked as `postprocess.validator_cmd` (default `python <validator> <csv>`); exit code != 0 fails.
  Our built-in checker always runs too.

## Phase 6 - Stage 2 (continual)
- A38. Slots for new classes: classes labelled in Set B that have no slot get the next free slots after the previous map, in class-list
  order (normally 30..39 when Set A has 30 classes); classes already mapped keep their slot and their labels are used as real GT.
  Stage 3 forces slots 40..49. Held-out (simulated-unknown) classes never receive a slot.
- A39. Stage 2/3 start from the EMA weights of the previous checkpoint; the same weights initialise the frozen teacher. Old prototypes
  are frozen by a gradient mask; backbone stem (+ layer1 for Stage 2, + layer2 for Stage 3) is frozen (`freeze.backbone_stages`).
- A40. Replay: images are chosen so each old class has ~`replay.per_class` labelled instances (rarest first, seeded), only old-class
  labels kept; validation images (split `val_ids`) are never replayed. `replay.ratio` of every batch comes from the buffer.
- A41. Pseudo-labels: teacher score sqrt(S_obj*p) >= `pseudo.score_thr`, class-aware NMS, dropped if IoU >= `overlap_iou` with real GT.
  Logit KD is KL over OLD-class slots only (temperature 2) on the student's sampled RoIs; feature KD is MSE on all FPN levels.
- A42. Retention report: old-class mAP (VOC, IoU .5) of the previous checkpoint ("before") vs the new EMA model ("after") on the
  validation images of the old data (Set A val split). New-class AP is only measurable on the few-shot training images (flagged optimistic).
- A43. SMOKE ablation on SYNTHETIC data (tiny model, 80 Stage-2 its, same checkpoint): old-class mAP 0.93 -> 0.19 with naive
  fine-tuning (no replay / pseudo-labels / KD / frozen prototypes), -> 0.62 with the full recipe at lr 1e-3 and 0.72 at lr 3e-4.
  Stage 2/3 learning rates should stay well below the Stage 1 LR (configs use 1e-4 / 8e-5).

## Phase 7 - Round 2, verification, bundle
- A44. "Verified Round 1 checkpoint" = `scripts/verify_checkpoint.py` returned PASS for that exact file; it writes
  `<ckpt>.verified.json` (with the checkpoint SHA-256). Stage 3 (`init.require_verified: true`) refuses a checkpoint without a matching
  passing record, and only accepts a `stage2` checkpoint (Round 1 final lineage = Set A then Set B).
- A45. verify_checkpoint pairs rows within (image, class) by smallest box distance (greedy), requires equal counts per (image, class),
  boxes within 1 px (L-inf) and scores within 1e-3; calibration / TTA default to what `<csv>.meta.json` recorded.
- A46. The bundle contains no data or images; the checkpoint is only referenced by SHA-256 unless `--include_checkpoint`. Bundling
  requires a passing verification record unless `--allow_unverified`.
- A47. Round 2 retention is measured on Set A validation images only (Set B has no held-out data); Round 2 reuses Round 1's
  calibration.json (no on-site calibration on test data).

## Phase 9 - repo audit
- A48. Commit `f89827f` ("Refresh repository contributor metadata") was verified via `git show f89827f --stat` and `git diff f89827f^ f89827f` to be an empty commit touching 0 files. Since it altered only commit metadata and no code, configs, or files, it is retained as-is per instructions without reverting.
- A49. Pretrained weight loading audit: full re-grep confirmed zero external or pretrained weight loading paths. All occurrences of `pretrained`, `weights=`, `torch.hub`, and `from_pretrained` are strict assertions (`PretrainedWeightError`), monkeypatches, or tests verifying that external weights are rejected.

