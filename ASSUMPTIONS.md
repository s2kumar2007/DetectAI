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
