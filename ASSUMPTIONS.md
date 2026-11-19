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
