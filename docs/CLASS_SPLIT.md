# Class Split & Slot Allocation (`anveshak-owod-rs`)

The detector pre-allocates **64 prototype slots** (`model.roi_head.num_slots: 64`) in the cosine classifier. Inactive slots are masked to `-inf` so architecture and parameter shapes remain identical across all stages (Hard Rule 3).

## Slot Allocation Schedule

| Stage | Dataset Split | Class IDs | Prototype Slots | Tiles | Empty Tiles | Total Boxes | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage 1** | `set_a` (`dataset_A`) | `0..29` (30 base classes) | `0..29` | 9,659 | 3,532 | 772,752 | Extreme class imbalance (265 to 432,749 boxes/class) |
| **Stage 2** | `set_b` (`dataset_B`) | `30..39` (10 few-shot classes) | `30..39` | 677 | 480 | 428 | 15–79 boxes/class; zero base-class labels (unlabelled base/unknown objects present) |
| **Stage 3 (Round 2)** | `round2` | `40..49` (10 on-site classes) | `40..49` | TBD | TBD | TBD | Continues from verified Round 1 checkpoint |
| **Reserved** | — | — | `50..63` | — | — | — | Unused reserved capacity |
| **Unknown** | All splits | `40` in `class_list.csv` | *None* (no slot) | — | — | — | Treated as unknown marker; never assigned a trainable prototype slot |

## Verified Set A Class Distribution (Server Ground Truth)

- **Dominant classes**: `id 29` (432,749 boxes), `id 28` (265,677 boxes) — together ~90.4% of all Set A boxes.
- **High-frequency classes**: `id 27` (14,181), `id 26` (8,926), `id 25` (7,194).
- **Mid-frequency classes**: `ids 20..24` (3,876–4,706 boxes each).
- **Rare base classes**: `ids 0..19` (265–2,110 boxes each).
- **Box scale**: median ~25×23 px, 5th percentile ~9×8 px, 95th percentile ~93×80 px on 1024×1024 RGB TIFF tiles (up to 2,102 boxes in a single tile).

## Verified Set B Few-Shot Distribution

- **10 new classes** (`ids 30..39`), 15–79 boxes per class (428 boxes total across 197 labelled tiles; 480 of 677 tiles have no annotations).
- Because Set B contains unlabelled base-class and unknown objects, Stage 2 runs the frozen Stage 1 teacher on every Set B tile to generate confident base-class pseudo-labels before training, preventing base-class forgetting and negative suppression.

## Validation Hold-Out Policy (`make_val_split.py`)

- Default real-data hold-out classes for simulated-unknown calibration: **6 classes** with 1,000–5,000 boxes (`11, 14, 17, 21, 23, 24`), explicitly excluding dominant classes (`28, 29`) and very rare classes (`< 500` boxes).
- Held-out class annotations are removed from the training view (staying in the tiles as unlabelled objects) and relabelled as `"unknown"` in the validation view.
