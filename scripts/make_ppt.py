#!/usr/bin/env python
"""Generate the 6-slide methodology deck (python-pptx): problem understanding, methodology (x2), first implementation,
Round 2 plan, team / roles placeholder. All numbers shown are SYNTHETIC smoke results and are labelled as such.

    python scripts/make_ppt.py --out docs/Anveshak_OWOD_RS_methodology.pptx --team "Team name"
"""
import argparse
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

NAVY, TEAL, GREY, LIGHT, WHITE, AMBER = (RGBColor(0x12, 0x25, 0x3F), RGBColor(0x1F, 0x8A, 0x8A), RGBColor(0x55, 0x5F, 0x6B),
                                          RGBColor(0xEE, 0xF3, 0xF6), RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xC9, 0x7B, 0x1A))


def _text(slide, x, y, w, h, lines, size=16, color=GREY, bold_first=False, align=PP_ALIGN.LEFT, bullet=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = ("• " if bullet else "") + line
        r.font.size, r.font.color.rgb, r.font.name = Pt(size), color, "Calibri"
        r.font.bold = bold_first and i == 0
        p.space_after = Pt(6)
    return tb


def _box(slide, x, y, w, h, title, sub="", fill=TEAL, size=14):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = title
    r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), True, WHITE, "Calibri"
    if sub:
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run()
        r2.text = sub
        r2.font.size, r2.font.color.rgb, r2.font.name = Pt(size - 3), WHITE, "Calibri"
    return s


def _arrow(slide, x, y, w=0.35):
    a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(0.3))
    a.fill.solid()
    a.fill.fore_color.rgb = NAVY
    a.line.fill.background()


def _slide(prs, title, kicker):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.15))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    _text(s, 0.5, 0.18, 12, 0.6, [title], size=30, color=WHITE)
    _text(s, 0.5, 0.72, 12, 0.4, [kicker], size=14, color=RGBColor(0xA9, 0xC7, 0xD6))
    _text(s, 0.5, 7.05, 12.3, 0.35, ["anveshak-owod-rs · Techfest IIT Bombay · Anveshak"], size=10, color=GREY)
    return s


def build(out: str, team: str) -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    s = _slide(prs, "Problem understanding", "Open-world object detection for remote sensing")
    _text(s, 0.6, 1.5, 6.1, 5.3, [
        "Detect objects in aerial / satellite tiles and name the known classes.",
        "Everything else must be flagged as `unknown`, not ignored and not mislabelled.",
        "Round 1: train Set A, continue the same model on Set B (40 known classes).",
        "Round 2: continue from the verified Round 1 checkpoint and learn 10 new few-shot classes on-site in ~30 minutes.",
        "No pretrained weights, no foundation / open-vocabulary models, no network. Random init, pure PyTorch.",
    ], size=17, bullet=True)
    _box(s, 7.2, 1.6, 5.5, 0.9, "mAP_K", "known classes, IoU 0.5, VOC all-point AP", TEAL)
    _box(s, 7.2, 2.7, 5.5, 0.9, "U-F1", "recall / precision of `unknown` boxes", TEAL)
    _box(s, 7.2, 3.8, 5.5, 0.9, "Conflict rule", "known vs unknown at IoU ≥ 0.5: higher score survives", GREY)
    _box(s, 7.2, 4.9, 5.5, 0.9, "OE-Score = 100·2·mAP_K·U-F1 / (mAP_K + U-F1)", "harmonic balance of both", NAVY, size=14)
    _text(s, 7.2, 6.0, 5.5, 0.8, ["Hard parts: forgetting, unlabelled objects treated as background, tiny few-shot data, a strict time budget."], size=13)

    s = _slide(prs, "Methodology (1/2): one architecture for every stage", "Pre-allocated 64 prototype slots, masked until used")
    xs = [0.45, 2.75, 5.05, 7.35, 9.65]
    titles = [("GN ResNet-50", "random init"), ("FPN P2–P6", "256 ch"), ("OLN RPN", "centerness + IoU quality"), ("RoIAlign + 2 FC", "256-d L2 embedding"), ("Cosine head", "64 slots, learnable T")]
    for x, (t, sub) in zip(xs, titles):
        _box(s, x, 1.7, 2.05, 1.2, t, sub, TEAL)
    for x in xs[:-1]:
        _arrow(s, x + 2.07, 2.17, 0.22)
    _box(s, 9.65, 3.15, 3.2, 0.9, "Box regression + S_obj", "class-agnostic, RoI quality", GREY)
    _text(s, 0.6, 3.3, 8.8, 3.6, [
        "Proposals are scored by localisation quality, not fg/bg: score = √(centerness · IoU quality).",
        "Negatives are sampled only far from GT; the ambiguous zone is ignored so unlabelled objects are never taught as background.",
        "A new class = one more prototype. Slots: 0–39 Round 1, 40–49 Round 2, rest reserved; inactive slots are −∞.",
        "Slot map JSON is stored beside every checkpoint; architecture signature is asserted when a stage loads a checkpoint.",
    ], size=16, bullet=True)

    s = _slide(prs, "Methodology (2/2): unknowns and no forgetting", "Unknown gate, calibrated on simulated unknowns; continual learning toolkit")
    _text(s, 0.6, 1.4, 6.0, 0.4, ["Unknown detection"], size=20, color=NAVY, bold_first=True)
    _text(s, 0.6, 1.9, 6.0, 4.8, [
        "unknown = S_obj · (1 − S_known_max) · α  >  τ",
        "Unknown mining in training: high objectness, low similarity to every known prototype → soft targets + margin loss.",
        "α, τ, cap, cos_ref tuned on a local split that holds out 6 known classes as fake unknowns (never on test images).",
        "4-view deterministic TTA + in-house weighted boxes fusion, conflict-aware suppression, strict CSV validation.",
    ], size=15, bullet=True)
    _text(s, 6.9, 1.4, 6.0, 0.4, ["Continual learning"], size=20, color=NAVY, bold_first=True)
    _text(s, 6.9, 1.9, 6.0, 4.8, [
        "Weight imprinting: new prototype = mean RoI embedding of its exemplars.",
        "Frozen-teacher pseudo-labels keep old objects from becoming background.",
        "Balanced exemplar replay + feature (L2) and logit (KL) distillation.",
        "Frozen stem / early stages and old prototypes, low LR, EMA weights.",
        "Retention report after each stage: old-class mAP before vs after.",
    ], size=15, bullet=True)

    s = _slide(prs, "First implementation", "Complete, tested pipeline; verified on SYNTHETIC data only (no real data yet)")
    _text(s, 0.6, 1.4, 6.3, 5.4, [
        "Built: data auto-detection (COCO / YOLO / VOC / CSV), model, 3-stage trainer, local scorer, postprocess, CSV checker, calibration, verification, bundle.",
        "115+ CPU tests: scorer and conflict rule (hand-computed), CSV checker, slot masking, imprinting, no-pretrained assertion, determinism.",
        "Every phase committed and tagged; same lineage Set A → Set B → Round 2.",
        "Stage 1 prints the SHA-256 of the initial weights; later stages only load our own checkpoint.",
    ], size=16, bullet=True)
    _box(s, 7.3, 1.5, 5.4, 0.55, "SYNTHETIC smoke results (tiny model, CPU)", "", AMBER, size=14)
    _text(s, 7.3, 2.2, 5.4, 4.4, [
        "Stage 1: loss 4.27 → 1.78 in 600 its; untrained mAP 0.0 → trained 1.0 on easy shapes.",
        "Stage 2: old-class mAP 0.93 → 0.19 with naive fine-tuning vs 0.62–0.72 with our recipe.",
        "Found and fixed: background-weighted RoI negatives made unknowns undetectable.",
        "These are plumbing checks, not accuracy claims.",
    ], size=15, bullet=True)

    s = _slide(prs, "Round 2 plan (on-site, ~60 min)", "Start only from the verified Round 1 checkpoint")
    steps = [("0–5", "Inspect data", "formats, 10 new classes"), ("5–7", "Verify R1 ckpt", "hash + verified.json"), ("7–37", "Train (≤30 min)", "imprint first, slots 40–49"),
             ("37–45", "Infer + validate", "organisers' + built-in checker"), ("45–55", "Verify + bundle", "re-run reproduces CSV")]
    for i, (t, a, b) in enumerate(steps):
        _box(s, 0.45 + i * 2.5, 1.6, 2.3, 1.5, a, f"min {t}\n{b}", TEAL if i != 2 else NAVY, size=15)
    _text(s, 0.6, 3.5, 12.2, 3.4, [
        "Imprint new prototypes → short schedule with stem + layers 1–2 frozen, replay from Sets A + B, teacher = Round 1 checkpoint.",
        "Round 2 images contain unlabelled base, few-shot and unknown objects: pseudo-labels and unknown mining handle them.",
        "Retention report decides whether to lower LR / raise replay; calibration is reused (never re-tuned on test data).",
        "Fallbacks: resume from last.pt, smaller batch on OOM, `--no_tta` if time is short, playbook troubleshooting table.",
    ], size=16, bullet=True)

    s = _slide(prs, "Team and roles", "Placeholder - fill in before submission")
    _text(s, 0.6, 1.4, 12, 0.5, [f"Team: {team}"], size=22, color=NAVY)
    roles = [("Lead / integration", "[name]"), ("Model & training", "[name]"), ("Data & evaluation", "[name]"), ("Round 2 on-site operator", "[name]")]
    for i, (r, n) in enumerate(roles):
        _box(s, 0.6 + (i % 2) * 6.2, 2.4 + (i // 2) * 1.9, 5.9, 1.5, r, n, TEAL if i % 2 == 0 else NAVY, size=18)
    prs.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/Anveshak_OWOD_RS_methodology.pptx")
    ap.add_argument("--team", default="[team name]")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    build(a.out, a.team)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
