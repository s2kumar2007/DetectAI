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
