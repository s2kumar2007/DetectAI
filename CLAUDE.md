# CLAUDE.md - Developer and Agent Execution Guidelines

## HARD RULES (Enforced in Code)
1. **Random Initialization Only**: Stage 1 strictly asserts that no checkpoint is configured and weights are bit-identical to a fresh seeded random initialization (SHA-256 printed). Stage 2 and Round 2 load **only** our own verified previous-stage checkpoints (tag, stage, and architecture checked). Pretrained weights (`weights=None`, `from_pretrained`, etc.) and `torch.hub` downloads are completely forbidden and blocked.
2. **Strict Offline Environment**: No open-vocabulary models, CLIP, SAM, DINO, LLMs, external models, or external network access. Socket connections to non-loopback hosts are hard-blocked at train and inference time.
3. **Consistent Architecture Across Lineage**: Same architecture throughout Stage 1 (Set A), Stage 2 (Set B), and Stage 3 (Round 2). 64 class slots are pre-allocated and masked (`inactive = -inf`).
4. **Inference-Only Test Sets**: Test images are strictly inference-only. All thresholds (`alpha`, `tau_unk`, `unk_cap`) are tuned strictly on a simulated-unknown validation split (`splits/set_a_split.json`) derived from training data.
5. **Pure PyTorch + Torchvision**: Core models rely strictly on standard PyTorch and Torchvision libraries without foreign dependencies.
6. **Deterministic Inference**: Inference is fully deterministic (fixed seeds, deterministic kernels, `torch.backends.cudnn.benchmark = False`, fixed 4-view TTA).

---

## EXECUTION RULES (Apply to Every Command)
1. **Command Timeout Limit**: Never run a blocking command longer than 10 minutes (tool kill timeout).
2. **Background Long-Running Jobs**: Start any training or long-running job in the background:
   ```bash
   python scripts/<script>.py ... > logs/<name>.log 2>&1 &
   ```
   Save the PID and poll with short status checks (`tail -n 5 logs/<name>.log`, `manage_task`, or `Get-Process`). Never use `while`/`sleep` loops that block the runner.
3. **Sequential Execution**: Run experiments one after the other, **never in parallel**. Set `torch.set_num_threads(2)` or fewer per run to prevent CPU thrashing.
4. **Smoke Tests & Diagnostics**: Smoke tests and diagnostics must use ~150-200 iterations, ResNet-18 GN, small image tiles (128x128), and small synthetic datasets. ResNet-50 GN is reserved for real training runs on the RTX A4000 GPU (16 GB).
5. **Logging & Flush Cadence**: Every training script must log loss every N iterations (default 10-25) along with iteration speed, and flush immediately. If no log line appears for 2 minutes, treat the job as stuck and investigate.
6. **Phase Completion Verification**: Before declaring any phase complete:
   - Run `pytest -q`
   - Run the phase-specific smoke test or diagnostic script
   - Verify `git status` is clean
   - Commit with message `phase-N: <title>` and create lightweight tag `phase-N` (or release tag `v1.1`)
7. **Synthetic Data Honesty**: The sandbox environment has no GPU. Never report synthetic-data smoke metrics as real detection accuracy.

---

## GIT WORKFLOW
- **One commit per phase**: Format: `phase-N: <title>` with a descriptive body detailing what changed and how it was verified.
- **Lightweight tags**: Create lightweight tags (`phase-9`, `phase-10`, `v1.1`, etc.).
- **Never squash**: Maintain linear commit history.
- **Never commit failing tests**: Start and finish from a clean working tree.
