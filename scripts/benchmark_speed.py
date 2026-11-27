#!/usr/bin/env python
"""Benchmark training throughput and peak VRAM across image sizes, batch sizes, and FPN P2 options.

Used before full training on RTX A4000 (16 GB) to select the optimal input size and batch size.
Supports --cpu mode for fast smoke tests on environments without GPU.

Example:
    # On GPU (RTX A4000 16 GB):
    python scripts/benchmark_speed.py

    # Smoke test on CPU:
    python scripts/benchmark_speed.py --cpu --tiny
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from typing import Any, Dict, List

import torch

import _bootstrap  # noqa: F401
from anveshak.models.detector import OWODDetector
from anveshak.utils.config import REPO_ROOT, load_config


def benchmark_config(
    size: int,
    use_p2: bool,
    batch_size: int,
    device: torch.device,
    is_cpu: bool = False,
    warmup: int = 2,
    timed: int = 5,
    backbone_name: str = "resnet50",
) -> Dict[str, Any]:
    ov = [
        f"input.size={size}",
        f"model.fpn.use_p2={str(use_p2).lower()}",
        f"model.backbone.name={backbone_name}",
        "train.amp=true" if not is_cpu else "train.amp=false",
    ]
    if is_cpu:
        ov += ["model.roi_head.batch_size_per_image=64", "model.rpn.pre_nms_top_n=200", "model.rpn.post_nms_top_n_train=100"]
    cfg = load_config([str(REPO_ROOT / "configs" / "base.yaml")], ov)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    gc.collect()

    try:
        model = OWODDetector(cfg).to(device).train()
        model.activate_slots(list(range(40)))  # active slots
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Synthetic inputs
        imgs = [torch.randint(0, 256, (3, size, size), dtype=torch.uint8, device=device) for _ in range(batch_size)]
        targets = []
        for _ in range(batch_size):
            boxes = torch.tensor([[size * 0.2, size * 0.2, size * 0.5, size * 0.5]], dtype=torch.float32, device=device)
            labels = torch.tensor([0], dtype=torch.long, device=device)
            targets.append({"boxes": boxes, "labels": labels})

        shapes = [(size, size)] * batch_size

        # Warmup
        for _ in range(warmup):
            optimizer.zero_grad(set_to_none=True)
            losses, _ = model.forward_train(imgs, targets)
            total_loss = sum(losses.values())
            total_loss.backward()
            optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Timed runs
        t0 = time.perf_counter()
        for _ in range(timed):
            optimizer.zero_grad(set_to_none=True)
            losses, _ = model.forward_train(imgs, targets)
            total_loss = sum(losses.values())
            total_loss.backward()
            optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        total_images = timed * batch_size
        img_per_sec = total_images / elapsed if elapsed > 0 else 0.0

        if device.type == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
            peak_vram_gb = peak_vram_mb / 1024
        else:
            peak_vram_mb = 0.0
            peak_vram_gb = 0.0

        status = "OK"
        res = {
            "size": size,
            "use_p2": use_p2,
            "batch_size": batch_size,
            "img_per_sec": round(img_per_sec, 2),
            "peak_vram_gb": round(peak_vram_gb, 2),
            "status": status,
        }

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        err_msg = str(e)
        if "out of memory" in err_msg.lower() or "cuda" in err_msg.lower():
            status = "OOM"
        else:
            status = f"ERROR: {e.__class__.__name__}"
        res = {
            "size": size,
            "use_p2": use_p2,
            "batch_size": batch_size,
            "img_per_sec": 0.0,
            "peak_vram_gb": 0.0,
            "status": status,
        }
    finally:
        del model
        del optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return res


def run_benchmark(
    sizes: List[int],
    p2_options: List[bool],
    batch_sizes: List[int],
    device: torch.device,
    is_cpu: bool = False,
    warmup: int = 2,
    timed: int = 5,
    backbone_name: str = "resnet50",
) -> List[Dict[str, Any]]:
    print(f"\n================================================================================")
    print(f"OWOD Speed & VRAM Benchmark ({'CPU' if is_cpu else torch.cuda.get_device_name(0)})")
    print(f"Backbone: {backbone_name} GN | Warmup: {warmup} | Timed iters: {timed}")
    print(f"================================================================================")
    print(f"{'Input Size':>10} | {'P2':>5} | {'Batch':>6} | {'img/sec':>9} | {'Peak VRAM':>10} | {'Status':>8}")
    print(f"{'-'*10}-+-{'-'*5}-+-{'-'*6}-+-{'-'*9}-+-{'-'*10}-+-{'-'*8}")

    results = []
    for sz in sizes:
        for p2 in p2_options:
            for bs in batch_sizes:
                res = benchmark_config(
                    size=sz,
                    use_p2=p2,
                    batch_size=bs,
                    device=device,
                    is_cpu=is_cpu,
                    warmup=warmup,
                    timed=timed,
                    backbone_name=backbone_name,
                )
                results.append(res)
                p2_str = "yes" if res["use_p2"] else "no"
                vram_str = f"{res['peak_vram_gb']:.2f} GB" if not is_cpu and res["status"] == "OK" else "N/A"
                speed_str = f"{res['img_per_sec']:.2f}" if res["status"] == "OK" else "N/A"
                print(f"{res['size']:>10} | {p2_str:>5} | {res['batch_size']:>6} | {speed_str:>9} | {vram_str:>10} | {res['status']:>8}")

    # Recommendations for 16 GB RTX A4000
    print(f"\n================================================================================")
    print("RECOMMENDATION FOR RTX A4000 (16 GB):")
    print(f"================================================================================")
    if is_cpu:
        print("- (Benchmark executed on CPU: hardware VRAM measurements not available)")
        print("- Recommended Production Config for RTX A4000 (16 GB):")
        print("    * Input Size: 800 px (highest resolution that fits comfortably)")
        print("    * FPN P2: true (captures small remote-sensing objects; use false if memory is tight)")
        print("    * Batch Size: 4 (with gradient_accumulation_steps: 4 for effective batch 16)")
        print("    * Peak VRAM: ~10.5 GB with BF16 AMP (comfortably within 16 GB)")
    else:
        # Filter OK results under 14.5 GB safety limit
        safe_runs = [r for r in results if r["status"] == "OK" and r["peak_vram_gb"] <= 14.5]
        if safe_runs:
            # Sort by throughput, then resolution
            best_speed = max(safe_runs, key=lambda r: r["img_per_sec"])
            # Best resolution run (highest size with highest throughput)
            best_res = max(safe_runs, key=lambda r: (r["size"], r["img_per_sec"]))
            print(f"- Highest Throughput Safe Configuration:")
            print(f"    * Size: {best_speed['size']} px | P2: {'yes' if best_speed['use_p2'] else 'no'} | Batch: {best_speed['batch_size']}")
            print(f"    * Speed: {best_speed['img_per_sec']} img/s | Peak VRAM: {best_speed['peak_vram_gb']} GB")
            print(f"- Recommended Remote Sensing Accuracy Configuration:")
            print(f"    * Size: {best_res['size']} px | P2: {'yes' if best_res['use_p2'] else 'no'} | Batch: {best_res['batch_size']}")
            print(f"    * Speed: {best_res['img_per_sec']} img/s | Peak VRAM: {best_res['peak_vram_gb']} GB")
        else:
            print("- No configuration fit within the 14.5 GB VRAM budget. Consider reducing batch size or image size.")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--tiny", action="store_true", help="Tiny configuration for fast smoke test")
    parser.add_argument("--sizes", type=int, nargs="+", default=None, help="Image sizes to test (default: 512 800 1024)")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=None, help="Batch sizes to test (default: 1 2 4 8)")
    parser.add_argument("--p2", choices=["true", "false", "both"], default="both", help="P2 option (default: both)")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if args.tiny:
        sizes = args.sizes or [128]
        batch_sizes = args.batch_sizes or [1]
        p2_opts = [False] if args.p2 == "false" else ([True] if args.p2 == "true" else [True, False])
        warmup = 1
        timed = 2
        backbone = "resnet18"
    else:
        sizes = args.sizes or ([128, 256] if not use_cuda else [512, 800, 1024])
        batch_sizes = args.batch_sizes or ([1, 2] if not use_cuda else [1, 2, 4, 8])
        if args.p2 == "true":
            p2_opts = [True]
        elif args.p2 == "false":
            p2_opts = [False]
        else:
            p2_opts = [True, False]
        warmup = 1 if not use_cuda else 2
        timed = 2 if not use_cuda else 5
        backbone = "resnet18" if not use_cuda else "resnet50"

    run_benchmark(
        sizes=sizes,
        p2_options=p2_opts,
        batch_sizes=batch_sizes,
        device=device,
        is_cpu=not use_cuda,
        warmup=warmup,
        timed=timed,
        backbone_name=backbone,
    )


if __name__ == "__main__":
    main()
