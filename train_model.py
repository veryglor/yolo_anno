#!/usr/bin/env python3

"""
Train a YOLO model (Ultralytics) on your dataset.

Examples:
  python train_model.py --data data.yaml --model yolo11n.pt --epochs 100 --imgsz 640
  python train_model.py --data data.yaml --model yolo11s.pt --device mps
  python train_model.py --data data.yaml --model yolo11n.yaml --resume
  python train_model.py --data data.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional

def _pick_default_device() -> str:
    """
    Prefer Apple Silicon GPU (MPS) when available, otherwise CPU.
    Ultralytics accepts: 'mps', 'cpu', '0'/'1' for CUDA GPUs.
    """
    try:
        import torch  # type: ignore
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def _as_float_or_none(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    v = str(v).strip().lower()
    if v in ("none", "null", ""):
        return None
    return float(v)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a YOLO model with Ultralytics.")
    p.add_argument("--data", required=True, help="Path to dataset YAML (e.g. data.yaml).")
    p.add_argument("--model", default="yolo11n.pt", help="Model: .pt checkpoint or .yaml config.")
    p.add_argument("--project", default="runs/train", help="Project directory.")
    p.add_argument("--name", default="exp", help="Run name.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default=None, help="Device: mps, cpu, or CUDA index like 0.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--resume", action="store_true", help="Resume last run in the same project/name.")
    p.add_argument("--pretrained", action="store_true", help="Use pretrained weights (when model is .yaml).")
    p.add_argument("--optimizer", default="auto", help="auto, SGD, Adam, AdamW, RMSProp...")
    p.add_argument("--cos_lr", action="store_true", help="Use cosine LR schedule.")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision if supported on the device.")
    p.add_argument("--cache", action="store_true", help="Cache images for faster training.")
    p.add_argument("--save_period", type=int, default=-1, help="Save checkpoint every N epochs. -1 disables.")

    # Common YOLO hyperparams you may want to tweak
    p.add_argument("--lr0", default=None, help="Initial LR. Example: 0.01. Use 'none' to let YOLO choose.")
    p.add_argument("--lrf", default=None, help="Final LR fraction. Example: 0.01. Use 'none' for default.")
    p.add_argument("--weight_decay", default=None, help="Weight decay. Example: 0.0005. Use 'none' for default.")
    p.add_argument("--warmup_epochs", default=None, help="Warmup epochs. Example: 3.0. Use 'none' for default.")

    return p.parse_args()

def main() -> int:
    args = parse_args()

    device = args.device or _pick_default_device()

    # Fail early if ultralytics not installed
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        print("ERROR: 'ultralytics' is not installed.")
        print("Install with: pip install ultralytics")
        print(f"Details: {e}")
        return 2

    if not os.path.exists(args.data):
        print(f"ERROR: --data not found: {args.data}")
        return 2

    # Parse optional float hyperparams safely for older Python
    lr0 = _as_float_or_none(args.lr0)
    lrf = _as_float_or_none(args.lrf)
    weight_decay = _as_float_or_none(args.weight_decay)
    warmup_epochs = _as_float_or_none(args.warmup_epochs)

    # Build train kwargs for Ultralytics
    train_kwargs: Dict[str, Any] = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        patience=args.patience,
        resume=args.resume,
        pretrained=args.pretrained,
        optimizer=args.optimizer,
        cos_lr=args.cos_lr,
        amp=args.amp,
        cache=args.cache,
        save_period=args.save_period,
    )

    # Only pass hyperparams if user explicitly sets them
    if lr0 is not None:
        train_kwargs["lr0"] = lr0
    if lrf is not None:
        train_kwargs["lrf"] = lrf
    if weight_decay is not None:
        train_kwargs["weight_decay"] = weight_decay
    if warmup_epochs is not None:
        train_kwargs["warmup_epochs"] = warmup_epochs

    print("Ultralytics YOLO training")
    print(f"  data:   {args.data}")
    print(f"  model:  {args.model}")
    print(f"  device: {device}")
    print(f"  out:    {os.path.join(args.project, args.name)}")

    model = YOLO(args.model)
    results = model.train(**train_kwargs)

    # Optionally evaluate after training
    try:
        metrics = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=device)
        print("Validation complete.")
        print(metrics)
    except Exception as e:
        print(f"Validation skipped or failed: {e}")

    # results can be None depending on version; just return success
    _ = results
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
