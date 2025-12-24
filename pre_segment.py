#!/usr/bin/env python3

"""
pre_segment.py

Generates EXACTLY ONE YOLO bbox label per image, saving to per-image .txt files.

Pipeline per image:
1) If --yolo-model is provided:
   - Run YOLO detection first.
   - If it finds >=1 box: pick the centermost box, SKIP SAM, save it with class 0.
2) Otherwise (no YOLO model OR YOLO found 0 boxes):
   - Run Ultralytics SAM "segment everything".
   - Convert masks -> boxes.
   - If SAM finds >=1 box: pick the centermost box.
3) If still no box:
   - Create a 10x10 pixel box at the image center.

Rules:
- Always output one and only one box per image.
- Output written to: <labels_dir>/<image_stem>.txt
- YOLO label format (bbox): 0 cx cy w h  (normalized)
- Shows progress percentage and current image name in console.

Install:
  pip install ultralytics opencv-python numpy

Run examples:
  python pre_segment.py /path/to/images --labels-dir /path/to/labels --sam-model sam_b.pt
  python pre_segment.py /path/to/images --labels-dir /path/to/labels --yolo-model best.pt --sam-model sam_b.pt --device mps
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO, SAM

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(directory: Path) -> List[Path]:
    return sorted([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def clamp_box_xyxy(box: Tuple[float, float, float, float], w: int, h: int, min_size: int = 2) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1 = int(max(0, min(round(x1), w - 1)))
    y1 = int(max(0, min(round(y1), h - 1)))
    x2 = int(max(x1 + min_size, min(round(x2), w)))
    y2 = int(max(y1 + min_size, min(round(y2), h)))
    return x1, y1, x2, y2


def xyxy_to_yolo_line(cls: int, box: Tuple[int, int, int, int], w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    bw = (x2 - x1) / float(w)
    bh = (y2 - y1) / float(h)
    cx = ((x1 + x2) / 2.0) / float(w)
    cy = ((y1 + y2) / 2.0) / float(h)
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def centermost_box_xyxy(boxes: List[Tuple[int, int, int, int]], w: int, h: int) -> Tuple[int, int, int, int]:
    icx, icy = w / 2.0, h / 2.0
    best_i = 0
    best_d2 = float("inf")
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        d2 = (cx - icx) ** 2 + (cy - icy) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return boxes[best_i]


def make_center_box_10x10(w: int, h: int) -> Tuple[int, int, int, int]:
    cx, cy = w // 2, h // 2
    half = 5
    x1 = cx - half
    y1 = cy - half
    x2 = cx + half
    y2 = cy + half
    return clamp_box_xyxy((x1, y1, x2, y2), w, h, min_size=2)


def sam_masks_to_boxes(masks: np.ndarray) -> List[Tuple[int, int, int, int]]:
    out: List[Tuple[int, int, int, int]] = []
    n, h, w = masks.shape
    for i in range(n):
        ys, xs = np.where(masks[i] > 0)
        if xs.size == 0 or ys.size == 0:
            continue
        x1 = int(xs.min())
        x2 = int(xs.max()) + 1
        y1 = int(ys.min())
        y2 = int(ys.max()) + 1
        out.append(clamp_box_xyxy((x1, y1, x2, y2), w, h, min_size=2))
    return out


def run_yolo_get_boxes(yolo: YOLO, img_path: Path, device: Optional[str], conf: float, imgsz: int) -> List[Tuple[int, int, int, int]]:
    kwargs = dict(conf=conf, imgsz=imgsz, verbose=False)
    if device is not None:
        kwargs["device"] = device
    res = yolo(str(img_path), **kwargs)
    if not res or res[0].boxes is None or res[0].boxes.xyxy is None or len(res[0].boxes) == 0:
        return []
    xyxy = res[0].boxes.xyxy.detach().cpu().numpy()
    return [(float(x1), float(y1), float(x2), float(y2)) for x1, y1, x2, y2 in xyxy]  # type: ignore[misc]


def run_sam_get_boxes(sam: SAM, img_path: Path, device: Optional[str], imgsz: int) -> List[Tuple[int, int, int, int]]:
    kwargs = dict(imgsz=imgsz, verbose=False)
    if device is not None:
        kwargs["device"] = device
    res = sam(str(img_path), **kwargs)
    if not res or res[0].masks is None or res[0].masks.data is None:
        return []
    masks = (res[0].masks.data.detach().cpu().numpy() > 0.0).astype(np.uint8)
    return sam_masks_to_boxes(masks)


def print_progress(i: int, total: int, name: str, method: str) -> None:
    pct = (i / total) * 100.0
    # single-line updating
    msg = f"[{i}/{total}] {pct:6.2f}%  {name}  {method}"
    print(msg, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images_dir", type=str, help="Directory containing images")
    ap.add_argument("--labels-dir", required=True, type=str, help="Directory to write per-image .txt labels")
    ap.add_argument("--yolo-model", default=None, type=str, help="Optional YOLO model path (e.g., best.pt). If provided, run YOLO first.")
    ap.add_argument("--sam-model", default="sam_b.pt", type=str, help="SAM weights (e.g., sam_b.pt, sam_l.pt)")
    ap.add_argument("--device", default=None, type=str, help="Device (cpu, mps, 0, 1, etc.)")
    ap.add_argument("--yolo-conf", default=0.25, type=float, help="YOLO confidence threshold")
    ap.add_argument("--yolo-imgsz", default=640, type=int, help="YOLO inference size")
    ap.add_argument("--sam-imgsz", default=1024, type=int, help="SAM inference size")
    args = ap.parse_args()

    img_dir = Path(args.images_dir)
    if not img_dir.exists() or not img_dir.is_dir():
        raise SystemExit(f"Not a directory: {img_dir}")

    labels_dir = Path(args.labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(img_dir)
    if not images:
        raise SystemExit(f"No images found in {img_dir}")

    yolo = YOLO(args.yolo_model) if args.yolo_model else None
    sam = SAM(args.sam_model)

    total = len(images)

    for i, img_path in enumerate(images, start=1):
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print_progress(i, total, img_path.name, "SKIP(unreadable)")
            continue

        h, w = img.shape[:2]
        chosen_box: Optional[Tuple[int, int, int, int]] = None
        method = ""

        # 1) YOLO first if provided
        if yolo is not None:
            yolo_boxes_raw = run_yolo_get_boxes(yolo, img_path, args.device, args.yolo_conf, args.yolo_imgsz)
            yolo_boxes = [clamp_box_xyxy(b, w, h, min_size=2) for b in yolo_boxes_raw]
            if yolo_boxes:
                chosen_box = centermost_box_xyxy(yolo_boxes, w, h)
                method = f"YOLO({len(yolo_boxes)})"

        # 2) Else SAM
        if chosen_box is None:
            sam_boxes = run_sam_get_boxes(sam, img_path, args.device, args.sam_imgsz)
            sam_boxes = [clamp_box_xyxy(b, w, h, min_size=2) for b in sam_boxes]
            if sam_boxes:
                chosen_box = centermost_box_xyxy(sam_boxes, w, h)
                method = f"SAM({len(sam_boxes)})"
            else:
                chosen_box = make_center_box_10x10(w, h)
                method = "FALLBACK(10x10)"

        out_path = labels_dir / f"{img_path.stem}.txt"
        out_path.write_text(xyxy_to_yolo_line(0, chosen_box, w, h) + "\n", encoding="utf-8")

        print_progress(i, total, img_path.name, f"-> {out_path.name}  {method}")

    print(f"\nDone. Labels written to: {labels_dir.resolve()}")


if __name__ == "__main__":
    main()
