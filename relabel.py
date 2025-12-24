#!/usr/bin/env python3

"""
Interactive YOLO bbox editor that edits label files IN-PLACE (single labels directory).

Your requested behavior:
- No out dir. Edit the existing label files in-place in --labels-dir.
- Labels ALWAYS saved as class 0 (interaction never changes class).
- Boxes have 4 control points (midpoints of each side).
- Only control points are clickable; box interior does nothing.
- Drag a control point to move that side.
- Navigation:
    ← Left Arrow  : previous image
    → Right Arrow : next image
    [ / ]         : fallback navigation (always works)
- Saves even if no changes were made (ensures a label file exists for visited images).
  If no input label exists, writes an empty .txt file.

Install:
  pip install opencv-python numpy

Run:
  python relabel.py /path/to/images --labels-dir /path/to/labels
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Literal, Dict

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
Side = Literal["left", "right", "top", "bottom"]
Box = Tuple[int, int, int, int]  # x1,y1,x2,y2 (x2/y2 exclusive)

# Arrow key codes seen across OpenCV backends
LEFT_CODES = {81, 2424832, 63234, 65361}
RIGHT_CODES = {83, 2555904, 63235, 65363}


def get_key() -> int:
    """Return raw waitKeyEx code (do NOT mask, arrows need extended codes)."""
    return cv2.waitKeyEx(30)


def is_left(k: int) -> bool:
    return k in LEFT_CODES


def is_right(k: int) -> bool:
    return k in RIGHT_CODES


def ascii_key(k: int) -> int:
    return k & 0xFF


# ---------- IO ----------

def list_images(d: Path) -> List[Path]:
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def clamp_box(b: Box, w: int, h: int, min_size: int) -> Box:
    x1, y1, x2, y2 = b
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(x1 + min_size, min(int(x2), w))
    y2 = max(y1 + min_size, min(int(y2), h))
    return x1, y1, x2, y2


def load_boxes(label_path: Path, w: int, h: int, min_size: int) -> List[Box]:
    """
    YOLO bbox format:
      <class> <cx> <cy> <bw> <bh>
    Loads geometry, ignores class values.
    """
    if not label_path.exists():
        return []
    boxes: List[Box] = []
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cx, cy, bw, bh = map(float, parts[1:])
        except ValueError:
            continue
        x1 = int(round((cx - bw / 2.0) * w))
        y1 = int(round((cy - bh / 2.0) * h))
        x2 = int(round((cx + bw / 2.0) * w))
        y2 = int(round((cy + bh / 2.0) * h))
        boxes.append(clamp_box((x1, y1, x2, y2), w, h, min_size=min_size))
    return boxes


def save_boxes_inplace(label_path: Path, boxes: List[Box], w: int, h: int) -> None:
    """
    Always write class 0 for all boxes, overwrite label_path.
    Empty boxes => empty file.
    """
    if not boxes:
        label_path.write_text("", encoding="utf-8")
        return
    lines = []
    for x1, y1, x2, y2 in boxes:
        cx = ((x1 + x2) / 2.0) / float(w)
        cy = ((y1 + y2) / 2.0) / float(h)
        bw = (x2 - x1) / float(w)
        bh = (y2 - y1) / float(h)
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------- Handles ----------

def handle_points(b: Box) -> Dict[Side, Tuple[int, int]]:
    x1, y1, x2, y2 = b
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return {
        "left": (x1, cy),
        "right": (x2, cy),
        "top": (cx, y1),
        "bottom": (cx, y2),
    }


def hit_handle(boxes: List[Box], x: int, y: int, radius: int) -> Optional[Tuple[int, Side]]:
    r2 = radius * radius
    best: Optional[Tuple[int, Side, int]] = None
    for i, b in enumerate(boxes):
        for side, (hx, hy) in handle_points(b).items():
            d2 = (x - hx) ** 2 + (y - hy) ** 2
            if d2 <= r2 and (best is None or d2 < best[2]):
                best = (i, side, d2)
    return None if best is None else (best[0], best[1])


def update_box_side(boxes: List[Box], i: int, side: Side, x: int, y: int, w: int, h: int, min_size: int) -> None:
    x1, y1, x2, y2 = boxes[i]
    if side == "left":
        x1 = x
    elif side == "right":
        x2 = x
    elif side == "top":
        y1 = y
    elif side == "bottom":
        y2 = y
    boxes[i] = clamp_box((x1, y1, x2, y2), w, h, min_size=min_size)


# ---------- Draw ----------

def draw(img: np.ndarray, boxes: List[Box], handle_radius: int, active: Optional[int]) -> np.ndarray:
    out = img.copy()
    green = (0, 255, 0)
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b
        thickness = 3 if i == active else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), green, thickness)
        for hx, hy in handle_points(b).values():
            cv2.circle(out, (hx, hy), handle_radius + 2, (0, 0, 0), -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (hx, hy), handle_radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    return out


# ---------- Main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images_dir", type=str, help="Directory with images")
    ap.add_argument("--labels-dir", required=True, type=str, help="Directory with YOLO label files to edit in-place")
    ap.add_argument("--handle-radius", type=int, default=8, help="Handle radius in pixels")
    ap.add_argument("--min-size", type=int, default=10, help="Minimum box width/height in pixels")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(images_dir)
    if not images:
        raise SystemExit(f"No images found in {images_dir}")

    win = "In-place editor | GREEN boxes | handles only | ←/→ navigate (or [ / ]) | ESC quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    idx = 0
    img: Optional[np.ndarray] = None
    boxes: List[Box] = []
    w = h = 0
    active: Optional[int] = None
    dirty = False

    def label_path_for(i: int) -> Path:
        return labels_dir / f"{images[i].stem}.txt"

    def save_current_always() -> None:
        """Always overwrite label file for current image (even if unchanged)."""
        nonlocal dirty
        if img is None:
            # If image unreadable, we can't infer w/h safely. Do nothing.
            dirty = False
            return
        save_boxes_inplace(label_path_for(idx), boxes, w, h)
        dirty = False

    def load_current() -> None:
        nonlocal img, boxes, w, h, active, dirty
        active = None
        dirty = False

        p = images[idx]
        img_local = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img_local is None:
            img = None
            boxes = []
            w = h = 0
            return

        img = img_local
        h, w = img.shape[:2]

        lp = label_path_for(idx)
        boxes[:] = load_boxes(lp, w, h, min_size=args.min_size)

        # Ensure label file exists / is normalized to class 0 even if unchanged
        save_current_always()

    # Mouse interaction
    dragging = False
    drag_i: Optional[int] = None
    drag_side: Optional[Side] = None

    def on_mouse(event, x, y, flags, param):
        nonlocal dragging, drag_i, drag_side, active, dirty
        if img is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            hit = hit_handle(boxes, x, y, radius=args.handle_radius + 3)
            if hit is None:
                return  # interior intentionally not clickable
            drag_i, drag_side = hit
            active = drag_i
            dragging = True
            update_box_side(boxes, drag_i, drag_side, x, y, w, h, min_size=args.min_size)
            dirty = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if not dragging or drag_i is None or drag_side is None:
                return
            update_box_side(boxes, drag_i, drag_side, x, y, w, h, min_size=args.min_size)
            dirty = True

        elif event == cv2.EVENT_LBUTTONUP:
            if dragging:
                dragging = False
                drag_i = None
                drag_side = None
                # Save on mouse release
                save_current_always()

    cv2.setMouseCallback(win, on_mouse)

    load_current()

    while True:
        if img is None:
            canvas = np.zeros((300, 900, 3), dtype=np.uint8)
            cv2.putText(canvas, f"Unreadable image: {images[idx].name}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, canvas)
        else:
            shown = draw(img, boxes, args.handle_radius, active)
            header = f"{idx+1}/{len(images)} {images[idx].name} | labels saved in-place as class 0 | autosave"
            cv2.putText(shown, header, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(shown, header, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow(win, shown)

        k = get_key()
        if k == -1:
            continue

        # ESC quits
        if k == 27:
            save_current_always()
            break

        # Next
        if is_right(k) or ascii_key(k) == ord(']'):
            save_current_always()
            if idx < len(images) - 1:
                idx += 1
                load_current()
            continue

        # Prev
        if is_left(k) or ascii_key(k) == ord('['):
            save_current_always()
            if idx > 0:
                idx -= 1
                load_current()
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
