# yolo_anno
Simple annotation tool for YOLO

Requirements: 

`pip install ultralytics opencv-python numpy`

# pre_segment.py
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



Run examples:

  `python pre_segment.py /path/to/images --labels-dir /path/to/labels --sam-model sam_b.pt`

  `python pre_segment.py /path/to/images --labels-dir /path/to/labels --yolo-model best.pt --sam-model sam_b.pt --device mps`

# relabel.py
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

Run:
  
  `python relabel.py /path/to/images --labels-dir /path/to/labels`

# train_model.py

Train a YOLO model (Ultralytics) on your dataset.

Examples:
  python train_model.py --data data.yaml --model yolo11n.pt --epochs 100 --imgsz 640
  python train_model.py --data data.yaml --model yolo11s.pt --device mps
  python train_model.py --data data.yaml --model yolo11n.yaml --resume
  python train_model.py --data data.yaml

Prepare directories:
```
datasets/
    images/
        train/
            <images for training>
        val/
            <images for validation>
    labels
        train/
            <labels for training>
        val/
            <labels for validation>
```

and a `data.yaml` file:
```
path: datasets
train: images/train
val: images/val

names:
  0: put_your_object_name_here
```