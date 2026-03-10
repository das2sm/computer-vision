"""
train_cardetection.py
=====================
Trains YOLOv4 on the car/traffic-sign detection dataset and tests on the
validation set. Drop inside your pytorch-YOLOv4 repo and run from there.

  cd pytorch-YOLOv4
  python train_cardetection.py --dataset /path/to/cardetection --gpu 0 --epochs 50

Dataset layout expected (Roboflow YOLO export):
  <dataset>/
    train/images/*.jpg   train/labels/*.txt
    valid/images/*.jpg   valid/labels/*.txt
    test/images/*.jpg    test/labels/*.txt
    data.yaml
"""

import argparse
import sys
from pathlib import Path

CLASSES = [
    'Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100',
    'Speed Limit 110', 'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30',
    'Speed Limit 40', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70',
    'Speed Limit 80', 'Speed Limit 90', 'Stop'
]
NUM_CLASSES = len(CLASSES)   # 15


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    required=True,         help="Dataset root dir")
    p.add_argument("--gpu",        type=int, default=0,   help="GPU id (-1 = CPU)")
    p.add_argument("--epochs",     type=int, default=50,  help="Training epochs")
    p.add_argument("--batch",      type=int, default=4,   help="Batch size")
    p.add_argument("--img-size",   type=int, default=416, help="Input size (multiple of 32)")
    p.add_argument("--lr",         type=float, default=1e-3, help="Learning rate")
    p.add_argument("--conf",       type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--nms",        type=float, default=0.4,  help="NMS IoU threshold")
    p.add_argument("--pretrained", type=str, default="yolov4.conv.137.pth",
                                              help="Pretrained backbone weights")
    p.add_argument("--resume",     type=str, default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--skip-train", action="store_true",   help="Skip to inference only")
    p.add_argument("--test-split", type=str, default="valid",
                                   choices=["train", "valid", "test"],
                                   help="Which split to run inference on (default: valid)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1 – convert YOLO labels -> repo annotation format
# ---------------------------------------------------------------------------
def convert_split(dataset_root: Path, split: str, out_txt: Path):
    """
    Reads <dataset>/<split>/images/ and <dataset>/<split>/labels/
    Writes repo annotation format:
        /abs/path/img.jpg  x1,y1,x2,y2,cls  x1,y1,x2,y2,cls ...
    """
    import cv2

    img_dir   = dataset_root / split / "images"
    label_dir = dataset_root / split / "labels"

    for d, name in [(img_dir, "images"), (label_dir, "labels")]:
        if not d.exists():
            sys.exit(f"[ERROR] {split}/{name} not found at {d}")

    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not images:
        sys.exit(f"[ERROR] No images in {img_dir}")

    lines, skipped = [], 0
    for img_path in images:
        lbl_path = label_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            skipped += 1
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]

        boxes = []
        for line in lbl_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = max(0, int((cx - bw / 2) * w))
            y1 = max(0, int((cy - bh / 2) * h))
            x2 = min(w, int((cx + bw / 2) * w))
            y2 = min(h, int((cy + bh / 2) * h))
            if x2 > x1 and y2 > y1 and bw > 0.02 and bh > 0.02:
                boxes.append(f"{x1},{y1},{x2},{y2},{cls_id}")

        if boxes:
            lines.append(str(img_path.resolve()) + " " + " ".join(boxes))
        else:
            skipped += 1

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + "\n")
    print(f"    {split:6s} : {len(lines)} images  ({skipped} skipped)  -> {out_txt.name}")
    return out_txt


# ---------------------------------------------------------------------------
# Step 2 – build config in memory
# ---------------------------------------------------------------------------
def make_config(train_txt: Path, val_txt: Path, args):
    import importlib.util

    spec = importlib.util.spec_from_file_location("cfg_user", "cfg.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.Cfg

    # ── paths ────────────────────────────────────────────────────────────────
    cfg.train_label    = str(train_txt)
    cfg.val_label      = str(val_txt)
    cfg.dataset_dir    = ""          # paths are already absolute

    # ── model ────────────────────────────────────────────────────────────────
    cfg.classes        = NUM_CLASSES
    cfg.pretrained     = args.pretrained if Path(args.pretrained).exists() else None
    cfg.use_darknet_cfg = False

    if cfg.pretrained is None:
        print("    [WARN] Pretrained weights not found, training from scratch")
    else:
        print(f"    Pretrained : {cfg.pretrained}")

    # ── training ─────────────────────────────────────────────────────────────
    cfg.batch          = args.batch
    cfg.subdivisions   = args.batch  # each subdivision = 1 image, accumulate gradients
    cfg.width          = args.img_size
    cfg.height         = args.img_size
    cfg.w              = args.img_size
    cfg.h              = args.img_size
    cfg.TRAIN_EPOCHS   = args.epochs
    cfg.learning_rate  = args.lr
    cfg.burn_in        = 0
    cfg.checkpoints    = "checkpoints_cardetection"

    # ── augmentation ─────────────────────────────────────────────────────────
    cfg.mosaic         = 0
    cfg.cutmix         = 0
    cfg.mixup          = 0
    cfg.jitter         = 0.2
    cfg.flip           = 1
    cfg.blur           = 0
    cfg.angle          = 0
    cfg.saturation     = 1.5
    cfg.exposure       = 1.5
    cfg.hue            = 0.1

    return cfg


# ---------------------------------------------------------------------------
# Step 3 – train
# ---------------------------------------------------------------------------
def run_training(cfg, gpu: int, resume: str = None):
    import torch
    from models import Yolov4
    from train import train

    device = torch.device(
        f"cuda:{gpu}" if (gpu >= 0 and torch.cuda.is_available()) else "cpu"
    )
    print(f"    Device  : {device}")
    print(f"    Classes : {cfg.classes}  ({', '.join(CLASSES[:3])}, ...)")
    print(f"    Epochs  : {cfg.TRAIN_EPOCHS}")
    print(f"    Batch   : {cfg.batch}")
    print(f"    Size    : {cfg.width}x{cfg.height}\n")

    Path(cfg.checkpoints).mkdir(parents=True, exist_ok=True)

    model = Yolov4(
        yolov4conv137weight=cfg.pretrained,
        n_classes=cfg.classes,
        inference=False,
    )

    if resume:
        print(f"    Resuming from: {resume}")
        state = torch.load(resume, map_location=device)
        if all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state)

    # Freeze backbone, only train head
    # for name, param in model.named_parameters():
    #     if not name.startswith("head"):
    #         param.requires_grad = False
    # trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # total = sum(p.numel() for p in model.parameters())
    # print(f"    Frozen backbone: training {trainable:,} / {total:,} params (head only)")

    model.to(device)

    train(
        model=model,
        device=device,
        config=cfg,
        epochs=cfg.TRAIN_EPOCHS,
        batch_size=cfg.batch,
        save_cp=True,
        log_step=20,
        img_scale=0.5,
    )


# ---------------------------------------------------------------------------
# Step 4 – inference on a split and print results
# ---------------------------------------------------------------------------
def run_inference(dataset_root: Path, split: str, cfg, gpu: int,
                  conf: float, nms: float):
    import torch
    import cv2
    import numpy as np
    from models import Yolov4
    from tool.utils import post_processing

    ckpt_dir = Path(cfg.checkpoints)
    ckpts = sorted(
        ckpt_dir.glob("Yolov4_epoch*.pth"),
        key=lambda p: int("".join(filter(str.isdigit, p.stem)) or 0)
    )
    if not ckpts:
        sys.exit(f"[ERROR] No checkpoints in {ckpt_dir}/")

    weight_file = ckpts[-1]
    print(f"    Checkpoint : {weight_file.name}")

    device = torch.device(
        f"cuda:{gpu}" if (gpu >= 0 and torch.cuda.is_available()) else "cpu"
    )

    model = Yolov4(yolov4conv137weight=None, n_classes=cfg.classes, inference=True)
    state = torch.load(str(weight_file), map_location=device)
    if all(k.startswith("module.") for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()

    img_dir = dataset_root / split / "images"
    label_dir = dataset_root / split / "labels"
    images  = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    print(f"    Split      : {split}  ({len(images)} images)")
    print(f"    Thresholds : conf={conf}  nms={nms}\n")

    # Per-class counters
    class_tp   = [0] * NUM_CLASSES   # detected at least 1 box for a GT class
    class_gt   = [0] * NUM_CLASSES   # total GT boxes per class
    class_pred = [0] * NUM_CLASSES   # total predicted boxes per class

    results = []
    with torch.no_grad():
        for img_path in images:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            h_orig, w_orig = img_bgr.shape[:2]

            # Count GT boxes for this image
            lbl_path = label_dir / (img_path.stem + ".txt")
            gt_classes = []
            if lbl_path.exists():
                for line in lbl_path.read_text().splitlines():
                    parts = line.strip().split()
                    if parts:
                        c = int(float(parts[0]))
                        gt_classes.append(c)
                        class_gt[c] += 1

            # Run inference
            resized = cv2.resize(img_bgr, (cfg.width, cfg.height))
            img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
            tensor  = torch.from_numpy(
                img_rgb.transpose(2, 0, 1)
            ).unsqueeze(0).to(device)

            bboxes, scores = model(tensor)
            dets = post_processing(None, conf, nms, (bboxes, scores))
            det_list = dets[0] if (dets and dets[0] is not None) else []

            pred_classes = []
            if det_list is not None:
                for d in det_list:
                    # det format: [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]
                    if len(d) >= 7:
                        c = int(d[6])
                        if 0 <= c < NUM_CLASSES:
                            pred_classes.append(c)
                            class_pred[c] += 1

            n = len(pred_classes)
            results.append((img_path.name, len(gt_classes), n))

    # ── per-image table ──────────────────────────────────────────────────────
    W = 45
    print(f"{'Image':<{W}} {'GT':>5} {'Pred':>6}")
    print("─" * (W + 14))
    zero = 0
    for name, gt, pred in results:
        flag = "  <- no detections" if pred == 0 else ""
        print(f"{name:<{W}} {gt:>5} {pred:>6}{flag}")
        if pred == 0:
            zero += 1

    # ── summary ──────────────────────────────────────────────────────────────
    tested     = len(results)
    total_pred = sum(p for _, _, p in results)
    total_gt   = sum(g for _, g, _ in results)

    print("─" * (W + 14))
    print(f"Images tested      : {tested}")
    print(f"Total GT boxes     : {total_gt}")
    print(f"Total predictions  : {total_pred}")
    print(f"Avg preds / image  : {total_pred / max(tested, 1):.1f}")
    print(f"Zero-det images    : {zero}")

    # ── per-class breakdown ──────────────────────────────────────────────────
    print(f"\n{'Class':<20} {'GT':>6} {'Pred':>6}")
    print("─" * 36)
    for i, name in enumerate(CLASSES):
        print(f"{name:<20} {class_gt[i]:>6} {class_pred[i]:>6}")

    # ── verdict ──────────────────────────────────────────────────────────────
    pct = (tested - zero) / max(tested, 1) * 100
    print()
    if zero == tested:
        print("RESULT [FAIL]     No detections. Try more --epochs or lower --conf.")
    elif pct >= 70:
        print(f"RESULT [PASS]     Detections on {tested-zero}/{tested} "
              f"({pct:.0f}%) images.")
    else:
        print(f"RESULT [PARTIAL]  Detections on {tested-zero}/{tested} "
              f"({pct:.0f}%). Consider more --epochs.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = get_args()

    for f in ("train.py", "models.py", "dataset.py", "cfg.py"):
        if not Path(f).exists():
            sys.exit(
                f"[ERROR] '{f}' not found.\n"
                "Run from inside the repo:  cd pytorch-YOLOv4"
            )

    dataset_root = Path(args.dataset).resolve()
    if not dataset_root.exists():
        sys.exit(f"[ERROR] Dataset not found: {dataset_root}")

    print("=" * 58)
    print("  YOLOv4  |  Car / Traffic Sign Detection")
    print("=" * 58)
    print(f"  Dataset : {dataset_root}")
    print(f"  Classes : {NUM_CLASSES}")

    print("\n[1/3] Converting annotations ...")
    train_txt = convert_split(dataset_root, "train", Path("data") / "cardet_train.txt")
    val_txt   = convert_split(dataset_root, "valid", Path("data") / "cardet_valid.txt")
    _         = convert_split(dataset_root, "test",  Path("data") / "cardet_test.txt")

    print("\n[2/3] Building config ...")
    cfg = make_config(train_txt, val_txt, args)

    if not args.skip_train:
        print(f"\n[2/3] Training ...")
        run_training(cfg, args.gpu, resume=args.resume)

    print(f"\n[3/3] Inference on '{args.test_split}' split ...")
    run_inference(dataset_root, args.test_split, cfg, args.gpu, args.conf, args.nms)

    print("\n[DONE]")


if __name__ == "__main__":
    main()