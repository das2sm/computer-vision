"""
overfit_coco128.py
==================
Drop this file inside your pytorch-YOLOv4 repo and run from there:

  cd pytorch-YOLOv4
  python overfit_coco128.py --coco128 /path/to/coco128 --gpu 0 --epochs 50

What it does:
  1. Converts COCO128 YOLO labels -> repo annotation format (train/val .txt)
  2. Builds a config object in memory (never touches cfg.py on disk)
  3. Calls train() from train.py directly with augmentation disabled
  4. Runs inference on all 128 training images with the saved checkpoint
  5. Prints a per-image detection table + pass/fail verdict
"""

import argparse
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coco128",    required=True,        help="COCO128 root dir")
    p.add_argument("--gpu",        type=int, default=0,  help="GPU id (-1 = CPU)")
    p.add_argument("--epochs",     type=int, default=50, help="Training epochs")
    p.add_argument("--batch",      type=int, default=8,  help="Batch size")
    p.add_argument("--img-size",   type=int, default=416,help="Input size (multiple of 32)")
    p.add_argument("--lr",         type=float, default=1e-3, help="Learning rate")
    p.add_argument("--conf",       type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--nms",        type=float, default=0.4,  help="NMS IoU threshold")
    p.add_argument("--resume",     type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--skip-train", action="store_true",  help="Skip to inference only")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1 – convert COCO128 YOLO labels -> repo annotation format
# ---------------------------------------------------------------------------
def build_annotation_txt(coco128_root: Path, out_dir: Path):
    """
    COCO128 label format (normalised):  cls cx cy w h
    Repo annotation format:  /abs/img.jpg  x1,y1,x2,y2,cls  ...

    train.txt == val.txt intentionally so the model is forced to overfit.
    """
    import cv2

    img_dir   = coco128_root / "images" / "train2017"
    label_dir = coco128_root / "labels" / "train2017"

    for d, name in [(img_dir, "images"), (label_dir, "labels")]:
        if not d.exists():
            sys.exit(f"[ERROR] {name} dir not found: {d}")

    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not images:
        sys.exit(f"[ERROR] No images found in {img_dir}")

    print(f"[1/3] Converting {len(images)} images ...")
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
            if x2 > x1 and y2 > y1:
                boxes.append(f"{x1},{y1},{x2},{y2},{cls_id}")

        if boxes:
            lines.append(str(img_path.resolve()) + " " + " ".join(boxes))
        else:
            skipped += 1

    print(f"    {len(lines)} usable images, {skipped} skipped")

    out_dir.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + "\n"
    train_txt = out_dir / "coco128_train.txt"
    val_txt   = out_dir / "coco128_val.txt"
    train_txt.write_text(content)
    val_txt.write_text(content)   # same file – intentional overfit
    print(f"    {train_txt}")
    print(f"    {val_txt}")
    return train_txt, val_txt


# ---------------------------------------------------------------------------
# Step 2 – build config in memory (no disk writes to cfg.py)
# ---------------------------------------------------------------------------
def make_config(train_txt: Path, args):
    """
    Load the user's cfg.py via importlib to inherit model/anchor settings,
    then override only the training fields we care about.
    """
    import importlib.util
    from easydict import EasyDict as edict

    spec = importlib.util.spec_from_file_location("cfg_user", "cfg.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = mod.Cfg

    print(f"    Base config loaded (classes={cfg.classes})")

    # ── paths ────────────────────────────────────────────────────────────────
    cfg.train_label = str(train_txt)
    cfg.val_label   = str(train_txt)  # same file - overfit check only

    # ── training hyperparams ─────────────────────────────────────────────────
    cfg.batch          = args.batch
    cfg.subdivisions   = 1
    cfg.width          = args.img_size
    cfg.height         = args.img_size
    cfg.w              = args.img_size
    cfg.h              = args.img_size
    cfg.TRAIN_EPOCHS   = args.epochs
    cfg.learning_rate  = args.lr
    cfg.burn_in        = 0
    cfg.checkpoints    = "checkpoints_coco128"
    cfg.pretrained     = "yolov4.conv.137.pth"  # pretrained backbone
    cfg.dataset_dir    = ""    # paths in annotation txt are already absolute
    cfg.use_darknet_cfg = False # always use Yolov4, not Darknet

    # ── augmentation OFF for clean overfit signal ────────────────────────────
    cfg.mosaic         = 0
    cfg.cutmix         = 0
    cfg.mixup          = 0
    cfg.jitter         = 0.0
    cfg.flip           = 0
    cfg.blur           = 0
    cfg.angle          = 0
    cfg.saturation     = 1.0
    cfg.exposure       = 1.0
    cfg.hue            = 0.0

    return cfg


# ---------------------------------------------------------------------------
# Step 3 – train by calling train() from train.py directly
# ---------------------------------------------------------------------------
def run_training(cfg, gpu: int, resume: str = None):
    import torch
    from models import Yolov4
    from train import train   # the actual train() function in the repo

    device = torch.device(
        f"cuda:{gpu}" if (gpu >= 0 and torch.cuda.is_available()) else "cpu"
    )
    print(f"    Device : {device}")
    print(f"    Epochs : {cfg.TRAIN_EPOCHS}")
    print(f"    Batch  : {cfg.batch}")
    print(f"    Size   : {cfg.width}x{cfg.height}\n")

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
        print(f"    Checkpoint loaded successfully\n")

    model.to(device)

    # train(model, device, config, epochs, batch_size, save_cp, log_step, img_scale)
    train(
        model=model,
        device=device,
        config=cfg,
        epochs=cfg.TRAIN_EPOCHS,
        batch_size=cfg.batch,
        save_cp=True,
        log_step=10,
        img_scale=0.5,
    )


# ---------------------------------------------------------------------------
# Step 4 – inference on all 128 training images
# ---------------------------------------------------------------------------
def run_inference(coco128_root: Path, cfg, gpu: int, conf: float, nms: float):
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
        sys.exit(f"[ERROR] No checkpoints found in {ckpt_dir}/")

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

    img_dir = coco128_root / "images" / "train2017"
    images  = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    print(f"    Images     : {len(images)}  (conf={conf}, nms={nms})\n")

    results = []
    with torch.no_grad():
        for img_path in images:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            resized = cv2.resize(img_bgr, (cfg.width, cfg.height))
            img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
            tensor  = torch.from_numpy(
                img_rgb.transpose(2, 0, 1)
            ).unsqueeze(0).to(device)

            bboxes, scores = model(tensor)
            dets = post_processing(None, conf, nms, (bboxes, scores))
            n = len(dets[0]) if (dets and dets[0] is not None) else 0
            results.append((img_path.name, n))

    # ── table ────────────────────────────────────────────────────────────────
    W = 45
    print(f"{'Image':<{W}} {'Dets':>6}")
    print("─" * (W + 9))
    zero = 0
    for name, n in results:
        flag = "  <- none" if n == 0 else ""
        print(f"{name:<{W}} {n:>6}{flag}")
        if n == 0:
            zero += 1

    tested = len(results)
    total  = sum(n for _, n in results)
    print("─" * (W + 9))
    print(f"Images tested    : {tested}")
    print(f"Total detections : {total}")
    print(f"Avg dets / image : {total / max(tested, 1):.1f}")
    print(f"Zero-det images  : {zero}")
    print()

    pct = (tested - zero) / max(tested, 1) * 100
    if zero == tested:
        print("RESULT [FAIL]     No detections. Try more --epochs or lower --conf.")
    elif pct >= 80:
        print(f"RESULT [PASS]     Detections on {tested-zero}/{tested} "
              f"({pct:.0f}%) -- overfit sanity check passed.")
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

    coco128_root = Path(args.coco128).resolve()
    if not coco128_root.exists():
        sys.exit(f"[ERROR] COCO128 not found: {coco128_root}")

    print("=" * 58)
    print("  YOLOv4  |  COCO128 overfit sanity check")
    print("=" * 58)

    print("\n[1/3] Building annotation files ...")
    train_txt, val_txt = build_annotation_txt(coco128_root, Path("data"))

    print("\n[2/3] Building config ...")
    cfg = make_config(train_txt, args)

    if not args.skip_train:
        print(f"\n[2/3] Training ...")
        run_training(cfg, args.gpu, resume=args.resume)

    print("\n[3/3] Inference ...")
    run_inference(coco128_root, cfg, args.gpu, args.conf, args.nms)

    print("\n[DONE]")


if __name__ == "__main__":
    main()