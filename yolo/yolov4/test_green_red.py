import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CLASSES = [
    'Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100',
    'Speed Limit 110', 'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30',
    'Speed Limit 40', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70',
    'Speed Limit 80', 'Speed Limit 90', 'Stop'
]

# Find images that contain BOTH green and red lights
label_dir = Path('car/train/labels')
img_dir   = Path('car/train/images')

both = []
for lbl in label_dir.glob('*.txt'):
    classes_in_img = set()
    for line in lbl.read_text().splitlines():
        parts = line.strip().split()
        if parts:
            classes_in_img.add(int(float(parts[0])))
    if 0 in classes_in_img and 1 in classes_in_img:
        both.append(lbl.stem)

print(f'Images with both Green and Red lights: {len(both)}')

# Visualise first 4
fig, axes = plt.subplots(2, 2, figsize=(14, 14))
for idx, stem in enumerate(both[:4]):
    img_path = img_dir / (stem + '.jpg')
    lbl_path = label_dir / (stem + '.txt')
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        color = (0, 255, 0) if cls_id == 0 else (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, CLASSES[cls_id], (x1, max(y1-5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    ax = axes[idx // 2][idx % 2]
    ax.imshow(img)
    ax.set_title(stem[:40])
    ax.axis('off')

plt.suptitle('Green (green box) vs Red Light (red box) comparison', fontsize=13)
plt.tight_layout()
plt.show()