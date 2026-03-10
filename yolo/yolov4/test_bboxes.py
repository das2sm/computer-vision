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

# Load image
img_path = Path('car/test/images/FisheyeCamera_1_00257_png.rf.fd48c32db227dac063bcd71340f4f334.jpg')
lbl_path = Path('car/test/labels') / (img_path.stem + '.txt')

img = cv2.imread(str(img_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h_orig, w_orig = img.shape[:2]
img_resized = cv2.resize(img, (416, 416))

# Draw stride 8 grid
grid_img = img_resized.copy()
for x in range(0, 416, 8):
    cv2.line(grid_img, (x, 0), (x, 416), (200, 200, 200), 1)
for y in range(0, 416, 8):
    cv2.line(grid_img, (0, y), (416, y), (200, 200, 200), 1)

# Draw bounding boxes
if lbl_path.exists():
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, bw, bh = map(float, parts[1:5])

        # Convert normalised -> pixel coords on resized image
        x1 = int((cx - bw / 2) * 416)
        y1 = int((cy - bh / 2) * 416)
        x2 = int((cx + bw / 2) * 416)
        y2 = int((cy + bh / 2) * 416)

        # Draw box
        cv2.rectangle(grid_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        label = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
        cv2.putText(grid_img, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Mark center cell
        cx_px = int(cx * 416)
        cy_px = int(cy * 416)
        cv2.circle(grid_img, (cx_px, cy_px), 4, (255, 0, 0), -1)
else:
    print(f'No label file found at {lbl_path}')

plt.figure(figsize=(12, 12))
plt.imshow(grid_img)
plt.title('Stride 8 grid + GT bounding boxes (green) + centers (blue dot)')
plt.axis('off')
plt.tight_layout()
plt.show()