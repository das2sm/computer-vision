from pathlib import Path
import numpy as np

label_dir = Path('car/train/labels')
collisions = 0
total = 0

for lbl in label_dir.glob('*.txt'):
    boxes = []
    for line in lbl.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy = float(parts[1]), float(parts[2])
        cls = int(float(parts[0]))
        # which grid cell at stride 8 (52x52)
        cell_x = int(cx * 52)
        cell_y = int(cy * 52)
        boxes.append((cell_x, cell_y, cls))
    
    # check for same-cell collisions
    cells = [(x, y) for x, y, c in boxes]
    total += len(boxes)
    collisions += len(cells) - len(set(cells))

print(f'Total boxes: {total}')
print(f'Grid cell collisions: {collisions} ({100*collisions/total:.1f}%)')