import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

CLASSES = [
    'Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100',
    'Speed Limit 110', 'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30',
    'Speed Limit 40', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70',
    'Speed Limit 80', 'Speed Limit 90', 'Stop'
]

def count_classes(split):
    label_dir = Path(f'car/{split}/labels')
    counts = Counter()
    box_sizes = []
    for lbl in label_dir.glob('*.txt'):
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            w, h = float(parts[3]), float(parts[4])
            counts[cls_id] += 1
            box_sizes.append((w * 416, h * 416))
    return counts, np.array(box_sizes)

train_counts, train_sizes = count_classes('train')
valid_counts, valid_sizes = count_classes('valid')
test_counts,  test_sizes  = count_classes('test')

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# ── 1. Class distribution per split ──────────────────────────────────────────
ax = axes[0, 0]
x = np.arange(len(CLASSES))
w = 0.28
ax.bar(x - w, [train_counts[i] for i in range(15)], w, label='Train', color='steelblue')
ax.bar(x,     [valid_counts[i] for i in range(15)], w, label='Valid', color='orange')
ax.bar(x + w, [test_counts[i]  for i in range(15)], w, label='Test',  color='green')
ax.set_xticks(x)
ax.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=9)
ax.set_title('Class Distribution per Split')
ax.set_ylabel('Number of boxes')
ax.legend()

# ── 2. Total boxes per class (all splits combined) ────────────────────────────
ax = axes[0, 1]
totals = [train_counts[i] + valid_counts[i] + test_counts[i] for i in range(15)]
bars = ax.barh(CLASSES, totals, color='steelblue')
ax.bar_label(bars, padding=3, fontsize=9)
ax.set_title('Total Boxes per Class (all splits)')
ax.set_xlabel('Count')

# ── 3. Bounding box size distribution ────────────────────────────────────────
ax = axes[1, 0]
all_sizes = np.concatenate([train_sizes, valid_sizes, test_sizes])
ax.scatter(all_sizes[:, 0], all_sizes[:, 1],
           alpha=0.2, s=5, color='steelblue')
ax.set_title('Bounding Box Size Distribution (px at 416x416)')
ax.set_xlabel('Width (px)')
ax.set_ylabel('Height (px)')
ax.set_xlim(0, 416)
ax.set_ylim(0, 416)
ax.axvline(x=32,  color='r', linestyle='--', alpha=0.5, label='32px')
ax.axvline(x=96,  color='g', linestyle='--', alpha=0.5, label='96px')
ax.axvline(x=192, color='b', linestyle='--', alpha=0.5, label='192px')
ax.legend(fontsize=8)

# ── 4. Images per split ───────────────────────────────────────────────────────
ax = axes[1, 1]
splits = ['Train', 'Valid', 'Test']
img_counts = [
    len(list(Path('car/train/images').glob('*.jpg'))),
    len(list(Path('car/valid/images').glob('*.jpg'))),
    len(list(Path('car/test/images').glob('*.jpg'))),
]
box_counts = [sum(train_counts.values()), sum(valid_counts.values()), sum(test_counts.values())]
x = np.arange(3)
w = 0.4
b1 = ax.bar(x - w/2, img_counts,  w, label='Images', color='steelblue')
b2 = ax.bar(x + w/2, box_counts,  w, label='Boxes',  color='orange')
ax.bar_label(b1, padding=3)
ax.bar_label(b2, padding=3)
ax.set_xticks(x)
ax.set_xticklabels(splits)
ax.set_title('Images vs Boxes per Split')
ax.set_ylabel('Count')
ax.legend()

plt.suptitle('Dataset Distribution Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('dataset_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved to dataset_distribution.png')