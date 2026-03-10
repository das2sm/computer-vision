import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load any image
img = cv2.imread('car/test/images/00000_00001_00020_png.rf.4fc3ed0f124f480c5152a541fff2dbf7.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (416, 416))

# Draw stride 8 grid (52x52)
grid_img = img_resized.copy()
stride = 8
color = (255, 0, 0)  # red
thickness = 1

for x in range(0, 416, stride):
    cv2.line(grid_img, (x, 0), (x, 416), color, thickness)
for y in range(0, 416, stride):
    cv2.line(grid_img, (0, y), (416, y), color, thickness)

plt.figure(figsize=(10, 10))
plt.imshow(grid_img)
plt.title(f'Stride 8 grid (52x52 cells) on 416x416 image\nEach cell = 8x8 pixels, responsible for detecting object centers')
plt.axis('off')
plt.tight_layout()
# plt.savefig('stride8_grid.png', dpi=150)
plt.show()
print('Saved to stride8_grid.png')