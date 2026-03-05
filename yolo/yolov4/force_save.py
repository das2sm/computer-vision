import torch
from tool.darknet2pytorch import Darknet

# 1. Load the architecture and weights
model = Darknet('cfg/yolov4.cfg', inference=True)
model.load_weights('weights/yolov4.weights')

# 2. Force the save
print("Saving weights to yolov4.pth...")
torch.save(model.state_dict(), 'weights/yolov4.pth')
print("Done! Check your weights folder now.")