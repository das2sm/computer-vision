import torch
import cv2
import numpy as np
from models import Yolov4
from tool.utils import load_class_names, post_processing, plot_boxes_cv2

def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Model
    model = Yolov4(inference=True)
    
    # Use the ZIP/Renaming loader we just built to avoid "Missing Keys"
    state_dict = torch.load("weights/yolov4.pth", map_location=device)
    new_state_dict = model.state_dict()
    for (name, param), (saved_name, saved_param) in zip(new_state_dict.items(), state_dict.items()):
        new_state_dict[name] = saved_param
    model.load_state_dict(new_state_dict)
    model.to(device).eval()

    # 2. Prepare Image
    img = cv2.imread("data/dog.jpg")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (416, 416)) # Standard YOLOv4 size
    
    # Convert to Tensor (Batch, Channels, H, W) and Scale to 0-1
    img_in = np.transpose(img_resized, (2, 0, 1)).astype(np.float32) / 255.0
    img_in = torch.from_numpy(img_in).unsqueeze(0).to(device)

    # 3. Inference
    with torch.no_grad():
        output = model(img_in)

    # 4. Post-Processing (Filtering the thousands of boxes)
    # conf_thresh: 0.4, nms_thresh: 0.6
    boxes = post_processing(img_in, 0.4, 0.6, output)

    # 5. Visualize and Save
    class_names = load_class_names("data/coco.names")
    plot_boxes_cv2(img, boxes[0], savename="final_prediction.jpg", class_names=class_names)
    print("Success! Result saved as 'final_prediction.jpg'")

if __name__ == "__main__":
    run_test()