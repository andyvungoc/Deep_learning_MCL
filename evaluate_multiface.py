## This script is for finding the mAP or F1 of the sliding window model given some parameters.
from find_best_params_for_multi_face_bayes import *

params = {
            "base_window_size": 32,
            "pyramid_scales": generate_pyramid(0.74),
            "step_ratio": 0.19666,
            "threshold": 0.72,
            "nms_iou": 0.30,
            "match_iou": 0.5,
            "decay": 0.7413
        }

from face_noface_networks import Net3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded = torch.load("net3_flipped.pth", map_location=device)

if isinstance(loaded, dict):
    model = Net3().to(device)
    model.load_state_dict(loaded)
else:
    # file may contain entire model object
    model = loaded.to(device)
model.eval()

image_dir = "multi_face_data/images/val"
label_dir = "multi_face_data/labels/val"

f1,maps,aplist = evaluate_detector(model,image_dir,label_dir,params,max_images=50)

print(f"F1 = {f1}")
print(f"mAP = {maps}")
