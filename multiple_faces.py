
import torch
import torchvision.transforms as transforms
from PIL import Image
from face_noface_networks import Net2

# Load your trained binary face classifier
PATH = './net2.pth'
model = Net2()
model.load_state_dict(torch.load(PATH))
model.eval()

# Same transform as during training
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0,), std=(1,))
])

import cv2
import numpy as np
import matplotlib.pyplot as plt

def sliding_window(image, step_size, window_size):
    """
    Slide a window across the image.
    """
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def detect_faces_multiscale(model, image_path, scales=[1.0, 1.5, 2.0], step_ratio=0.25, threshold=0.9):
    """
    Perform multi-scale sliding window detection on an image using a face classifier.
    - base_window_size: size the model was trained on
    - scales: scales to resize the window (e.g. 1x, 1.5x, 2x)
    - step_ratio: how much to move the window (fraction of window size)
    - threshold: probability cutoff for "face"
    """
    # Load image
    original_img = Image.open(image_path).convert('L')
    img_np = np.array(original_img)
    detections = []

    base_window_size = img_np.shape[0]/10

    for scale in scales:
        win_size = int(base_window_size * scale)
        step_size = int(win_size * step_ratio)

        for (x, y, window) in sliding_window(img_np, step_size, (win_size, win_size)):
            if window.shape[0] != win_size or window.shape[1] != win_size:
                continue

            window_img = Image.fromarray(window)
            input_tensor = transform(window_img).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.softmax(output, dim=1)[0][1].item()  # class 1 = face

            if prob > threshold:
                detections.append((x, y, win_size, win_size, prob))

    return detections

def visualize_detections(image_path, detections):
    img = cv2.imread(image_path)
    for (x, y, w, h, prob) in detections:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"{prob:.2f}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

def non_max_suppression(detections, iou_thresh=0.1):
    boxes = np.array([[x, y, x+w, y+h, p] for x,y,w,h,p in detections])
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2, scores = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter_w = np.maximum(0, xx2 - xx1 + 1)
        inter_h = np.maximum(0, yy2 - yy1 + 1)
        inter = inter_w * inter_h

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return [detections[i] for i in keep]

if False:
# Example usage:
    image_path = "mface_test_img.png"
    detections = detect_faces_multiscale(
        model, image_path, base_window_size=128,
        scales=[1.0, 1.5, 2.0], step_ratio=0.25, threshold=0.999
    )
    detections_nms = non_max_suppression(detections)
    #visualize_detections(image_path, detections) 
    visualize_detections(image_path, detections_nms)

## -------------------- Improving params--------------------------------
import numpy as np
import glob
import os
import random
from tqdm import tqdm
image_dir = "multi_face_data/images/train"
label_dir = "multi_face_data/labels/val"


def iou(boxA, boxB):
    # box = (x_min, y_min, x_max, y_max)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def evaluate_detector(model, image_dir, label_dir, params, max_images=100, early_stop=True):
    """
    Evaluate sliding-window detector with early stopping.
    """
    all_tp, all_fp, all_fn = 0, 0, 0
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    random.shuffle(image_paths)

    # Limit to subset of dataset
    image_paths = image_paths[:max_images]

    for i, image_path in enumerate(tqdm(image_paths, desc="Evaluating", leave=False)):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(label_dir, base_name + ".txt")

        detections = detect_faces_multiscale(
            model,
            image_path,
            base_window_size=params["base_window_size"],
            scales=params["scales"],
            step_ratio=params["step_ratio"],
            threshold=params["threshold"]
        )
        detections = non_max_suppression(detections, iou_thresh=params["nms_iou"])

        # --- Load ground truth boxes ---
        gt_boxes = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                x1, y1, x2, y2 = map(float, parts[1:])
                gt_boxes.append((x1, y1, x2, y2))

        matched_gt = set()
        tp, fp = 0, 0
        for det in detections:
            x, y, w, h, _ = det
            pred_box = (x, y, x + w, y + h)
            best_iou, best_gt = 0, None
            for j, gt_box in enumerate(gt_boxes):
                iou_val = iou(pred_box, gt_box)
                if iou_val > best_iou:
                    best_iou, best_gt = iou_val, j
            if best_iou >= params["match_iou"]:
                tp += 1
                matched_gt.add(best_gt)
            else:
                fp += 1

        fn = len(gt_boxes) - len(matched_gt)
        all_tp += tp
        all_fp += fp
        all_fn += fn

        # --- Early stop check ---
        if early_stop and i > 20:  # wait until at least 20 images
            precision = all_tp / (all_tp + all_fp + 1e-6)
            recall = all_tp / (all_tp + all_fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            # If precision and recall are both very low, abort this config
            if f1 < 0.3:
                return {"precision": precision, "recall": recall, "f1": f1, "stopped_early": True}

    precision = all_tp / (all_tp + all_fp + 1e-6)
    recall = all_tp / (all_tp + all_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {"precision": precision, "recall": recall, "f1": f1, "stopped_early": False}



#### AUTOMATIC SEARCH
if False:
    best_f1 = 0
    best_params = None
    results = []

    search_space = {
        "step_ratio": [0.1, 0.2, 0.3],
        "threshold": [0.92,0.94,0.96,0.98,0.99],
        "nms_iou": [0.1,0.3],
    }

    for step in search_space["step_ratio"]:
        for thr in search_space["threshold"]:
            for nms_thr in search_space["nms_iou"]:
                params = {
                    "base_window_size": [32,64,128,264],
                    "scales": [1.0, 1.5, 2.0,2.5,3.0],
                    "step_ratio": step,
                    "threshold": thr,
                    "nms_iou": nms_thr,
                    "match_iou": 0.5,
                }
                print(f"\n🔍 Testing {params}")
                metrics = evaluate_detector(
                    model,
                    image_dir="/path/to/images/val",
                    label_dir="/path/to/labels2/val",
                    params=params,
                    max_images=150,       # evaluate on up to 150 images
                    early_stop=True       # stop early if poor performance
                )

                results.append({**params, **metrics})
                if not metrics["stopped_early"] and metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    best_params = params.copy()


image_paths = ["multi_face_data/images/train/0a5fbc2c83104330.jpg",
               "multi_face_data/images/train/0a9c0a5131f6487b.jpg",
               "multi_face_data/images/train/ff6d9e96e0c73de4.jpg"]

for image in image_paths:
    detections = detect_faces_multiscale(model, image, scales=[1, 1.5, 2, 2.5, 3, 4,5], threshold=0.99,step_ratio=0.25)
    nms = non_max_suppression(detections,iou_thresh=0.01)
    visualize_detections(image, nms)




