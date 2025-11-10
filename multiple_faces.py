
import torch
import torchvision.transforms as transforms
from PIL import Image
from face_noface_networks import Net2,Net3
from torchvision.ops import nms as NMS
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import random
from tqdm import tqdm

# Load your trained binary face classifier
PATH = './net3_flipped.pth'
model = Net3()
model.load_state_dict(torch.load(PATH))
model.eval()

# Same transform as during training
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0,), std=(1,))
])


def sliding_window(image, step_size, window_size):
    """
    Slide a window across the image.
    """
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def detect_faces_multiscale(model, image_path, base_window_size=32, scales=[1.0, 1.5, 2.0], step_ratio=0.25, threshold=0.9, batch_size=128, device=None):
    """
    Multi-scale sliding-window detector (batched, GPU-aware).
    - base_window_size: size the model expects (pixel width/height)
    - scales: scales to resize the base window
    - step_ratio: fraction of window size to move the window each step
    - threshold: probability cutoff for "face"
    - batch_size: number of windows to batch through the model
    - device: 'cuda' or 'cpu' (auto-detected if None)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    original_img = Image.open(image_path).convert('L')
    img_np = np.array(original_img)
    h_img, w_img = img_np.shape[:2]
    detections = []

    for scale in scales:
        win_size = int(round(base_window_size * scale))
        if win_size <= 0:
            continue
        step_size = max(1, int(round(win_size * step_ratio)))

        # collect windows and coordinates
        windows = []
        coords = []
        for y in range(0, h_img - win_size + 1, step_size):
            for x in range(0, w_img - win_size + 1, step_size):
                window = img_np[y:y + win_size, x:x + win_size]
                if window.shape[0] != win_size or window.shape[1] != win_size:
                    continue
                windows.append(window)
                coords.append((x, y, win_size, win_size))

                # run batch if size reached
                if len(windows) >= batch_size:
                    batch_t = torch.stack([transform(Image.fromarray(w)) for w in windows]).to(device)
                    with torch.no_grad():
                        outputs = model(batch_t)
                        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    for (x0, y0, ww, hh), p in zip(coords, probs):
                        if p > threshold:
                            detections.append((x0, y0, ww, hh, float(p)))
                    windows, coords = [], []

        # leftover windows
        if len(windows) > 0:
            batch_t = torch.stack([transform(Image.fromarray(w)) for w in windows]).to(device)
            with torch.no_grad():
                outputs = model(batch_t)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            for (x0, y0, ww, hh), p in zip(coords, probs):
                if p > threshold:
                    detections.append((x0, y0, ww, hh, float(p)))
            windows, coords = [], []

    return detections

def detect_faces_pyramid(model, image_path, base_window_size=32,
                         pyramid_scales=[1.0, 0.75, 0.5, 0.35, 0.25],
                         step_ratio=0.25, threshold=0.9, device=None):
    """
    Sliding-window face detector using an image pyramid (better for large images).
    - model: trained 32x32 classifier
    - pyramid_scales: scales applied to the *image* (not the window)
    - step_ratio: step size as fraction of window
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    img_gray = Image.open(image_path).convert('L')
    img_np = np.array(img_gray)
    h_img, w_img = img_np.shape
    detections = []

    for scale in pyramid_scales:
        # Resize image
        scaled_w = int(w_img * scale)
        scaled_h = int(h_img * scale)
        resized = cv2.resize(img_np, (scaled_w, scaled_h))
        step_size = int(base_window_size * step_ratio)
        windows = []
        coords = []

        # Slide 32x32 window across scaled image
        for y in range(0, scaled_h - base_window_size + 1, step_size):
            for x in range(0, scaled_w - base_window_size + 1, step_size):
                window = resized[y:y + base_window_size, x:x + base_window_size]
                windows.append(window)
                # Map back to original coordinates
                x_orig = int(x / scale)
                y_orig = int(y / scale)
                w_orig = int(base_window_size / scale)
                h_orig = int(base_window_size / scale)
                coords.append((x_orig, y_orig, w_orig, h_orig))

        # Process windows in batches
        batch_size = 128
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i+batch_size]
            batch_t = torch.stack([transform(Image.fromarray(w)) for w in batch]).to(device)
            with torch.no_grad():
                outputs = model(batch_t)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            for (x, y, w, h), p in zip(coords[i:i+batch_size], probs):
                if p > threshold:
                    detections.append((x, y, w, h, float(p)))

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

def non_max_suppression_torch(detections, iou_threshold=0.3, score_threshold=0.5):
    """
    Use torchvision NMS. detections = [(x,y,w,h,score), ...]
    Returns filtered list preserving (x,y,w,h,score).
    """
    if len(detections) == 0:
        return []
    boxes = torch.tensor([[x, y, x + w, y + h] for x, y, w, h, s in detections], dtype=torch.float32)
    scores = torch.tensor([s for x, y, w, h, s in detections], dtype=torch.float32)

    # filter by score first (optional)
    keep_scores_mask = scores >= score_threshold
    if keep_scores_mask.sum().item() == 0:
        return []
    boxes = boxes[keep_scores_mask]
    scores = scores[keep_scores_mask]
    idxs = torch.nonzero(keep_scores_mask).view(-1)

    keep = NMS(boxes, scores, iou_threshold)
    keep_global = idxs[keep].tolist()
    return [detections[i] for i in keep_global]

def verify_and_filter_detections(model, image_path, detections, transform, base_window_size=32,
                                 pad=0.25, verify_threshold=0.6, batch_size=64, device=None):
    """
    Re-classify each detected box with context padding using the model.
    Keeps detections with re-score >= verify_threshold.
    - pad: fraction of box size to add as context on each side
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    if len(detections) == 0:
        return []

    pil = Image.open(image_path).convert('L')
    W, H = pil.size
    crops = []
    coords = []

    for (x, y, w, h, score) in detections:
        pad_x = int(w * pad)
        pad_y = int(h * pad)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(W, x + w + pad_x)
        y2 = min(H, y + h + pad_y)
        crop = pil.crop((x1, y1, x2, y2)).resize((base_window_size, base_window_size))
        crops.append(transform(crop))
        coords.append((x, y, w, h, score))

    # batch predict
    keep = []
    with torch.no_grad():
        for i in range(0, len(crops), batch_size):
            batch = torch.stack(crops[i:i+batch_size]).to(device)
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            for j, p in enumerate(probs):
                if p >= verify_threshold:
                    keep.append(coords[i+j])

    return keep
# ...existing code...


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
        detections = non_max_suppression_torch(detections, iou_thresh=params["nms_iou"])

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




image_paths = [f for f in glob.glob("multi_face_data/images/val/*.jpg")]

#for image in image_paths:
imageidx = 91
thres = 0.70

#detections = detect_faces_multiscale(model, image_paths[imageidx], scales=[2, 3,4,5, 6,8,10], threshold=thres, step_ratio=0.4)
detections = detect_faces_pyramid(model, image_paths[imageidx],pyramid_scales=[0.75,0.50, 0.35, 0.25,0.20,0.10])

#nms = non_max_suppression(detections, iou_threshold=0.7)
nms = non_max_suppression_torch(detections, iou_threshold=0.3,score_threshold=thres)
#filtered = verify_and_filter_detections(model,image_paths[imageidx],nms,transform)
visualize_detections(image_paths[imageidx], nms)




