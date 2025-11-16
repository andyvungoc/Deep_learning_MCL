## This script runs the sliding window model with the trained binary classifier.
## The final lines are for two different figures used in the report.



import torch
import torchvision.transforms as transforms
from PIL import Image
from face_noface_networks import *
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


def visualize_detections(image_path, detections, ax=None, show=True):
    """
    Draw detections on image and either show in a new figure or plot into provided Axes.
    - ax: matplotlib Axes object to draw into (if None, a new figure is created)
    - show: if True, call plt.show() after drawing (ignored when ax is provided and show=False)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_disp = img.copy()
    for (x, y, w, h, prob) in detections:
        cv2.rectangle(img_disp, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(img_disp, f"{prob:.2f}", (int(x), int(y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    img_rgb = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)

    if ax is None:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
    ax.imshow(img_rgb)
    ax.axis("off")

    if show:
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



if __name__=='__main__':
    from find_best_params_for_multi_face_bayes import generate_pyramid
    figno = 1

    if figno == 1:
        image_paths = glob.glob("multi_face_data/images/val/*.jpg")
        
        im = 76
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        img_path = image_paths[im]
        detections = detect_faces_pyramid(
            model,
            img_path,
            pyramid_scales=generate_pyramid(0.8613),
            threshold=0.98,
            step_ratio=0.19
        )
        nms = non_max_suppression_torch(detections, iou_threshold=0.1, score_threshold=0.95)
        visualize_detections(img_path, nms, show=True)
        
    if figno ==2:    
        image_paths = glob.glob("multi_face_data/images/val/*.jpg")
        ims = [76, 93, 521, 45]
        ims = [i for i in ims if i < len(image_paths)]
        if len(ims) == 0:
            raise RuntimeError("No valid image indices found in 'ims' for available images.")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for ax, im_idx in zip(axes, ims):
            img_path = image_paths[im_idx]
            detections = detect_faces_pyramid(
                model,
                img_path,
                pyramid_scales=generate_pyramid(0.7313),
                threshold=0.82,
                step_ratio=0.19
            )
            nms = non_max_suppression_torch(detections, iou_threshold=0.3, score_threshold=0.72)
            visualize_detections(img_path, nms, ax=ax, show=False)

        plt.tight_layout()
        plt.show()




