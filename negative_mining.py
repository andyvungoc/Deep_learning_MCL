## This script is for adding false positives to training data.
import os
from pathlib import Path
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from torchvision.ops import nms

# replace with your network import path
from face_noface_networks import Net3

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".pgm"}


def read_label_file(label_path, img_w, img_h):
    """
    Support two simple formats:
      - YOLO: class x_center y_center w h (normalized 0..1)  -> 5 values per line
      - XYWH absolute: x y w h  -> 4 values per line
    Returns list of boxes in absolute xywh (x,y,w,h).
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            vals = list(map(float, parts))
            if len(vals) >= 5:
                # YOLO: skip class
                _, xc, yc, w, h = vals[:5]
                x = (xc - w / 2.0) * img_w
                y = (yc - h / 2.0) * img_h
                boxes.append([int(round(x)), int(round(y)), int(round(w * img_w)), int(round(h * img_h))])
            elif len(vals) == 4:
                x, y, w, h = vals
                boxes.append([int(round(x)), int(round(y)), int(round(w)), int(round(h))])
            else:
                continue
    return boxes


def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def iou_xywh(a, b):
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression_torch(detections, iou_threshold=0.3, score_threshold=0.0):
    # detections: list of (x,y,w,h,score)
    if len(detections) == 0:
        return []
    boxes = torch.tensor([[x, y, x + w, y + h] for x, y, w, h, s in detections], dtype=torch.float32)
    scores = torch.tensor([s for x, y, w, h, s in detections], dtype=torch.float32)
    mask = scores >= score_threshold
    if mask.sum().item() == 0:
        return []
    boxes = boxes[mask]
    scores = scores[mask]
    idxs = torch.nonzero(mask).view(-1)
    keep = nms(boxes, scores, iou_threshold)
    keep_global = idxs[keep].tolist()
    return [detections[i] for i in keep_global]


def detect_faces_multiscale_batched(model, pil_img, transform, base_window_size=32,
                                    pyramid_scales=(1.0, 0.75, 0.5), step_ratio=0.25,
                                    score_threshold=0.9, device=None, batch_size=256):
    """
    Simple image-pyramid + sliding window batched detector producing detections as (x,y,w,h,score).
    base_window_size: model input size (square)
    pyramid_scales: scales of the *image*; windows remain base_window_size
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    img_gray = pil_img.convert("L")
    img_np = np.array(img_gray)
    H, W = img_np.shape[:2]
    detections = []
    windows = []
    coords = []

    with torch.no_grad():
        for scale in pyramid_scales:
            if scale <= 0:
                continue
            new_w = max(1, int(round(W * scale)))
            new_h = max(1, int(round(H * scale)))
            resized = img_gray.resize((new_w, new_h), Image.BILINEAR)
            r_np = np.array(resized)
            h_r, w_r = r_np.shape[:2]

            win = base_window_size
            step = max(1, int(round(win * step_ratio)))
            # iterate windows on resized image
            for y in range(0, h_r - win + 1, step):
                for x in range(0, w_r - win + 1, step):
                    crop = resized.crop((x, y, x + win, y + win))
                    windows.append(transform(crop))
                    # map coords back to original image scale
                    scale_x = W / w_r
                    scale_y = H / h_r
                    x0 = int(round(x * scale_x))
                    y0 = int(round(y * scale_y))
                    w0 = int(round(win * scale_x))
                    h0 = int(round(win * scale_y))
                    coords.append((x0, y0, w0, h0))

                    if len(windows) >= batch_size:
                        batch = torch.stack(windows).to(device)
                        outputs = model(batch)
                        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                        for (cx, cy, cw, ch), p in zip(coords, probs):
                            if p >= score_threshold:
                                detections.append((cx, cy, cw, ch, float(p)))
                        windows, coords = [], []

            # leftover
        if len(windows) > 0:
            batch = torch.stack(windows).to(device)
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            for (cx, cy, cw, ch), p in zip(coords, probs):
                if p >= score_threshold:
                    detections.append((cx, cy, cw, ch, float(p)))
            windows, coords = [], []

    return detections


def save_negative_crop(pil_img, box, out_dir, prefix="neg_fp", ext=".jpg"):
    x, y, w, h = box
    W, H = pil_img.size
    # clip
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = pil_img.crop((x1, y1, x2, y2))
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{prefix}_{np.random.randint(1e9)}{ext}"
    path = out_dir / fname
    crop.save(path)
    return path


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")

    # load model
    model = Net3()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((args.base_window_size, args.base_window_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0,), std=(1,))
    ])

    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    out_neg_dir = Path(args.output_dir)

    saved = 0
    for img_path in tqdm(sorted(image_dir.rglob("*"))):
        if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTS:
            continue
        pil = Image.open(img_path).convert("RGB")
        W, H = pil.size
        label_path = label_dir / f"{img_path.stem}.txt"
        gt_boxes = read_label_file(str(label_path), W, H)  # absolute xywh

        detections = detect_faces_multiscale_batched(
            model, pil, transform,
            base_window_size=args.base_window_size,
            pyramid_scales=tuple(map(float, args.pyramid_scales.split(","))),
            step_ratio=args.step_ratio,
            score_threshold=args.score_threshold,
            device=device,
            batch_size=args.batch_size
        )

        # NMS
        detections = non_max_suppression_torch(detections, iou_threshold=args.nms_iou, score_threshold=args.score_threshold)

        # For each detection, if max IoU with any GT < iou_match -> consider false positive
        fp_count = 0
        for (x, y, w, h, score) in detections:
            max_iou = 0.0
            for gt in gt_boxes:
                max_iou = max(max_iou, iou_xywh([x, y, w, h], gt))
            if max_iou < args.iou_match:
                # save crop to output (under a folder for negatives)
                saved_path = save_negative_crop(pil, (x, y, w, h), out_neg_dir)
                if saved_path is not None:
                    saved += 1
                    fp_count += 1
                    if fp_count >= args.max_per_image:
                        break
        if saved >= args.max_total:
            break

    print(f"Saved {saved} negative crops to {out_neg_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, default="multi_face_data/images/train", help="images with ground-truth faces")
    parser.add_argument("--label-dir", type=str, default="multi_face_data/labels/train", help="label txt files (same basename). Supports YOLO normalized or absolute xywh")
    parser.add_argument("--output-dir", type=str, default="/Users/alberthogsted/Desktop/DTU/5. Semester/Machine Learning and Data Analytics/Scripts/deep_learning_project/train_images/0_fp", help="where to save negative crops (can be inside your ImageFolder class 0 dir)")
    parser.add_argument("--model-path", type=str, default="./net3_flipped.pth")
    parser.add_argument("--base-window-size", type=int, default=32)
    parser.add_argument("--pyramid-scales", type=str, default="1.0,0.75,0.5")
    parser.add_argument("--step-ratio", type=float, default=0.25)
    parser.add_argument("--score-threshold", type=float, default=0.95)
    parser.add_argument("--nms-iou", type=float, default=0.3)
    parser.add_argument("--iou-match", type=float, default=0.2, help="if detection IoU with any GT < this => treat as false positive")
    parser.add_argument("--max-per-image", type=int, default=3)
    parser.add_argument("--max-total", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--use-cuda", action="store_true")
    args = parser.parse_args()
    main(args)