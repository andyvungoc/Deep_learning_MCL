## This script is for finding the best parameters for the sliding window. It takes forever to run.


import os
import glob
import random
import csv
from functools import lru_cache

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from multiple_faces import *

_img_size_cache = {}

def get_image_size(image_path):
    if image_path in _img_size_cache:
        return _img_size_cache[image_path]
    with Image.open(image_path) as img:
        w, h = img.size
    _img_size_cache[image_path] = (w, h)
    return (w, h)


def iou(boxA, boxB):
    # box = (x_min, y_min, x_max, y_max)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    interArea = inter_w * inter_h
    boxAArea = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    boxBArea = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))
    denom = boxAArea + boxBArea - interArea
    return interArea / (denom + 1e-9) if denom > 0 else 0.0


def iou_matrix(preds, gts):
    """
    preds: (N,4) xmin,ymin,xmax,ymax
    gts:   (M,4)
    returns (N,M) IoU matrix
    """
    if preds.size == 0 or gts.size == 0:
        return np.zeros((preds.shape[0], gts.shape[0]))
    N = preds.shape[0]
    M = gts.shape[0]

    # expand dims to broadcast
    px1 = preds[:, 0:1]
    py1 = preds[:, 1:2]
    px2 = preds[:, 2:3]
    py2 = preds[:, 3:4]

    gx1 = gts[None, :, 0]
    gy1 = gts[None, :, 1]
    gx2 = gts[None, :, 2]
    gy2 = gts[None, :, 3]

    inter_x1 = np.maximum(px1, gx1)
    inter_y1 = np.maximum(py1, gy1)
    inter_x2 = np.minimum(px2, gx2)
    inter_y2 = np.minimum(py2, gy2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    pred_area = np.maximum(0.0, (px2 - px1) * (py2 - py1))
    gt_area = np.maximum(0.0, (gx2 - gx1) * (gy2 - gy1))

    union = pred_area + gt_area - inter_area
    ious = inter_area / (union + 1e-9)
    return ious


def load_ground_truth(label_path, image_path):
    """
    Loads ground truth boxes from either YOLO-style (normalized) or VOC-style (pixel)
    label files.
    Returns numpy array boxes in pixel coordinates as (x_min, y_min, x_max, y_max).
    """
    try:
        w, h = get_image_size(image_path)
    except Exception:
        # fallback: open image once
        with Image.open(image_path) as img:
            w, h = img.size
        _img_size_cache[image_path] = (w, h)

    boxes = []

    with open(label_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) == 0:
            continue

        # YOLO format: class x_center y_center width height (normalized)
        if parts[0].isdigit() and len(parts) == 5:
            _, x_c, y_c, bw, bh = map(float, parts)
            x_c *= w
            y_c *= h
            bw *= w
            bh *= h
            x_min = x_c - bw / 2.0
            y_min = y_c - bh / 2.0
            x_max = x_c + bw / 2.0
            y_max = y_c + bh / 2.0
            boxes.append([x_min, y_min, x_max, y_max])
            continue

        # VOC-like format (e.g., "human x_min y_min x_max y_max")
        if parts[0].lower().startswith("human") and len(parts) == 5:
            _, x_min, y_min, x_max, y_max = parts
            boxes.append([float(x_min), float(y_min), float(x_max), float(y_max)])
            continue

        # fallback: if 4 numbers only assume pixel bbox
        if len(parts) == 4:
            x_min, y_min, x_max, y_max = map(float, parts)
            boxes.append([x_min, y_min, x_max, y_max])

    if len(boxes) == 0:
        return np.zeros((0, 4))
    return np.array(boxes, dtype=float)


def evaluate_detector(model, image_dir, label_dir, params, max_images=50, return_curves=False):
    """
    Evaluate model and return:
        f1, mAP, per_image_AP
    If return_curves=True: also return precision_curve, recall_curve
    """
    # Prefer iterating over label files to avoid images without labels
    label_paths = glob.glob(os.path.join(label_dir, "*.txt"))
    random.shuffle(label_paths)
    label_paths = label_paths[:max_images]

    # For F1
    all_tp, all_fp, all_fn = 0, 0, 0

    # For mAP
    all_detections = []   # entries: (confidence, TP_flag)
    num_gt_total = 0
    per_image_AP = []

    # aggregated curves (for optional return)
    precision_curve = np.array([])
    recall_curve = np.array([])

    for label_path in tqdm(label_paths, leave=False):
        base = os.path.splitext(os.path.basename(label_path))[0]
        img_path = os.path.join(image_dir, base + ".jpg")
        if not os.path.exists(img_path):
            # try other extensions
            found = None
            for ext in (".png", ".jpeg", ".bmp"):
                p = os.path.join(image_dir, base + ext)
                if os.path.exists(p):
                    found = p
                    break
            if found is None:
                continue
            img_path = found

        gt_boxes = load_ground_truth(label_path, img_path)
        if gt_boxes.size == 0:
            gt_boxes = np.zeros((0, 4))
        num_gt_total += gt_boxes.shape[0]

        # Predictions
        detections = detect_faces_pyramid(
            model,
            img_path,
            base_window_size=params["base_window_size"],
            pyramid_scales=params["pyramid_scales"],
            step_ratio=params.get("step_ratio", 0.2),
            threshold=params["threshold"]
        )
        detections = non_max_suppression_torch(detections, params["nms_iou"])

        # convert detections to array (xmin,ymin,xmax,ymax,score)
        if len(detections) == 0:
            det_arr = np.zeros((0, 5))
        else:
            # detections may be list of (x,y,w,h,score) or tensors
            det_list = []
            for d in detections:
                try:
                    x, y, w, h, s = d
                except Exception:
                    # if tensor
                    d = list(map(float, d))
                    x, y, w, h, s = d
                det_list.append([float(x), float(y), float(x + w), float(y + h), float(s)])
            det_arr = np.array(det_list, dtype=float)

        # Sort by score descending
        if det_arr.shape[0] > 0:
            order = np.argsort(-det_arr[:, 4])
            det_arr = det_arr[order]

        # Compute IoU matrix between preds and gts
        preds_xy = det_arr[:, :4] if det_arr.shape[0] > 0 else np.zeros((0, 4))
        gts_xy = gt_boxes if gt_boxes.shape[0] > 0 else np.zeros((0, 4))
        ious = iou_matrix(preds_xy, gts_xy)

        matched_gt = set()
        img_conf = []
        img_tp_flag = []

        # Greedy matching in score order (det_arr already sorted)
        for pi in range(det_arr.shape[0]):
            score = det_arr[pi, 4]
            if gts_xy.shape[0] == 0:
                # no GT: everything is FP
                img_conf.append(score)
                img_tp_flag.append(0)
                all_fp += 1
                all_detections.append((score, 0))
                continue

            best_gt_idx = int(np.argmax(ious[pi]))
            best_iou = float(ious[pi, best_gt_idx])
            if best_iou >= params.get("match_iou", 0.3) and best_gt_idx not in matched_gt:
                matched_gt.add(best_gt_idx)
                img_conf.append(score)
                img_tp_flag.append(1)
                all_tp += 1
                all_detections.append((score, 1))
            else:
                img_conf.append(score)
                img_tp_flag.append(0)
                all_fp += 1
                all_detections.append((score, 0))

        fn = gts_xy.shape[0] - len(matched_gt)
        all_fn += fn

        # Per-image AP (Pascal VOC 11/101-point interpolation)
        if len(img_conf) > 0 and gts_xy.shape[0] > 0:
            tp_sorted = np.array(img_tp_flag)
            cum_tp = np.cumsum(tp_sorted)
            cum_fp = np.cumsum(1 - tp_sorted)
            precision = cum_tp / (cum_tp + cum_fp + 1e-9)
            recall = cum_tp / (gts_xy.shape[0] + 1e-9)

            AP = 0.0
            for t in np.linspace(0, 1, 101):
                p = np.max(precision[recall >= t]) if np.any(recall >= t) else 0.0
                AP += p / 101.0
            per_image_AP.append(AP)
        else:
            per_image_AP.append(0.0)

    # Global F1
    precision_global = all_tp / (all_tp + all_fp + 1e-9)
    recall_global = all_tp / (all_tp + all_fn + 1e-9)
    f1 = 2 * precision_global * recall_global / (precision_global + recall_global + 1e-9)

    # Global mAP
    if len(all_detections) == 0 or num_gt_total == 0:
        mAP = 0.0
        precision_curve = np.array([])
        recall_curve = np.array([])
    else:
        dets_sorted = sorted(all_detections, key=lambda d: -d[0])
        tp_sorted = np.array([d[1] for d in dets_sorted])
        cum_tp = np.cumsum(tp_sorted)
        cum_fp = np.cumsum(1 - tp_sorted)
        precision_curve = cum_tp / (cum_tp + cum_fp + 1e-9)
        recall_curve = cum_tp / (num_gt_total + 1e-9)

        mAP = 0.0
        for t in np.linspace(0, 1, 101):
            p = np.max(precision_curve[recall_curve >= t]) if np.any(recall_curve >= t) else 0.0
            mAP += p / 101.0

    if return_curves:
        return f1, mAP, per_image_AP, precision_curve, recall_curve
    return f1, mAP, per_image_AP



def generate_pyramid(decay, max_levels=5):
    scales = [1.0]
    while len(scales) < max_levels:
        new = scales[-1] * decay
        if new < 0.2:
            break
        scales.append(new)
    return scales

def bayesian_like_search_opt_map(model,
                                 image_dir,
                                 label_dir,
                                 max_images=50,
                                 n_iter=30,
                                 initial_explore=10,
                                 log_csv="bayes_map_search.csv",
                                 seed=42):
    random.seed(seed)
    np.random.seed(seed)

    best_map = -1.0
    best_params = None
    history = []

    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iter", "threshold", "nms_iou", "scale_decay", "pyramid_scales", "map", "f1"
        ])

    def _generate_pyramid(decay, max_levels=5):
        scales = [1.0]
        while len(scales) < max_levels:
            new = scales[-1] * decay
            if new < 0.2:
                break
            scales.append(round(new, 4))
        return scales

    for i in range(n_iter):
        do_explore = (best_params is None) or (i < initial_explore) or (random.random() < 0.25)

        if do_explore:
            threshold = random.uniform(0.6, 0.95)
            nms_iou = random.uniform(0.1, 0.5)
            decay = random.uniform(0.72, 0.92)
        else:
            thr_center = best_params["threshold"]
            nms_center = best_params["nms_iou"]
            dec_center = best_params["decay"]
            threshold = float(np.clip(np.random.normal(thr_center, 0.04), 0.6, 0.98))
            nms_iou = float(np.clip(np.random.normal(nms_center, 0.04), 0.05, 0.6))
            decay = float(np.clip(np.random.normal(dec_center, 0.03), 0.65, 0.98))

        pyramid_scales = _generate_pyramid(decay)

        params = {
            "base_window_size": 32,
            "pyramid_scales": pyramid_scales,
            "step_ratio": 0.20,
            "threshold": threshold,
            "nms_iou": nms_iou,
            "match_iou": 0.3,
            "decay": decay
        }

        print(f"\nIter {i+1}/{n_iter} | explore={do_explore} | thr={threshold:.3f} nms={nms_iou:.3f} decay={decay:.3f} scales={pyramid_scales}")

        try:
            f1, mAP, per_image_AP = evaluate_detector(
                model=model,
                image_dir=image_dir,
                label_dir=label_dir,
                params=params,
                max_images=max_images,
                return_curves=False
            )
        except Exception as e:
            print(f"Evaluation error: {e}  (mAP=0)")
            mAP, f1 = 0.0, 0.0
            per_image_AP = []

        print(f"→ mAP = {mAP:.4f}, F1 = {f1:.4f}")

        history.append((i+1, threshold, nms_iou, decay, pyramid_scales, mAP, f1))

        with open(log_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([i+1, threshold, nms_iou, decay, pyramid_scales, mAP, f1])

        if mAP > best_map + 1e-9:
            best_map = mAP
            best_params = params.copy()
            print("New best mAP!")

        if i >= 12:
            last_maps = [h[5] for h in history[-6:]]
            if max(last_maps) <= best_map and all(m == last_maps[0] for m in last_maps):
                print("Early stopping — no improvement.")
                break

    if best_params is None:
        if len(history) > 0:
            last = history[-1]
            best_params = {
                "base_window_size": 32,
                "pyramid_scales": last[4],
                "step_ratio": 0.25,
                "threshold": last[1],
                "nms_iou": last[2],
                "match_iou": 0.3,
                "decay": last[3]
            }
        else:
            return None, 0.0, history

    print("Search finished.")
    print(f"Best mAP = {best_map:.4f}")

    maps = [h[5] for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(maps) + 1), maps, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("mAP")
    plt.title("Bayesian-like search (mAP) progress")
    plt.grid(True)
    plt.show()

    return best_params, best_map, history

if __name__=='__main__':
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

    

    if False:
        best_params, best_f1, history = bayesian_like_search_opt_map(
        model=model,
        image_dir="multi_face_data/images/val",
        label_dir="multi_face_data/labels/val",
        max_images=60,
        n_iter=25
        )


