#!/usr/bin/env python3
"""
评估脚本：对一批已生成的 LIME 结果计算定量指标。
输出 CSV 包含：Overlap@k (k=0.01,0.05,0.1)、IoU@k、Deletion AUC、Insertion AUC

用法示例：
python3 evaluate_metrics.py --results_dir ./results/batch --data_dir ../pattern_recognition/dataset/BraTS2021_Data \
    --model_path ../pattern_recognition/checkpoints/nnunet/brats/nnunet_best.pth --out_csv ./results/metrics_summary.csv
"""
import os
import argparse
import numpy as np
import csv
from glob import glob

# 导入工具与模型加载
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import importlib.util
spec = importlib.util.spec_from_file_location('lime_analysis', os.path.join(os.path.dirname(__file__), 'lime_analysis.py'))
lime_analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lime_analysis)

import utils

import torch
try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None


def top_k_mask_from_heatmap(heatmap, k_ratio):
    flat = heatmap.flatten()
    k = int(len(flat) * k_ratio)
    if k <= 0:
        k = 1
    thresh = np.partition(flat, -k)[-k]
    mask = (heatmap >= thresh).astype(np.uint8)
    return mask


def iou(mask1, mask2):
    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def deletion_insertion_auc(model_wrapper, original_image, heatmap, segmentation_fn, device, mode='deletion', steps=50, baseline_type='mean', gaussian_sigma=2.0):
    # original_image: (H,W) normalized
    H, W = original_image.shape
    # flatten indices by importance descending
    hm = heatmap.copy()
    # rank pixels
    idxs = np.argsort(hm.flatten())[::-1]
    total = H * W
    fractions = np.linspace(0, 1.0, steps)
    scores = []

    # baseline for insertion: mean image or gaussian blurred image
    if baseline_type == 'gaussian' and gaussian_filter is not None:
        baseline = gaussian_filter(original_image, sigma=gaussian_sigma)
    else:
        baseline = np.full_like(original_image, np.mean(original_image))

    for f in fractions:
        k = int(total * f)
        mask = np.zeros(total, dtype=bool)
        if k > 0:
            mask[idxs[:k]] = True
        mask2d = mask.reshape(H, W)

        if mode == 'deletion':
            # remove top-k pixels -> set to baseline (mean or gaussian)
            pert = original_image.copy()
            pert[mask2d] = baseline[mask2d]
        else:
            # insertion: start from baseline, insert top-k pixels from original
            pert = baseline.copy()
            pert[mask2d] = original_image[mask2d]

        # call segmentation_prediction_wrapper expects (H,W) or (H,W,C)
        pred = segmentation_fn_single(segmentation_fn=segmentation_fn, img=pert, device=device)
        scores.append(pred)

    # compute normalized AUC: integrate normalized score (use trapezoid)
    scores = np.array(scores)
    eps = 1e-8
    norm_scores = scores / (scores[0] + eps)
    auc = np.trapz(norm_scores, fractions) / (fractions[-1] - fractions[0])
    return float(auc), fractions, scores


def segmentation_fn_single(segmentation_fn, img, device):
    # segmentation_fn is lime_analysis.segmentation_prediction_wrapper
    # It expects either (H,W) or batch; return tumor score (index 1)
    res = segmentation_fn(img)
    if isinstance(res, np.ndarray):
        # take first sample and tumor class
        if res.ndim == 2 and res.shape[1] >= 2:
            return float(res[0, 1])
        elif res.ndim == 1:
            return float(res[0])
    # fallback
    return 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--out_csv', type=str, default='./results/metrics_summary.csv')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--baseline', type=str, default='gaussian', choices=['mean','gaussian'], help='baseline type for insertion/deletion')
    parser.add_argument('--gaussian_sigma', type=float, default=2.0, help='sigma for gaussian baseline')
    args = parser.parse_args()

    DEVICE = torch.device(args.device)

    # load model
    MODEL = lime_analysis.load_trained_model(args.model_path, DEVICE)
    # set global MODEL in lime_analysis so segmentation_prediction_wrapper uses it
    lime_analysis.MODEL = MODEL
    lime_analysis.DEVICE = DEVICE

    case_dirs = sorted(glob(os.path.join(args.results_dir, '*')))
    rows = []
    for case_dir in case_dirs:
        case_id = os.path.basename(case_dir)
        heat_path = os.path.join(case_dir, f"{case_id}_lime_heatmap.npy")
        gt_path = os.path.join(case_dir, f"{case_id}_gt_mask.npy")
        if not os.path.exists(heat_path) or not os.path.exists(gt_path):
            print(f"Skipping {case_id}, missing files")
            continue

        heat = np.load(heat_path)
        gt = np.load(gt_path)

        # load original slice from dataset
        img_path = os.path.join(args.data_dir, case_id, f"{case_id}_flair.nii.gz")
        mask_path = os.path.join(args.data_dir, case_id, f"{case_id}_seg.nii.gz")
        vol = utils.load_nifti_image(img_path)
        mvol = utils.load_nifti_image(mask_path)
        if vol.ndim == 4:
            vol = vol[0]
        best_idx = utils.get_best_slice(mvol)
        img_slice = vol[:, :, best_idx]
        img_norm = utils.normalize_image(img_slice)

        # Overlap@k and IoU@k
        overlaps = {}
        ious = {}
        for k in [0.01, 0.05, 0.1]:
            mask_k = top_k_mask_from_heatmap(heat, k)
            # Overlap = intersection / topk_count
            inter = np.logical_and(mask_k, gt > 0).sum()
            topk_count = mask_k.sum()
            overlap = inter / (topk_count + 1e-10)
            overlaps[f'overlap@{int(k*100)}'] = overlap
            ious[f'iou@{int(k*100)}'] = iou(mask_k, gt > 0)

            # Deletion and Insertion AUC (use gaussian baseline when available)
            del_auc, del_frac, del_scores = deletion_insertion_auc(MODEL, img_norm, heat, lime_analysis.segmentation_prediction_wrapper, DEVICE, mode='deletion', steps=args.steps, baseline_type=args.baseline, gaussian_sigma=args.gaussian_sigma)
            ins_auc, ins_frac, ins_scores = deletion_insertion_auc(MODEL, img_norm, heat, lime_analysis.segmentation_prediction_wrapper, DEVICE, mode='insertion', steps=args.steps, baseline_type=args.baseline, gaussian_sigma=args.gaussian_sigma)

            # save curves
            curves_dir = os.path.join(case_dir, 'curves')
            os.makedirs(curves_dir, exist_ok=True)
            np.save(os.path.join(curves_dir, 'deletion_scores.npy'), del_scores)
            np.save(os.path.join(curves_dir, 'insertion_scores.npy'), ins_scores)
            np.save(os.path.join(curves_dir, 'fractions.npy'), del_frac)

        row = {
            'case_id': case_id,
            **overlaps,
            **ious,
            'deletion_auc': float(del_auc),
            'insertion_auc': float(ins_auc)
        }
        rows.append(row)
        print(f"Evaluated {case_id}: del_auc={del_auc:.4f}, ins_auc={ins_auc:.4f}")

    # write CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    keys = ['case_id', 'overlap@1', 'overlap@5', 'overlap@10', 'iou@1', 'iou@5', 'iou@10', 'deletion_auc', 'insertion_auc']
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in keys})

    print(f"Metrics saved to {args.out_csv}")
