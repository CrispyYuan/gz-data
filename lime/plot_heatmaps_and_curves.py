#!/usr/bin/env python3
"""
绘制热力图与 Deletion/Insertion 曲线并保存 PNG。
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import importlib.util
spec = importlib.util.spec_from_file_location('lime_analysis', os.path.join(os.path.dirname(__file__), 'lime_analysis.py'))
lime_analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lime_analysis)
import utils
import torch


def compute_deletion_insertion(segmentation_fn, img, heatmap, device, steps=50, baseline_type='mean', gaussian_sigma=2.0):
    H, W = img.shape
    total = H * W
    idxs = np.argsort(heatmap.flatten())[::-1]
    fractions = np.linspace(0, 1.0, steps)
    del_scores = []
    ins_scores = []

    if baseline_type == 'gaussian':
        try:
            from scipy.ndimage import gaussian_filter
            baseline = gaussian_filter(img, sigma=gaussian_sigma)
        except Exception:
            baseline = np.full_like(img, np.mean(img))
    else:
        baseline = np.full_like(img, np.mean(img))

    for f in fractions:
        k = int(total * f)
        mask = np.zeros(total, dtype=bool)
        if k > 0:
            mask[idxs[:k]] = True
        mask2d = mask.reshape(H, W)

        # deletion: replace top-k with baseline
        pert_del = img.copy()
        pert_del[mask2d] = np.mean(img)
        res_del = segmentation_fn(pert_del)
        score_del = float(res_del[0, 1]) if (isinstance(res_del, np.ndarray) and res_del.ndim==2) else float(res_del)
        del_scores.append(score_del)

        # insertion: start from baseline, insert top-k from original
        pert_ins = baseline.copy()
        pert_ins[mask2d] = img[mask2d]
        res_ins = segmentation_fn(pert_ins)
        score_ins = float(res_ins[0, 1]) if (isinstance(res_ins, np.ndarray) and res_ins.ndim==2) else float(res_ins)
        ins_scores.append(score_ins)

    return fractions, np.array(del_scores), np.array(ins_scores)


def plot_case(case_dir, data_dir, model, device, out_dir, steps=50):
    case_id = os.path.basename(case_dir)
    heat_path = os.path.join(case_dir, f"{case_id}_lime_heatmap.npy")
    gt_path = os.path.join(case_dir, f"{case_id}_gt_mask.npy")
    if not os.path.exists(heat_path) or not os.path.exists(gt_path):
        print(f"Skipping {case_id}, missing files")
        return

    heat = np.load(heat_path)
    gt = np.load(gt_path)

    # load original slice
    img_path = os.path.join(data_dir, case_id, f"{case_id}_flair.nii.gz")
    vol = utils.load_nifti_image(img_path)
    if vol.ndim == 4:
        vol = vol[0]
    best_idx = utils.get_best_slice(gt)
    img_slice = vol[:, :, best_idx]
    img_norm = utils.normalize_image(img_slice)

    # ensure model in lime_analysis globals
    lime_analysis.MODEL = model
    lime_analysis.DEVICE = device

    # plot heatmap overlay
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(img_norm, cmap='gray')
    ax[0].set_title('Original')
    ax[0].axis('off')

    im = ax[1].imshow(heat, cmap='RdBu_r')
    ax[1].set_title('LIME Heatmap')
    ax[1].axis('off')
    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

    ax[2].imshow(img_norm, cmap='gray')
    ax[2].imshow(gt, cmap='jet', alpha=0.5)
    ax[2].set_title('GT Overlay')
    ax[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{case_id}_heatmap_panel.png"), dpi=200)
    plt.close()

    # try to load existing saved curves
    curves_dir = os.path.join(case_dir, 'curves')
    if os.path.exists(curves_dir):
        try:
            del_scores = np.load(os.path.join(curves_dir, 'deletion_scores.npy'))
            ins_scores = np.load(os.path.join(curves_dir, 'insertion_scores.npy'))
            fractions = np.load(os.path.join(curves_dir, 'fractions.npy'))
        except Exception:
            fractions, del_scores, ins_scores = compute_deletion_insertion(lime_analysis.segmentation_prediction_wrapper, img_norm, heat, device, steps=steps, baseline_type='gaussian', gaussian_sigma=1.5)
            os.makedirs(curves_dir, exist_ok=True)
            np.save(os.path.join(curves_dir, 'deletion_scores.npy'), del_scores)
            np.save(os.path.join(curves_dir, 'insertion_scores.npy'), ins_scores)
            np.save(os.path.join(curves_dir, 'fractions.npy'), fractions)
    else:
        fractions, del_scores, ins_scores = compute_deletion_insertion(lime_analysis.segmentation_prediction_wrapper, img_norm, heat, device, steps=steps, baseline_type='gaussian', gaussian_sigma=1.5)
        os.makedirs(curves_dir, exist_ok=True)
        np.save(os.path.join(curves_dir, 'deletion_scores.npy'), del_scores)
        np.save(os.path.join(curves_dir, 'insertion_scores.npy'), ins_scores)
        np.save(os.path.join(curves_dir, 'fractions.npy'), fractions)

    # normalize by initial score
    del_norm = del_scores / (del_scores[0] + 1e-8)
    ins_norm = ins_scores / (ins_scores.max() + 1e-8)

    plt.figure(figsize=(6, 4))
    plt.plot(fractions, del_norm, label='Deletion (norm)')
    plt.plot(fractions, ins_norm, label='Insertion (norm)')
    plt.xlabel('Fraction of pixels changed')
    plt.ylabel('Normalized score')
    plt.title(f'{case_id} Deletion/Insertion Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f"{case_id}_del_ins_curves.png"), dpi=200)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--out_root', type=str, default='./results/plots')
    parser.add_argument('--steps', type=int, default=25)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    DEVICE = torch.device(args.device)
    MODEL = lime_analysis.load_trained_model(args.model_path, DEVICE)
    lime_analysis.MODEL = MODEL
    lime_analysis.DEVICE = DEVICE

    case_dirs = sorted(glob(os.path.join(args.results_dir, '*')))
    os.makedirs(args.out_root, exist_ok=True)
    for case_dir in case_dirs:
        case_id = os.path.basename(case_dir)
        out_dir = os.path.join(args.out_root, case_id)
        print(f'Processing {case_id}...')
        plot_case(case_dir, args.data_dir, MODEL, DEVICE, out_dir, steps=args.steps)
    print('All done.')
