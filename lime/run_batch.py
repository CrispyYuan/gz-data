#!/usr/bin/env python3
"""
简单的批量运行脚本：对 BraTS 数据集的前 N 个病例调用 lime_analysis.py
将每个病例的结果保存在 ./results/batch/<case_id>/ 下。
"""
import os
import subprocess
from glob import glob

DATA_DIR = os.path.abspath("../pattern_recognition/dataset/BraTS2021_Data")
MODEL_PATH = os.path.abspath("../pattern_recognition/checkpoints/nnunet/brats/nnunet_best.pth")
LIME_SCRIPT = os.path.abspath("./lime_analysis.py")
OUTPUT_ROOT = os.path.abspath("./results/batch")

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# 收集病例文件夹（假设每个病例是以 BraTS2021_ 开头的文件夹）
cases = sorted(glob(os.path.join(DATA_DIR, "BraTS2021_*")))
print(f"Found {len(cases)} cases, will process up to first 5 by default.")

# 可以通过环境变量或修改这里改变数量
N = 5
cases = cases[:N]

for case_path in cases:
    case_id = os.path.basename(case_path)
    flair = os.path.join(case_path, f"{case_id}_flair.nii.gz")
    seg = os.path.join(case_path, f"{case_id}_seg.nii.gz")
    out_dir = os.path.join(OUTPUT_ROOT, case_id)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "python3", LIME_SCRIPT,
        "--img_path", flair,
        "--mask_path", seg,
        "--model_path", MODEL_PATH,
        "--output_dir", out_dir,
        "--case_id", case_id
    ]

    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=os.path.dirname(LIME_SCRIPT))
    print(f"Case {case_id} finished with returncode {res.returncode}\n")

print("Batch run completed.")
