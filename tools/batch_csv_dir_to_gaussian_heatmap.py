import os
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================
# 路径配置
# =========================
REPO_ROOT = Path(__file__).resolve().parents[1]
IN_ROOT_DIR = REPO_ROOT / "data" / "results"
OUT_ROOT_DIR = REPO_ROOT / "data" / "gaussian_heatmaps"

CSV_NAME = "df_stenosis.csv"

# =========================
# 参数
# =========================
H, W = 512, 512
TARGET_FRAME = 12

SEVERE_FACTOR = 2.0
NORMALIZE = True

SAVE_NPY = True
SAVE_PNG = True
OVERWRITE = True

# =========================
# 工具函数
# =========================
def parse_box(box_str):
    return ast.literal_eval(box_str)

def select_best_row(group, target_frame):
    idx = (group["frame"] - target_frame).abs().idxmin()
    return group.loc[idx]

def gaussian_2d(H, W, cx, cy, sx, sy):
    x = np.arange(W)
    y = np.arange(H)
    xx, yy = np.meshgrid(x, y)
    return np.exp(
        -(((xx - cx) ** 2) / (2 * sx ** 2)
          + ((yy - cy) ** 2) / (2 * sy ** 2))
    )

def gaussian_from_box(box, H, W, scale=0.3):
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    return gaussian_2d(H, W, cx, cy, bw * scale, bh * scale)

# =========================
# 单 CSV → 热图
# =========================
def generate_heatmap(csv_path):
    df = pd.read_csv(csv_path)
    heatmap = np.zeros((H, W), dtype=np.float32)

    for _, group in df.groupby("source_csv"):
        row = select_best_row(group, TARGET_FRAME)

        weight = float(row["percent_stenosis"])
        if float(row.get("severe_stenosis", 0)) == 1:
            weight *= SEVERE_FACTOR

        heatmap += gaussian_from_box(
            parse_box(row["box_resized"]), H, W
        ) * weight

    if NORMALIZE:
        heatmap -= heatmap.min()
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

    return heatmap

# =========================
# 主流程
# =========================
def main():
    results_dirs = sorted(
        d for d in os.listdir(IN_ROOT_DIR)
        if d.startswith("results_")
    )

    # === 先统计总 case 数（所有 results_* 下）===
    all_cases = []
    for res in results_dirs:
        res_dir = os.path.join(IN_ROOT_DIR, res)
        for case in os.listdir(res_dir):
            case_dir = os.path.join(res_dir, case)
            if os.path.isdir(case_dir):
                all_cases.append((res, case))

    print(f"[INFO] Total cases: {len(all_cases)}")

    # === case 级进度条 ===
    for res, case in tqdm(all_cases, desc="Processing cases", ncols=100):
        in_case_dir = os.path.join(IN_ROOT_DIR, res, case)
        out_case_dir = os.path.join(OUT_ROOT_DIR, res, case)

        for series in os.listdir(in_case_dir):
            in_series_dir = os.path.join(in_case_dir, series)
            csv_path = os.path.join(in_series_dir, CSV_NAME)

            if not os.path.exists(csv_path):
                continue

            out_series_dir = os.path.join(out_case_dir, series)
            os.makedirs(out_series_dir, exist_ok=True)

            out_npy = os.path.join(out_series_dir, "gaussian_heatmap.npy")
            out_png = os.path.join(out_series_dir, "gaussian_heatmap.png")

            if not OVERWRITE and os.path.exists(out_npy):
                continue

            try:
                heatmap = generate_heatmap(csv_path)

                if SAVE_NPY:
                    np.save(out_npy, heatmap)

                if SAVE_PNG:
                    plt.imsave(out_png, heatmap, cmap="gray")

            except Exception as e:
                print(f"[ERROR] {res}/{case}/{series}: {e}")

    print("[DONE] All cases processed.")

# =========================
if __name__ == "__main__":
    main()
