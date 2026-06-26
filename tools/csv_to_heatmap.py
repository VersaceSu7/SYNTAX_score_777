import pandas as pd
import numpy as np
import ast
import os
import matplotlib.pyplot as plt


# =========================
# 配置
# =========================
CSV_PATH = "examples/df_stenosis.csv"
OUT_DIR = "examples/heatmap_outputs"
H, W = 512, 512             # heatmap 尺寸
TARGET_FRAME = 12
NORMALIZE = True
SAVE_NPY = True
SAVE_PNG = True


# =========================
# 工具函数
# =========================
def parse_box(box_str):
    return ast.literal_eval(box_str)  # (x1, y1, x2, y2)


def select_best_row(group, target_frame=12):
    idx = (group["frame"] - target_frame).abs().idxmin()
    return group.loc[idx]


def gaussian_2d(H, W, cx, cy, sigma_x, sigma_y):
    """
    生成二维高斯分布
    """
    x = np.arange(W)
    y = np.arange(H)
    xx, yy = np.meshgrid(x, y)

    g = np.exp(
        -(((xx - cx) ** 2) / (2 * sigma_x ** 2)
          + ((yy - cy) ** 2) / (2 * sigma_y ** 2))
    )
    return g


def gaussian_from_box(box, H, W, scale=0.3):
    """
    box -> Gaussian
    scale 控制 sigma 相对 box 尺寸的比例
    """
    x1, y1, x2, y2 = map(float, box)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)

    sigma_x = bw * scale
    sigma_y = bh * scale

    return gaussian_2d(H, W, cx, cy, sigma_x, sigma_y)


# =========================
# 核心函数
# =========================
def generate_gaussian_heatmap(
    csv_path,
    H,
    W,
    target_frame=12,
    normalize=True
):
    df = pd.read_csv(csv_path)

    heatmap = np.zeros((H, W), dtype=np.float32)

    for source_csv, group in df.groupby("source_csv"):
        row = select_best_row(group, target_frame)

        box = parse_box(row["box_resized"])
        stenosis = float(row["percent_stenosis"])

        g = gaussian_from_box(box, H, W)

        heatmap += g * stenosis

    if normalize:
        heatmap -= heatmap.min()
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

    return heatmap


# =========================
# 主流程
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[INFO] Processing {CSV_PATH}")
    heatmap = generate_gaussian_heatmap(
        CSV_PATH,
        H,
        W,
        TARGET_FRAME,
        NORMALIZE
    )

    name = os.path.splitext(os.path.basename(CSV_PATH))[0]

    if SAVE_NPY:
        np.save(os.path.join(OUT_DIR, f"{name}_gaussian_heatmap.npy"), heatmap)

    if SAVE_PNG:
        plt.figure(figsize=(6, 6))
        plt.imshow(heatmap, cmap="gray")
        plt.title("Gaussian Stenosis Heatmap (Grayscale)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{name}_gaussian_heatmap.png"), dpi=200)
        plt.close()

    print("[DONE] Gaussian heatmap generated.")


if __name__ == "__main__":
    main()
