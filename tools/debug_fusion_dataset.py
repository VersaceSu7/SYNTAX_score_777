import argparse
import torch

from experiments.htmap.seq_dataset_htmp import SyntaxFusionDataset


def main():
    parser = argparse.ArgumentParser(description="Debug SyntaxFusionDataset shapes.")
    parser.add_argument("--root", default="data", help="dataset root (contains npy videos)")
    parser.add_argument("--meta", default="all.json", help="metadata json (relative to root)")
    parser.add_argument("--heatmap-root", default="data/gaussian_heatmaps", help="heatmap root")
    parser.add_argument("--artery", default="left", choices=["left", "right"])
    parser.add_argument("--label", default="syntax_left", help="label field")
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    ds = SyntaxFusionDataset(
        root=args.root,
        meta_json=args.meta,
        train=args.train,
        length=args.length,
        label=args.label,
        artery=args.artery,
        heatmap_root=args.heatmap_root,
        inference=args.inference,
    )

    print(f"Dataset size: {len(ds)}")
    sample = ds[args.index]

    videos, heatmaps, label, target, weight, sid = sample
    print(f"study_uid: {sid}")
    print(f"videos shape: {videos.shape}")
    print(f"heatmaps shape: {heatmaps.shape}")
    print(f"label shape: {label.shape}, value: {label}")
    print(f"target shape: {target.shape}, value: {target}")
    print(f"weight: {weight}")


if __name__ == "__main__":
    main()
