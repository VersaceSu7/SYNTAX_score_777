import os
import json
import numpy as np
import torch
import argparse
import math

from typing import Callable, Optional, Tuple

from torch import Tensor
from torch.utils.data import Dataset

class SyntaxFusionDataset(Dataset):
    def __init__(
        self,
        root: str,
        meta_json: str,
        train: bool,
        length: int,
        label: str,
        artery: str,
        heatmap_root: str,
        val_ratio: float = 0.2,
        seed: int = 42,
        inference: bool = False,
        transform: Optional[Callable] = None,
        heatmap_size: Tuple[int, int] = (512, 512),
    ) -> None:
        self.root = root
        self.train = train
        self.length = length
        self.label = label
        self.artery = artery
        self.heatmap_root = heatmap_root
        self.val_ratio = val_ratio
        self.seed = seed
        self.inference = inference
        self.transform = transform
        self.heatmap_size = heatmap_size
        self._case_dir_cache = {}

        meta_path = os.path.join(root, meta_json)
        with open(meta_path, "r", encoding="utf-8") as f:
            studies = json.load(f)

        if artery not in ("left", "right"):
            raise ValueError(f"Unknown artery '{artery}'")

        artery_tag = "LCA" if artery == "left" else "RCA"

        if inference:
            selected = studies
        else:
            study_ids = [rec.get("study_uid") for rec in studies]
            uniq_ids = sorted({sid for sid in study_ids if sid is not None})
            rng = np.random.RandomState(seed)
            rng.shuffle(uniq_ids)
            val_count = int(len(uniq_ids) * val_ratio)
            val_ids = set(uniq_ids[:val_count])

            if train:
                selected = [rec for rec in studies if rec.get("study_uid") not in val_ids]
            else:
                selected = [rec for rec in studies if rec.get("study_uid") in val_ids]

        records = []
        for study in selected:
            videos = [v for v in study.get("videos", []) if v.get("artery") == artery_tag]
            if not inference:
                videos = [v for v in videos if self._heatmap_exists(study.get("study_uid"), v)]

            if not inference and len(videos) == 0:
                continue
            records.append(
                {
                    "study_uid": study.get("study_uid"),
                    "syntax": study.get("syntax"),
                    "syntax_left": study.get("syntax_left"),
                    "syntax_right": study.get("syntax_right"),
                    "videos": videos,
                }
            )

        if train:
            self.dataset = [rec for rec in records if (rec.get(self.label) or 0) > 0]
            self.negative_dataset = [rec for rec in records if (rec.get(self.label) or 0) == 0]

            for rec in self.dataset:
                rec["weight"] = 1.0
            for rec in self.negative_dataset:
                rec["weight"] = 1.0
        else:
            self.dataset = records
            self.negative_dataset = None
            for rec in self.dataset:
                rec["weight"] = 1.0

    def __len__(self):
        coef = 2 if self.negative_dataset else 1
        return coef * len(self.dataset)

    def _load_video(self, full_path: str) -> np.ndarray:
        video = np.load(full_path)
        if video.ndim == 4 and video.shape[-1] == 1:
            video = video[..., 0]
        return video

    def _resolve_case_dir(self, study_uid: str, video_path: str) -> Optional[str]:
        prefix = video_path.split("/")[0] if video_path else ""
        cache_key = (study_uid, prefix)
        if cache_key in self._case_dir_cache:
            return self._case_dir_cache[cache_key]

        candidates = [
            os.path.join(self.heatmap_root, f"results_{prefix}", study_uid),
            os.path.join(self.heatmap_root, prefix, study_uid),
            os.path.join(self.heatmap_root, study_uid),
        ]

        case_dir = None
        for path in candidates:
            if os.path.isdir(path):
                case_dir = path
                break

        if case_dir is None and os.path.isdir(self.heatmap_root):
            for entry in os.listdir(self.heatmap_root):
                if entry.startswith("results_"):
                    path = os.path.join(self.heatmap_root, entry, study_uid)
                    if os.path.isdir(path):
                        case_dir = path
                        break

        self._case_dir_cache[cache_key] = case_dir
        return case_dir

    def _series_uid_from_path(self, video_path: str) -> Optional[str]:
        if not video_path:
            return None
        base = os.path.basename(video_path)
        if base.endswith(".npy"):
            return base[:-4]
        return None

    def _heatmap_path(self, study_uid: str, video_rec: dict) -> Optional[str]:
        path = video_rec.get("path", "")
        series_uid = self._series_uid_from_path(path)
        if not path or not series_uid:
            return None

        case_dir = self._resolve_case_dir(study_uid, path)
        if case_dir is None:
            return None

        return os.path.join(case_dir, series_uid, "gaussian_heatmap.npy")

    def _heatmap_exists(self, study_uid: str, video_rec: dict) -> bool:
        npy_path = self._heatmap_path(study_uid, video_rec)
        return npy_path is not None and os.path.exists(npy_path)

    def _load_heatmap(self, npy_path: Optional[str]) -> Tensor:
        if npy_path is None or not os.path.exists(npy_path):
            h, w = self.heatmap_size
            return torch.zeros((1, h, w), dtype=torch.float32)

        img = np.load(npy_path).astype(np.float32)
        if img.ndim == 3:
            img = img[0]
        if img.max() > 1.0:
            img = img / 255.0
        return torch.from_numpy(img).unsqueeze(0)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:
        if self.negative_dataset:
            if idx % 2 == 0:
                idx = idx // 2
                rec = self.dataset[idx]
            else:
                idx = torch.randint(low=0, high=len(self.negative_dataset), size=(1,))
                rec = self.negative_dataset[idx]
        else:
            rec = self.dataset[idx]

        weight = rec["weight"]
        sid = rec["study_uid"]
        label = torch.tensor([int((rec.get(self.label) or 0) > 0)], dtype=torch.float32)
        target = torch.tensor([np.log(1.0 + (rec.get(self.label) or 0))], dtype=torch.float32)

        nv = len(rec["videos"])
        if self.inference:
            if nv == 0:
                return 0, 0, label, target, weight, sid
            seq = range(nv)
        else:
            seq = torch.randint(low=0, high=nv, size=(4,))

        videos = []
        heatmaps = []
        for vi in seq:
            vi = int(vi)
            video_rec = rec["videos"][vi]
            path = video_rec["path"]
            full_path = os.path.join(self.root, path)
            video = self._load_video(full_path)

            while len(video) < self.length:
                video = np.concatenate([video, video])
            t = len(video)
            if self.train:
                begin = torch.randint(low=0, high=t - self.length + 1, size=(1,))
                end = begin + self.length
                video = video[begin:end, :, :]
            else:
                begin = (t - self.length) // 2
                end = begin + self.length
                video = video[begin:end, :, :]

            video = torch.tensor(np.stack([video, video, video], axis=-1))

            if self.transform is not None:
                video = self.transform(video)
            videos.append(video)

            npy_path = self._heatmap_path(sid, video_rec)
            heatmaps.append(self._load_heatmap(npy_path))

        videos = torch.stack(videos, dim=0)
        heatmaps = torch.stack(heatmaps, dim=0)

        return videos, heatmaps, label, target, weight, sid



def main():
    parser = argparse.ArgumentParser(description="Debug SyntaxFusionDataset shapes.")
    parser.add_argument("--root", default="/224010165/Project/rely/dataset/CardioSyntax", help="dataset root (contains npy videos)")
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
    print(f"target shape: {target.shape}, value: {math.e**target}")
    print(f"weight: {weight}")


if __name__ == "__main__":
    main()
