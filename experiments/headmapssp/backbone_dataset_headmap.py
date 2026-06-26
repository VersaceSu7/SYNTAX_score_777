import os
import json
import numpy as np
import torch

from typing import Callable, Optional, Tuple

from torch import Tensor
from torch.utils.data import Dataset


class SyntaxDataset(Dataset):
    def __init__(
        self,
        root: str,
        meta_json: str,
        train: bool,
        length: int,
        label: str,
        artery: str,
        heatmap_root: Optional[str] = None,
        video_root: Optional[str] = None,
        val_ratio: float = 0.2,
        seed: int = 42,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.train = train
        self.length = length
        self.label = label
        self.artery = artery
        self.heatmap_root = heatmap_root
        self.video_root = video_root if video_root is not None else root
        self.val_ratio = val_ratio
        self.seed = seed
        self.transform = transform

        meta_path = os.path.join(root, meta_json)
        with open(meta_path, "r", encoding="utf-8") as f:
            studies = json.load(f)

        if artery not in ("left", "right"):
            raise ValueError(f"Unknown artery '{artery}'")

        artery_tag = "LCA" if artery == "left" else "RCA"

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
            for video in study.get("videos", []):
                if video.get("artery") != artery_tag:
                    continue
                records.append(
                    {
                        "path": video["path"],
                        "series_uid": os.path.basename(video["path"]).replace(".npy", ""),
                        "study_uid": study.get("study_uid"),
                        "syntax": study.get("syntax"),
                        "syntax_left": study.get("syntax_left"),
                        "syntax_right": study.get("syntax_right"),
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
        if not self.heatmap_root:
            return None
        prefix = video_path.split("/")[0] if video_path else ""
        candidates = [
            os.path.join(self.heatmap_root, f"results_{prefix}", study_uid),
            os.path.join(self.heatmap_root, prefix, study_uid),
            os.path.join(self.heatmap_root, f"{prefix}{study_uid}"),
            os.path.join(self.heatmap_root, study_uid),
        ]

        for path in candidates:
            if os.path.isdir(path):
                return path

        if os.path.isdir(self.heatmap_root):
            for entry in os.listdir(self.heatmap_root):
                if entry.startswith("results_"):
                    path = os.path.join(self.heatmap_root, entry, study_uid)
                    if os.path.isdir(path):
                        return path
        return None

    def _heatmap_path(self, study_uid: str, video_path: str) -> Optional[str]:
        if not self.heatmap_root:
            return None
        case_dir = self._resolve_case_dir(study_uid, video_path)
        if case_dir is None:
            return None
        series_uid = os.path.basename(video_path).replace(".npy", "")
        return os.path.join(case_dir, series_uid, "gaussian_heatmap.npy")

    def _load_heatmap(self, npy_path: Optional[str]) -> Tensor:
        if npy_path is None or not os.path.exists(npy_path):
            return torch.zeros((1, 512, 512), dtype=torch.float32)
        img = np.load(npy_path).astype(np.float32)
        if img.ndim == 3:
            img = img[0]
        if img.max() > 1.0:
            img = img / 255.0
        return torch.from_numpy(img).unsqueeze(0)

    def _video_path(self, study_uid: str, video_path: str) -> Optional[str]:
        if not video_path:
            return None

        direct = os.path.join(self.video_root, video_path)
        if os.path.exists(direct):
            return direct

        parts = video_path.split("/")
        if len(parts) >= 2:
            prefix = parts[0]
            sid = parts[1]
            filename = os.path.basename(video_path)
            combined_dir = f"{prefix}{sid}"
            combined_path = os.path.join(self.video_root, combined_dir, filename)
            if os.path.exists(combined_path):
                return combined_path
            alt_path = os.path.join(self.video_root, prefix, sid, filename)
            if os.path.exists(alt_path):
                return alt_path

        return None

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        if self.negative_dataset:
            if idx % 2 == 0:
                idx = idx // 2
                rec = self.dataset[idx]
            else:
                idx = torch.randint(low=0, high=len(self.negative_dataset), size=(1,))
                rec = self.negative_dataset[idx]
        else:
            rec = self.dataset[idx]

        path = rec["path"]
        weight = rec["weight"]
        full_path = self._video_path(rec.get("study_uid"), path)
        if full_path is None:
            raise FileNotFoundError(f"Video not found for path: {path}")
        video = self._load_video(full_path)
        label = torch.tensor([int((rec.get(self.label) or 0) > 0)], dtype=torch.float32)

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

        heatmap = self._load_heatmap(self._heatmap_path(rec.get("study_uid"), path))

        return video, heatmap, label, weight, path
