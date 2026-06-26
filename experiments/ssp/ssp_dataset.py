import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split

class CaseHeatmapDataset(Dataset):
    def __init__(self, heatmap_root, cases, labels, transform=None):
        """
        heatmap_root:
            data/gaussian_heatmaps/results_x/<study_uid>/
        cases:
            list of study_uid
        labels:
            dict: {study_uid: syntax_score}
        """
        self.heatmap_root = heatmap_root
        self.labels = labels
        self.transform = transform

        self.cases = []
        for case in cases:
            case_dir = os.path.join(heatmap_root, case)
            if not os.path.isdir(case_dir):
                continue
            if self._has_heatmaps(case_dir):
                self.cases.append(case)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        case_dir = os.path.join(self.heatmap_root, case)

        imgs = []
        for series in os.listdir(case_dir):
            npy_path = os.path.join(case_dir, series, "gaussian_heatmap.npy")
            if not os.path.exists(npy_path):
                continue

            img = np.load(npy_path).astype(np.float32)
            if img.ndim == 3:
                img = img[0]
            if img.max() > 1.0:
                img = img / 255.0
            img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]

            if self.transform:
                img = self.transform(img)

            imgs.append(img)

        imgs = torch.stack(imgs)  # [N, 1, H, W]

        y_reg = float(self.labels[case])
        y_cls = 1.0 if y_reg > 0 else 0.0

        return imgs, torch.tensor(y_reg), torch.tensor(y_cls)

    @staticmethod
    def _has_heatmaps(case_dir):
        for series in os.listdir(case_dir):
            if os.path.exists(os.path.join(case_dir, series, "gaussian_heatmap.npy")):
                return True
        return False
    


def prepare_splits(json_path, heatmap_root, test_size=0.2, val_size=0.1,
                   random_state=42):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    labels = {os.path.join(item["videos"][0]["path"].split("/")[0],item["study_uid"]): float(item["syntax"]) for item in data}

    cases = []
    for case in labels:
        case_dir = os.path.join(heatmap_root, case)
        # print(case_dir)
        if os.path.isdir(case_dir) and CaseHeatmapDataset._has_heatmaps(case_dir):
            cases.append(case)
    print(f"[INFO] Total cases with heatmaps: {len(cases)}")


    train_cases, test_cases = train_test_split(
        cases, test_size=test_size, random_state=random_state
    )

    train_cases, val_cases = train_test_split(
        train_cases, test_size=val_size, random_state=random_state
    )

    return train_cases, val_cases, test_cases, labels

def collate_case(batch):
    imgs, y_reg, y_cls = zip(*batch)

    max_len = max(x.size(0) for x in imgs)
    batch_size = len(imgs)
    _, _, h, w = imgs[0].shape

    padded = imgs[0].new_zeros((batch_size, max_len, 1, h, w))
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, x in enumerate(imgs):
        n = x.size(0)
        padded[i, :n] = x
        mask[i, :n] = True

    return padded, mask, torch.stack(y_reg), torch.stack(y_cls)