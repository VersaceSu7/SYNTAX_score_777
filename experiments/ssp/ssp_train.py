import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .ssp_dataset import CaseHeatmapDataset, prepare_splits, collate_case
from .sspmodel import SyntaxPredictor
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

def train_epoch(model, loader, optimizer, device, w_cls, w_reg):
    model.train()
    total = 0

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="Train", leave=False)

    for imgs, mask, y_reg, y_cls in iterator:
        imgs = imgs.to(device)
        mask = mask.to(device)
        y_reg, y_cls = y_reg.to(device), y_cls.to(device)
        cls_logit, reg_pred = model(imgs, mask=mask)

        loss_cls = F.binary_cross_entropy_with_logits(cls_logit, y_cls)
        loss_reg = F.l1_loss(reg_pred, y_reg)
        loss = w_cls * loss_cls + w_reg * loss_reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()
        if tqdm is not None:
            iterator.set_postfix(loss=loss.item())

    return total / len(loader)


def _compute_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)

    if y_true.min() == y_true.max():
        return np.nan

    order = np.argsort(y_score)
    y_true = y_true[order]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return np.nan

    ranks = np.arange(1, len(y_true) + 1)
    sum_pos_ranks = np.sum(ranks[y_true == 1])
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    cls_losses, reg_losses = [], []
    all_y_reg, all_y_cls = [], []
    all_reg_pred, all_cls_prob = [], []

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="Eval", leave=False)

    for imgs, mask, y_reg, y_cls in iterator:
        imgs = imgs.to(device)
        mask = mask.to(device)
        y_reg, y_cls = y_reg.to(device), y_cls.to(device)
        cls_logit, reg_pred = model(imgs, mask=mask)

        cls_losses.append(
            F.binary_cross_entropy_with_logits(cls_logit, y_cls).item()
        )
        reg_losses.append(
            F.l1_loss(reg_pred, y_reg).item()
        )

        cls_prob = torch.sigmoid(cls_logit)
        cls_pred = (cls_prob >= 0.1).float()
        reg_pred_adj = reg_pred.clone()
        reg_pred_adj[cls_pred == 0] = 0.0

        all_y_reg.append(y_reg.detach().cpu().numpy())
        all_y_cls.append(y_cls.detach().cpu().numpy())
        all_reg_pred.append(reg_pred_adj.detach().cpu().numpy())
        all_cls_prob.append(cls_prob.detach().cpu().numpy())

        if tqdm is not None:
            iterator.set_postfix(cls=cls_losses[-1], reg=reg_losses[-1])

    y_reg_np = np.concatenate(all_y_reg) if all_y_reg else np.array([])
    y_cls_np = np.concatenate(all_y_cls) if all_y_cls else np.array([])
    reg_pred_np = np.concatenate(all_reg_pred) if all_reg_pred else np.array([])
    cls_prob_np = np.concatenate(all_cls_prob) if all_cls_prob else np.array([])

    bias = reg_pred_np - y_reg_np
    ss_res = np.sum((y_reg_np - reg_pred_np) ** 2)
    ss_tot = np.sum((y_reg_np - np.mean(y_reg_np)) ** 2) if y_reg_np.size > 0 else 0.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    metrics = {
        "loss_cls": float(np.mean(cls_losses)) if cls_losses else np.nan,
        "loss_reg": float(np.mean(reg_losses)) if reg_losses else np.nan,
        "r2": float(r2) if not np.isnan(r2) else np.nan,
        "bias_mean": float(np.mean(bias)) if bias.size > 0 else np.nan,
        "bias_median": float(np.median(bias)) if bias.size > 0 else np.nan,
        "dev_std": float(np.std(bias)) if bias.size > 0 else np.nan,
        "acc": float(np.mean((cls_prob_np >= 0.5) == y_cls_np)) if y_cls_np.size > 0 else np.nan,
        "auc": _compute_auc(y_cls_np, cls_prob_np) if y_cls_np.size > 0 else np.nan,
    }

    return metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_cases, val_cases, test_cases, labels = prepare_splits(
        json_path="data/all.json",
        heatmap_root="/224010165/Project/syntax-score-prediction/data/gaussian_heatmaps"
    )

    train_ds = CaseHeatmapDataset("/224010165/Project/syntax-score-prediction/data/gaussian_heatmaps", train_cases, labels)
    val_ds   = CaseHeatmapDataset("/224010165/Project/syntax-score-prediction/data/gaussian_heatmaps", val_cases, labels)
    test_ds  = CaseHeatmapDataset("/224010165/Project/syntax-score-prediction/data/gaussian_heatmaps", test_cases, labels)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                              collate_fn=collate_case)
    val_loader = DataLoader(val_ds, batch_size=4,
                            collate_fn=collate_case)
    test_loader = DataLoader(test_ds, batch_size=4,
                             collate_fn=collate_case)

    print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
    model = SyntaxPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(30):
        w_cls = 1.0
        w_reg = min(1.0, epoch / 10)

        train_loss = train_epoch(
            model, train_loader, optimizer, device, w_cls, w_reg
        )
        val_metrics = eval_epoch(model, val_loader, device)

        print(
            f"[Epoch {epoch:02d}] "
            f"Train {train_loss:.4f} | "
            f"Val R2 {val_metrics['r2']:.4f} "
            f"Bias(mean) {val_metrics['bias_mean']:.4f} "
            f"Bias(med) {val_metrics['bias_median']:.4f} "
            f"STD {val_metrics['dev_std']:.4f} | "
            f"Acc {val_metrics['acc']:.4f} AUC {val_metrics['auc']:.4f}"
        )

    print("Evaluating on test set...")
    test_metrics = eval_epoch(model, test_loader, device)
    print(
        f"[TEST] "
        f"R2 {test_metrics['r2']:.4f} "
        f"Bias(mean) {test_metrics['bias_mean']:.4f} "
        f"Bias(med) {test_metrics['bias_median']:.4f} "
        f"STD {test_metrics['dev_std']:.4f} | "
        f"Acc {test_metrics['acc']:.4f} AUC {test_metrics['auc']:.4f}"
    )

if __name__ == "__main__":
    main()
