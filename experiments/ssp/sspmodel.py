import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class SyntaxPredictor(nn.Module):
    def __init__(self, feat_dim=128, lstm_hidden=128):
        super().__init__()

        self.encoder = HeatmapEncoder(feat_dim)

        # 新增 LSTM
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        final_dim = lstm_hidden * 2

        self.cls_head = nn.Linear(final_dim, 1)
        self.reg_head = nn.Linear(final_dim, 1)

    def forward(self, x, mask=None):
        """
        x: [B, N, 1, H, W]
        mask: [B, N] (True for valid)
        """
        B, N = x.shape[:2]

        # encode heatmaps
        x = x.view(B * N, 1, x.size(-2), x.size(-1))
        feats = self.encoder(x)          # [B*N, D]
        feats = feats.view(B, N, -1)     # [B, N, D]

        # LSTM
        lstm_out, _ = self.lstm(feats)   # [B, N, 2H]

        # ---- mean pooling with mask ----
        if mask is None:
            case_feat = lstm_out.mean(dim=1)
        else:
            mask_f = mask.float().unsqueeze(-1)
            denom = mask_f.sum(dim=1).clamp_min(1.0)
            case_feat = (lstm_out * mask_f).sum(dim=1) / denom

        cls_logit = self.cls_head(case_feat).squeeze(-1)
        reg_pred = self.reg_head(case_feat).squeeze(-1)

        return cls_logit, reg_pred



bce_loss = nn.BCEWithLogitsLoss()
reg_loss = nn.L1Loss()

def compute_loss(cls_logit, reg_pred, y_cls, y_reg,
                 w_cls=1.0, w_reg=0.2):
    loss_cls = bce_loss(cls_logit, y_cls)
    loss_reg = reg_loss(reg_pred, y_reg)
    return w_cls * loss_cls + w_reg * loss_reg, loss_cls, loss_reg