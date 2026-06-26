from typing import Any, Optional, Tuple
import torch
from torch import nn, optim
import torch.nn.functional as F
import lightning.pytorch as pl
import torchvision.models.video as tvmv
import sklearn.metrics as skm


class R3DHeatmapAttention(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = tvmv.r3d_18(weights=tvmv.R3D_18_Weights.DEFAULT)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

        self.attn_alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, heatmap: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, C, T, H, W]
        x = self.model.stem(x)
        x = self.model.layer1(x)

        if heatmap is not None:
            if heatmap.dim() == 4:
                heatmap = heatmap.unsqueeze(2)  # [B, 1, 1, H, W]
            elif heatmap.dim() != 5:
                raise ValueError(f"heatmap must be 4D or 5D, got {heatmap.dim()}D")

            heatmap = heatmap.float()
            heatmap = F.interpolate(
                heatmap,
                size=x.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
            gate = 1.0 + self.attn_alpha * heatmap
            x = x * gate

        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return self.model.fc(x)


class SyntaxLightningModuleHeatmap(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        lr: float,
        weight_decay: float = 0,
        max_epochs: int = None,
        weight_path: str = None,
        save_path: str = None,
    ):
        self.save_hyperparameters()
        super().__init__()
        self.num_classes = num_classes
        self.save_path = save_path

        self.model = R3DHeatmapAttention(num_classes=num_classes)

        self.lr = lr
        self.loss_func = nn.BCEWithLogitsLoss(reduction="none")

        self.weight_path = weight_path
        if weight_path is not None:
            self.model.load_state_dict(torch.load(weight_path))

        self.max_epochs = max_epochs
        self.weight_decay = weight_decay

        self.y_val = []
        self.p_val = []
        self.r_val = []

    def forward(self, x, heatmap: Optional[torch.Tensor] = None):
        return self.model(x, heatmap=heatmap)

    @staticmethod
    def _unpack_batch(batch: Tuple[Any, ...]):
        if len(batch) == 5:
            x, heatmap, y, sample_weight, path = batch
        elif len(batch) == 4:
            x, y, sample_weight, path = batch
            heatmap = None
        else:
            raise ValueError(f"Unexpected batch size: {len(batch)}")
        return x, heatmap, y, sample_weight, path

    def training_step(self, batch, batch_idx):
        x, heatmap, y, sample_weight, path = self._unpack_batch(batch)
        y_hat = self(x, heatmap=heatmap)

        loss = self.loss_func(y_hat, y)
        loss = loss * sample_weight
        loss = loss.mean()

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, heatmap, y, sample_weight, path = self._unpack_batch(batch)
        y_hat = self(x, heatmap=heatmap)

        loss = self.loss_func(y_hat, y)
        loss = loss * sample_weight
        loss = loss.mean()

        y_pred = torch.sigmoid(y_hat)

        self.y_val.append(int(y[..., 0].cpu()))
        self.p_val.append(float(y_pred[..., 0].cpu()))
        self.r_val.append(round(float(y_pred[..., 0].cpu())))

        return loss

    def on_validation_epoch_end(self):
        try:
            self.log("val_roc_auc_art", skm.roc_auc_score(self.y_val, self.p_val), prog_bar=True)
            self.log("val_f1_score_art", skm.f1_score(self.y_val, self.r_val, zero_division=0), prog_bar=True)
            self.log("val_accuracy_art", skm.accuracy_score(self.y_val, self.r_val), prog_bar=True)
        except ValueError as err:
            print(err)
            print("Y_VAL", self.y_val)
            print("P_VAL", self.p_val)
        self.y_val.clear()
        self.p_val.clear()
        self.r_val.clear()
        if self.save_path:
            torch.save(self.model.state_dict(), self.save_path)

    def on_train_epoch_end(self) -> None:
        self.log("lr", self.optimizers().optimizer.param_groups[0]["lr"], on_step=False, on_epoch=True, sync_dist=True)
        if self.save_path:
            torch.save(self.model.state_dict(), self.save_path + ".train")

    def configure_optimizers(self):
        if not self.weight_path:
            params = self.model.model.fc.parameters()
        else:
            params = self.model.parameters()
        optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.max_epochs is not None:
            lr_scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer, max_lr=self.lr, total_steps=self.max_epochs
            )
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, heatmap, y, sample_weight, path = self._unpack_batch(batch)
        y_hat = self(x, heatmap=heatmap)
        y_pred = torch.sigmoid(y_hat)

        return {"y": y, "y_pred": torch.round(y_pred), "y_prob": y_pred}
