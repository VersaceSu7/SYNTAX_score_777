from typing import Any, Optional
import torch
import torch.nn.functional as F
from torch import nn, optim
import lightning.pytorch as pl
import torchvision.models.video as tvmv
import sklearn.metrics as skm
from .backbone_model_heatmap import R3DHeatmapAttention


class SyntaxLightningModuleHeatmap(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        lr: float,
        variant: str,
        weight_decay: float = 0,
        max_epochs: int = None,
        weight_path: str = None,
        save_path: str = None,
        pl_weight_path: str = None,
        **kwargs,
    ):
        self.save_hyperparameters()
        super().__init__()
        self.num_classes = num_classes
        print(f"num_classes: {num_classes}")
        self.save_path = save_path
        self.weight_path = weight_path
        self.lr = lr
        self.loss_clf = nn.BCEWithLogitsLoss(reduction='none')
        self.loss_reg = nn.MSELoss(reduction='none')
        self.variant = variant

        self.best_auc = 0.0

        self.model = R3DHeatmapAttention(num_classes=num_classes)

        # ===== 找到backbone的fc =====
        if hasattr(self.model, "model") and hasattr(self.model.model, "fc"):
            backbone_fc = self.model.model.fc
        else:
            backbone_fc = self.model.fc

        in_features = backbone_fc.in_features

        # ===== 构造与checkpoint一致的fc =====
        if hasattr(self.model, "model") and hasattr(self.model.model, "fc"):
            self.model.model.fc = nn.Linear(in_features, 1)
        else:
            self.model.fc = nn.Linear(in_features, 1)

        # ===== 加载权重 =====
        if weight_path is not None:
            print("Load model weights")
            state_dict = torch.load(weight_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)

        # ===== 再去掉fc =====
        if self.variant != "mean_out":
            if hasattr(self.model, "model") and hasattr(self.model.model, "fc"):
                self.model.model.fc = nn.Identity()
            else:
                self.model.fc = nn.Identity()


        if self.variant == "mean_out":
            pass
        elif self.variant in ("gru_mean", "gru_last"):
            self.rnn = nn.GRU(in_features, in_features // 4, batch_first=True)
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(in_features=in_features // 4, out_features=num_classes, bias=True)
        elif self.variant in ("lstm_mean", "lstm_last"):
            self.lstm = nn.LSTM(
                input_size=in_features, hidden_size=in_features // 4, proj_size=num_classes, batch_first=True
            )
        elif self.variant == "mean":
            self.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
        elif self.variant in ("bert_mean", "bert_cls"):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=in_features, nhead=4, batch_first=True, dim_feedforward=in_features // 4
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
        else:
            raise ValueError(f"Unknown model variant {self.variant}")

        if pl_weight_path is not None:
            print("Load LightningModule weights")
            pl_state_dict = torch.load(pl_weight_path)["state_dict"]
            self.load_weights(pl_state_dict, self.model, "model")
            if self.variant == "mean_out":
                pass
            elif self.variant in ("gru_mean", "gru_last"):
                self.load_weights(pl_state_dict, self.rnn, "rnn")
                self.load_weights(pl_state_dict, self.fc, "fc")
            elif self.variant in ("lstm_mean", "lstm_last"):
                self.load_weights(pl_state_dict, self.lstm, "lstm")
            elif self.variant == "mean":
                self.load_weights(pl_state_dict, self.fc, "fc")
            elif self.variant in ("bert_mean", "bert_cls"):
                self.load_weights(pl_state_dict, self.encoder, "encoder")
                self.load_weights(pl_state_dict, self.fc, "fc")
            else:
                raise ValueError(f"Unknown model variant {self.variant}")

        self.max_epochs = max_epochs
        self.weight_decay = weight_decay

        self.y_val = []
        self.p_val = []
        self.r_val = []
        self.ty_val = []
        self.tp_val = []

    def load_weights(self, pl_state_dict, model, prefix):
        model_state_dict = {}
        for key, value in pl_state_dict.items():
            if key.startswith(f"{prefix}."):
                new_key = key.split(".", 1)[1]
                assert new_key not in model_state_dict
                model_state_dict[new_key] = value
        model.load_state_dict(model_state_dict)

    def forward(self, x, heatmap: Optional[torch.Tensor] = None):
        batch_seq_shape = x.shape[0:2]
        x = torch.flatten(x, start_dim=0, end_dim=1)
        if heatmap is not None:
            heatmap = torch.flatten(heatmap, start_dim=0, end_dim=1)
        x = self.model(x, heatmap=heatmap)
        x = torch.unflatten(x, 0, batch_seq_shape)

        if self.variant == "mean_out":
            x = torch.mean(x, dim=1)
        elif self.variant in ("gru_mean", "gru_last"):
            _all_outs_, [_last_out_] = self.rnn(x)
            if self.variant == "gru_mean":
                x = torch.mean(_all_outs_, dim=1)
            else:
                x = _last_out_
            x = self.dropout(x)
            x = self.fc(x)
        elif self.variant in ("lstm_mean", "lstm_last"):
            _all_outs_, (_last_out_, _last_state_) = self.lstm(x)
            if self.variant == "lstm_mean":
                x = torch.mean(_all_outs_, dim=1)
            else:
                x = _last_out_
        elif self.variant == "mean":
            x = torch.mean(x, dim=1)
            x = self.fc(x)
        elif self.variant in ("bert_mean", "bert_cls"):
            if self.variant == "bert_cls":
                x = F.pad(x, (0, 0, 1, 0), "constant", 0)
            x = self.encoder(x)
            if self.variant == "bert_mean":
                x = torch.mean(x, dim=1)
            else:
                x = x[:, 0, :]
            x = self.dropout(x)
            x = self.fc(x)
        else:
            raise ValueError(f"Unknown model variant {self.variant}")

        return x

    @staticmethod
    def _unpack_batch(batch):
        if len(batch) == 6:
            x, heatmap, y, target, sample_weight, path = batch
        else:
            x, y, target, sample_weight, path = batch
            heatmap = None
        return x, heatmap, y, target, sample_weight, path

    def training_step(self, batch, batch_idx):
        x, heatmap, y, target, sample_weight, path = self._unpack_batch(batch)
        y_hat = self(x, heatmap=heatmap)
        yp_clf = y_hat[:, 0:1]
        yp_reg = y_hat[:, 1:]

        clf_loss = self.loss_clf(yp_clf, y)
        clf_loss = clf_loss * sample_weight
        clf_loss = clf_loss.mean()

        reg_loss = self.loss_reg(yp_reg, target)
        reg_loss = reg_loss * sample_weight
        reg_loss = reg_loss.mean()

        loss = clf_loss + reg_loss

        y_pred = torch.sigmoid(yp_clf)
        y_bin = torch.round(y.cpu().detach()).int()
        y_pred_bin = torch.round(y_pred.cpu().detach()).int()

        self.log("train_loss", clf_loss, prog_bar=True)
        self.log("train_f1", skm.f1_score(y_bin, y_pred_bin, zero_division=0), prog_bar=True)
        self.log("train_acc", skm.accuracy_score(y_bin, y_pred_bin), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, heatmap, y, target, sample_weight, path = self._unpack_batch(batch)
        y_hat = self(x, heatmap=heatmap)
        yp_clf = y_hat[:, 0:1]
        yp_reg = y_hat[:, 1:]

        loss = self.loss_clf(yp_clf, y)
        loss = loss * sample_weight
        loss = loss.mean()

        y_pred = torch.sigmoid(yp_clf)

        self.y_val.append(int(y[..., 0].cpu()))
        self.p_val.append(float(y_pred[..., 0].cpu()))
        self.r_val.append(round(float(y_pred[..., 0].cpu())))

        self.ty_val.append(float(target[..., 0].cpu()))
        self.tp_val.append(float(yp_reg[..., 0].cpu()))

        return loss

    def on_validation_epoch_end(self):
        try:
            auc = skm.roc_auc_score(self.y_val, self.p_val)
            f1 = skm.f1_score(self.y_val, self.r_val, zero_division=0)
            acc = skm.accuracy_score(self.y_val, self.r_val)
            self.log("val_auc", auc, prog_bar=True)
            self.log("val_f1", f1, prog_bar=True)
            self.log("val_acc", acc, prog_bar=True)

            rmse = skm.mean_squared_error(self.ty_val, self.tp_val, squared=False)
            self.log("val_rmse", rmse, prog_bar=True)

            if self.save_path and auc > self.best_auc:
                torch.save(self.state_dict(), self.save_path)
                self.best_auc = auc
        except ValueError as err:
            print(err)
            print("Y_VAL", self.y_val)
            print("P_VAL", self.p_val)
        self.y_val.clear()
        self.p_val.clear()
        self.r_val.clear()
        self.ty_val.clear()
        self.tp_val.clear()

    def on_train_epoch_end(self) -> None:
        self.log("lr", self.optimizers().optimizer.param_groups[0]["lr"], on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        if self.weight_path:
            if self.variant == "mean_out":
                ms = [self.model.fc]
            elif self.variant in ("gru_mean", "gru_last"):
                ms = [self.rnn, self.fc]
            elif self.variant in ("lstm_mean", "lstm_last"):
                ms = [self.lstm]
            elif self.variant == "mean":
                ms = [self.fc]
            elif self.variant in ("bert_mean", "bert_cls"):
                ms = [self.encoder, self.fc]

            params = []
            for m in ms:
                for p in m.parameters():
                    params.append(p)
        else:
            params = self.parameters()

        optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.max_epochs is not None:
            lr_scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer, max_lr=self.lr, total_steps=self.max_epochs
            )
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, heatmap, y, target, sample_weight, path = self._unpack_batch(batch)
        y_hat = self(x, heatmap=heatmap)
        yp_clf = y_hat[:, 0:1]
        y_pred = torch.sigmoid(yp_clf)

        return {"y": y, "y_pred": torch.round(y_pred), "y_prob": y_pred}
