import os
import json
import tqdm
import torch
import click
import numpy as np
from collections import defaultdict
import lightning.pytorch as pl
from pytorchvideo.transforms import Normalize
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from torchvision.transforms._transforms_video import ToTensorVideo
import sklearn.metrics as skm

from .my_seq_dataset import SyntaxDataset
from syntax_pre.seq_model import SyntaxLightningModule


@click.command()
@click.option("-r", "--dataset-root", type=click.Path(exists=True), required=True, help="path to dataset root (e.g. data).")
@click.option("-w", "--weights-root", type=click.Path(exists=True), required=True, help="path to models weights.")
@click.option("--meta-json", type=str, default="all_test.json", show_default=True, help="metadata json path relative to dataset root.")
@click.option("--fold", type=int, default=0, show_default=True, help="fold number in weight names.")
@click.option("-nc", "--num-classes", type=int, default=2, help="num of classes of dataset.")
@click.option("-f", "--frames-per-clip", type=int, default=32, help="frame per clip.")
@click.option("-v", "--video-size", type=click.Tuple([int, int]), default=(256, 256), help="frame per clip.")
@click.option("--max-epochs", type=int, default=30, help="max epochs.")
@click.option("--num-workers", type=int, default=0)
@click.option("--device", type=str, default="cuda", show_default=True, help="device to run on, e.g. cuda, cuda:0, cpu.")
@click.option("--seed", type=int, default=42, help="random seed.")
def main(
    dataset_root,
    weights_root,
    meta_json,
    fold,
    num_classes,
    frames_per_clip,
    video_size,
    max_epochs,
    num_workers,
    device,
    seed,
):
    metrics = defaultdict(list)
    for variant in (
        # "mean", 
        "lstm_mean", 
        # "bert_mean",
        ):
        run_variant(
            dataset_root,
            weights_root,
            meta_json,
            fold,
            variant,
            num_classes,
            frames_per_clip,
            video_size,
            max_epochs,
            num_workers,
            device,
            seed,
            metrics,
        )
    print(metrics)
    print(json.dumps(metrics))


def run_variant(
    dataset_root,
    weights_root,
    meta_json,
    fold,
    variant,
    num_classes,
    frames_per_clip,
    video_size,
    max_epochs,
    num_workers,
    device,
    seed,
    metrics,
):
    print(variant)
    left_bin_prob, left_bin, left_syntax, left_sids = run_variant_artery(
        dataset_root,
        weights_root,
        meta_json,
        fold,
        "left",
        variant,
        num_classes,
        frames_per_clip,
        video_size,
        max_epochs,
        num_workers,
        device,
        seed,
        metrics,
    )

    right_bin_prob, right_bin, right_syntax, right_sids = run_variant_artery(
        dataset_root,
        weights_root,
        meta_json,
        fold,
        "right",
        variant,
        num_classes,
        frames_per_clip,
        video_size,
        max_epochs,
        num_workers,
        device,
        seed,
        metrics,
    )

    left_map = {sid: (l_prob, l_bin, l_syn) for sid, l_prob, l_bin, l_syn in zip(left_sids, left_bin_prob, left_bin, left_syntax)}
    right_map = {sid: (r_prob, r_bin, r_syn) for sid, r_prob, r_bin, r_syn in zip(right_sids, right_bin_prob, right_bin, right_syntax)}

    src_path = os.path.join(dataset_root, meta_json)
    dataset = json.load(open(src_path, "r", encoding="utf-8"))

    syntax_true = []
    syntax_pred = []
    for rec in dataset:
        sid = rec.get("study_uid")
        if sid not in left_map or sid not in right_map:
            continue
        l_prob, l_bin, l_syn = left_map[sid]
        r_prob, r_bin, r_syn = right_map[sid]
        rec["prediction"] = {
            "left_prob": l_prob,
            "left_bin": l_bin,
            "left_syntax": l_syn,
            "right_prob": r_prob,
            "right_bin": r_bin,
            "right_syntax": r_syn,
            "syntax": l_syn + r_syn,
        }
        if rec.get("syntax") is not None:
            syntax_true.append(rec["syntax"])
            syntax_pred.append(l_syn + r_syn)

    if len(syntax_true) > 1:
        r2 = skm.r2_score(syntax_true, syntax_pred)
        bias = np.array(syntax_pred) - np.array(syntax_true)
        metrics[f"{variant}_syntax_r2"].append(r2)
        metrics[f"{variant}_syntax_bias_mean"].append(float(np.mean(bias)))
        metrics[f"{variant}_syntax_bias_median"].append(float(np.median(bias)))
        metrics[f"{variant}_syntax_dev_std"].append(float(np.std(bias)))
        print("SYNTAX R2", r2)
        print("SYNTAX Bias mean", float(np.mean(bias)))
        print("SYNTAX Bias median", float(np.median(bias)))
        print("SYNTAX Deviation STD", float(np.std(bias)))

    base_name = os.path.splitext(os.path.basename(meta_json))[0]
    out_name = f"{base_name}.{variant}.pred.json"
    out_path = os.path.join(dataset_root, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)


def run_variant_artery(
    dataset_root,
    weights_root,
    meta_json,
    fold,
    artery,
    variant,
    num_classes,
    frames_per_clip,
    video_size,
    max_epochs,
    num_workers,
    device,
    seed,
    metrics,
):
    VARIANTS = "mean_out, mean, lstm_mean, lstm_last, gru_mean, gru_last, bert_mean, bert_cls".split(", ")
    print(variant)
    assert variant in VARIANTS

    Artery = artery.capitalize()

    model_path = os.path.join(weights_root, f"{Artery}BinSyntax_R3D_fold{fold:02d}_{variant}_post_best.pt")
    if not os.path.isfile(model_path):
        print(model_path)

    pl.seed_everything(seed)

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    test_transform = T.Compose(
        [
            ToTensorVideo(),
            T.Resize(size=video_size, antialias=True),
            Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    test_set = SyntaxDataset(
        root=dataset_root,
        meta_json=meta_json,
        train=False,
        length=frames_per_clip,
        label=f"syntax_{artery}",
        artery=artery,
        inference=True,
        transform=test_transform,
    )

    test_dataloader = DataLoader(
        test_set,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    model = SyntaxLightningModule(
        num_classes=num_classes,
        lr=1e-5,
        variant=variant,
        weight_decay=0.001,
        max_epochs=max_epochs,
    )

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    if device.startswith("cuda"):
        try:
            torch.cuda.get_device_properties(device)
        except Exception:
            print(f"Requested device '{device}' is not available, falling back to cuda:0")
            device = "cuda:0"
    device = torch.device(device)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    model.eval()

    Y = []
    Y_syntax = []
    P_bin_prob = []
    P_bin = []
    P_syntax = []
    sids = []

    with torch.no_grad():
        for x, [y], [t], [_weight_], [sid] in tqdm.tqdm(test_dataloader):
            if len(x.shape) == 1:
                bin_prob = 0.0
                val_syntax = 0.0
                bin_val = 0
            else:
                x = x.to(device)
                [pred] = model(x)
                bin_logit, val_log = pred
                bin_prob = float(torch.sigmoid(bin_logit).cpu())
                val_syntax = max(0.0, float(torch.exp(val_log).cpu()) - 1)
                bin_val = round(bin_prob)

            y_syntax = max(0, float(torch.exp(t)) - 1)
            Y.append(y)
            Y_syntax.append(y_syntax)
            P_bin_prob.append(bin_prob)
            P_bin.append(bin_val)
            P_syntax.append(val_syntax)
            sids.append(sid)

    m = {}
    try:
        m["auc"] = skm.roc_auc_score(Y, P_bin_prob)
        m["f1"] = skm.f1_score(Y, P_bin, zero_division=0)
        m["acc"] = skm.accuracy_score(Y, P_bin)
    except ValueError as err:
        print(err)

    if len(Y_syntax) > 1:
        m["r2"] = skm.r2_score(Y_syntax, P_syntax)
        bias = np.array(P_syntax) - np.array(Y_syntax)
        m["bias_mean"] = float(np.mean(bias))
        m["bias_median"] = float(np.median(bias))
        m["dev_std"] = float(np.std(bias))

    print()
    print(variant, fold, artery)
    print(json.dumps(m, indent=4))
    for k, v in m.items():
        metrics[f"{variant}_{artery}_{k}"].append(v)

    return P_bin_prob, P_bin, P_syntax, sids


if __name__ == "__main__":
    main()
