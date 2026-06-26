import click
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from pytorchvideo.transforms import Normalize, Permute, RandAugment
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from torchvision.transforms._transforms_video import ToTensorVideo

from .backbone_dataset_htmp import SyntaxDataset
from .backbone_model_htmp import SyntaxLightningModuleHeatmap


@click.command()
@click.option("-r", "--dataset-root", type=click.Path(exists=True), required=True, help="path to dataset root (e.g. data).")
@click.option("--meta-json", type=str, default="all_train.json", show_default=True, help="metadata json path relative to dataset root.")
@click.option("--heatmap-root", type=click.Path(), default="data/gaussian_heatmaps", show_default=True, help="gaussian heatmaps root.")
@click.option("--video-root", type=click.Path(), default=None, show_default=True, help="video npy root (defaults to dataset-root).")
@click.option("--fold", type=int, default=0, show_default=True, help="reserved for compatibility; not used for split.")
@click.option("-a", "--artery", type=str, required=True, help="left or right artery.")
@click.option("-nc", "--num-classes", type=int, default=1, help="num of classes of dataset.")
@click.option("-b", "--batch-size", type=int, default=8, help="batch size.")
@click.option("-f", "--frames-per-clip", type=int, default=32, help="frame per clip.")
@click.option("-v", "--video-size", type=click.Tuple([int, int]), default=(256, 256), help="frame per clip.")
@click.option("--max-epochs", type=int, default=30, help="max epochs.")
@click.option("--num-workers", type=int, default=0)
@click.option("--val-ratio", type=float, default=0.2, show_default=True, help="validation ratio by study id.")
@click.option("--fast-dev-run", type=bool, is_flag=True, show_default=True, default=False)
@click.option("--seed", type=int, default=42, help="random seed.")
def main(
    dataset_root,
    meta_json,
    heatmap_root,
    video_root,
    fold,
    artery,
    num_classes,
    batch_size,
    frames_per_clip,
    video_size,
    max_epochs,
    num_workers,
    val_ratio,
    fast_dev_run,
    seed,
):
    print(video_size)
    Artery = artery.capitalize()
    if artery not in ("left", "right"):
        raise ValueError(f"Unknown artery '{artery}'")

    pl.seed_everything(seed)

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = T.Compose(
        [
            ToTensorVideo(),
            Permute(dims=[1, 0, 2, 3]),
            RandAugment(magnitude=10, num_layers=2),
            Permute(dims=[1, 0, 2, 3]),
            T.RandomChoice([
                T.Resize(size=video_size, antialias=False),
                T.Resize(size=video_size, antialias=True),
            ]),
            Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    test_transform = T.Compose(
        [
            ToTensorVideo(),
            T.Resize(size=video_size, antialias=True),
            Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    train_set = SyntaxDataset(
        root=dataset_root,
        meta_json=meta_json,
        train=True,
        length=frames_per_clip,
        label=f"syntax_{artery}",
        artery=artery,
        heatmap_root=heatmap_root,
        video_root=video_root,
        val_ratio=val_ratio,
        seed=seed,
        transform=train_transform,
    )

    val_set = SyntaxDataset(
        root=dataset_root,
        meta_json=meta_json,
        train=False,
        length=frames_per_clip,
        label=f"syntax_{artery}",
        artery=artery,
        heatmap_root=heatmap_root,
        video_root=video_root,
        val_ratio=val_ratio,
        seed=seed,
        transform=test_transform,
    )

    train_dataloader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_set,
        batch_size=1,
        num_workers=num_workers,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    x, h, y, w, p = next(iter(train_dataloader))
    print(x.shape)

    model = SyntaxLightningModuleHeatmap(
        num_classes=num_classes,
        lr=1e-4,
        weight_decay=0.001,
        max_epochs=10,
        save_path=f"backbone_hm/{artery}_pre_fold{fold:02d}.pt",
    )

    callbacks = [pl.callbacks.LearningRateMonitor(logging_interval="epoch")]
    logger = TensorBoardLogger("back_logs", name=f"{Artery}BinSyntax_R3D_pre_fold{fold:02d}")

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        fast_dev_run=fast_dev_run,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.save_checkpoint(f"backbone_hm/{Artery}BinSyntax_R3D_pre_fold{fold:02d}.pt")

    model = SyntaxLightningModuleHeatmap(
        num_classes=num_classes,
        lr=1e-4,
        weight_decay=0.001,
        max_epochs=max_epochs,
        weight_path=f"backbone_hm/{artery}_pre_fold{fold:02d}.pt",
        save_path=f"backbone_hm/{artery}_post_fold{fold:02d}.pt",
    )

    callbacks = [pl.callbacks.LearningRateMonitor(logging_interval="epoch")]
    logger = TensorBoardLogger("back_logs", name=f"{Artery}BinSyntax_R3D_full_fold{fold:02d}")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        fast_dev_run=fast_dev_run,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.save_checkpoint(f"backbone_hm/{Artery}BinSyntax_R3D_full_fold{fold:02d}.pt")


if __name__ == "__main__":
    main()
