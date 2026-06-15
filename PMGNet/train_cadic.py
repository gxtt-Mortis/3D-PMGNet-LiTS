# trainer.py

import os
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.backends import cudnn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from monai.losses import DiceLoss
from PMGNet.MC_network_backbone import PMGNet

from dataset_cadic import CTCACSimpleDataset, DEFAULT_TARGET_SHAPE

cudnn.benchmark = True

class CTTrainer:
    def __init__(
        self,
        data_root: str,
        num_classes: int = 4,
        epochs: int = 300,
        lr: float = 1e-4,
        batch_size: int = 1,
        alpha: float = 0.5,
        num_workers: int = 4,
        target_shape: tuple = DEFAULT_TARGET_SHAPE,
        label_suffix: str = "label"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.epochs = epochs
        self.alpha = alpha

        # Model
        self.model = PMGNet(
            in_chans=1,
            out_chans=num_classes,
            depths=[2,2,2,2],
            feat_size=[48,96,192,384],
            spatial_dims=3
        ).to(self.device)

        # Datasets & Loaders
        train_ds = CTCACSimpleDataset(
            data_root, phase="train",
            target_shape=target_shape,
            label_suffix=label_suffix
        )
        val_ds = CTCACSimpleDataset(
            data_root, phase="val",
            target_shape=target_shape,
            label_suffix=label_suffix
        )
        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True
        )

        # Losses
        self.dice_loss = DiceLoss(
            to_onehot_y=True, softmax=True, include_background=False
        )
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

        # Optimizer & AMP
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scaler = GradScaler()

        # Logging & checkpoints
        self.writer = SummaryWriter(log_dir="runs/ct")
        self.best_dice = 0.0
        self.pred_dir = "predictions"
        os.makedirs(self.pred_dir, exist_ok=True)

        # 固定样本
        self._create_fixed_loader(train_ds, num_samples=1)

    def _create_fixed_loader(self, full_dataset, num_samples: int = 1):
        np.random.seed(42)
        idxs = np.random.choice(len(full_dataset), num_samples, replace=False)
        subset = Subset(full_dataset, idxs)
        self.fixed_loader = DataLoader(
            subset, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=True
        )
        self.fixed_indices = idxs


    def compute_metrics(self, pred: np.ndarray, target: np.ndarray):
        metrics = {}
        for c in range(1, self.num_classes):
            pm = (pred == c); tm = (target == c)
            inter = (pm & tm).sum()
            union = pm.sum() + tm.sum()
            metrics[f"dice_{c}"] = 2 * inter / (union + 1e-5)
        return metrics

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        acc = {f"dice_{c}": [] for c in range(1, self.num_classes)}
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for batch in pbar:
            imgs = batch["image"].to(self.device)
            lbls = batch["label"].squeeze(1).to(self.device)
            self.optimizer.zero_grad()
            with autocast():
                out = self.model(imgs)
                loss_d = self.dice_loss(out, lbls.unsqueeze(1))
                loss_ce = self.ce_loss(out, lbls)
                loss = self.alpha * loss_d + (1 - self.alpha) * loss_ce
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            with torch.no_grad():
                pred = torch.argmax(out, dim=1).cpu().numpy()
                tgt  = lbls.cpu().numpy()
                for b in range(pred.shape[0]):
                    for k, v in self.compute_metrics(pred[b], tgt[b]).items():
                        acc[k].append(v)
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        avg_loss = total_loss / len(self.train_loader)
        avg_m = {k: float(np.mean(v)) for k, v in acc.items()}
        return {"loss": avg_loss, **avg_m}

    def validate(self, epoch: int = None):
        self.model.eval()
        acc = {f"dice_{c}": [] for c in range(1, self.num_classes)}
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]  ", leave=False)
        with torch.no_grad():
            for batch in pbar:
                imgs = batch["image"].to(self.device)
                lbls = batch["label"].squeeze(1).to(self.device)
                out = self.model(imgs)
                pred = torch.argmax(out, dim=1).cpu().numpy()
                tgt  = lbls.cpu().numpy()
                for b in range(pred.shape[0]):
                    for k, v in self.compute_metrics(pred[b], tgt[b]).items():
                        acc[k].append(v)
        return {k: float(np.mean(v)) for k, v in acc.items()}

    def run(self):
        print(f"Training on {self.device}")
        for epoch in range(1, self.epochs + 1):
            tr_res = self.train_epoch(epoch)
            val_res = self.validate(epoch)
            self.save_predictions(epoch)

            # TensorBoard logging
            self.writer.add_scalar("Loss/train", tr_res["loss"], epoch)
            for k,v in tr_res.items():
                if k != "loss":
                    self.writer.add_scalar(f"Train/{k}", v, epoch)
            for k,v in val_res.items():
                self.writer.add_scalar(f"Val/{k}", v, epoch)

            # Console output
            print(
                f"Epoch {epoch}/{self.epochs} | "
                f"Train Loss: {tr_res['loss']:.4f} | "
                + " | ".join([f"{k}:{tr_res[k]:.4f}" for k in tr_res if k!="loss"])
            )
            print("Val   | " + " | ".join([f"{k}:{val_res[k]:.4f}" for k in val_res]))

            # Save best model
            avg_dice = np.mean([val_res[f"dice_{c}"] for c in range(1, self.num_classes)])
            if avg_dice > self.best_dice:
                self.best_dice = avg_dice
                torch.save(self.model.state_dict(), "best_ct_model.pth")
                print(f"[Info] New best model saved (avg Dice: {avg_dice:.4f})")

        self.writer.close()

if __name__ == "__main__":
    data_root = "D:\paper\second\PMGNet\data"
    trainer = CTTrainer(
        data_root=data_root,
        num_classes=4,
        epochs=300,
        lr=1e-4,
        batch_size=4,
        alpha=0.5,
        num_workers=4,
        target_shape=DEFAULT_TARGET_SHAPE,
        label_suffix="label"  # or "labelnew"
    )
    trainer.run()
