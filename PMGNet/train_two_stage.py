#!/usr/bin/env python3
# train_two_stage.py  — LiTS 肝脏→肿瘤两阶段训练
#
# Stage 1 (liver):  CT → 肝脏 mask（肝脏+肿瘤合并为前景）
# Stage 2 (tumor):  [CT, liver_mask] → 肿瘤 mask（仅在肝脏区域内预测）
#
# 用法:
#   python train_two_stage.py --stage liver              # 从头训练肝脏
#   python train_two_stage.py --stage liver --resume     # 续训肝脏
#   python train_two_stage.py --stage tumor              # 训练肿瘤
#   python train_two_stage.py --stage all                # 依次训练两个

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.backends import cudnn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from monai.losses import DiceLoss
from PMGNet.MC_network_backbone import PMGNet

from dataset_cadic import LiTSDataset, DEFAULT_TARGET_SHAPE

cudnn.benchmark = True

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


STAGE1_CONFIG = {
    "in_chans": 1, "out_chans": 2, "stage": "liver",
    "model_path": "../models/best_liver_model.pth",
    "log_dir": "../runs/liver",
}

STAGE2_CONFIG = {
    "in_chans": 2, "out_chans": 2, "stage": "tumor",
    "model_path": "../models/best_tumor_model.pth",
    "log_dir": "../runs/tumor",
}


class StageTrainer:

    def __init__(
        self,
        data_dir: str,
        in_chans: int = 1,
        out_chans: int = 2,
        stage: str = "liver",
        epochs: int = 300,
        lr: float = 1e-4,
        batch_size: int = 2,
        alpha: float = 0.5,
        num_workers: int = 4,
        target_shape: tuple = DEFAULT_TARGET_SHAPE,
        val_ratio: float = 0.2,
        seed: int = 42,
        model_path: str = "best_model.pth",
        log_dir: str = "runs/default",
        roi_crop: tuple = (20, 428, 92, 418),
        resume: bool = False,
        patience: int = 0,         # 早停：连续无改善 epoch 数，0=禁用
        min_delta: float = 0.001,  # 视为改善的最小 dice 提升
        use_mc_refine: bool = True, # MC+Refine 概率模块，batch>1 需关闭
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stage = stage; self.out_chans = out_chans
        self.epochs = epochs; self.alpha = alpha
        self.model_path = model_path; self.log_dir = log_dir
        self.lr = lr
        self.patience = patience; self.min_delta = min_delta
        self.use_mc_refine = use_mc_refine

        os.makedirs(log_dir, exist_ok=True)
        self.ckpt_path = os.path.join(log_dir, "checkpoint.pth")

        # ---- Model ----
        self.model = PMGNet(in_chans=in_chans, out_chans=out_chans,
                            depths=[2,2,2,2], feat_size=[48,96,192,384],
                            spatial_dims=3,
                            use_mc_refine=use_mc_refine).to(self.device)

        # ---- Datasets ----
        ds_kwargs = dict(data_dir=data_dir, stage=stage, val_ratio=val_ratio,
                         target_shape=target_shape, roi_crop=roi_crop, seed=seed)
        train_ds = LiTSDataset(phase="train", **ds_kwargs)
        val_ds   = LiTSDataset(phase="val",   **ds_kwargs)
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                       num_workers=num_workers, pin_memory=True)
        self.val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers, pin_memory=True)

        # ---- Losses ----
        self.dice_loss = DiceLoss(to_onehot_y=True, softmax=True, include_background=False)
        self.ce_loss   = nn.CrossEntropyLoss(ignore_index=-100)

        # ---- Optimizer & AMP ----
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scaler = GradScaler()

        # ---- State ----
        self.writer = SummaryWriter(log_dir=log_dir)
        self.start_epoch = 1
        self.best_dice = 0.0
        self.no_improve = 0       # 早停计数
        self.history = {"stage": stage, "epoch": [], "train_loss": [], "val_dice": []}

        # ---- Resume ----
        if resume:
            if os.path.exists(self.ckpt_path):
                self._load_checkpoint()          # 完整断点
            elif os.path.exists(self.model_path):
                self._load_model_weights()       # 只有模型权重

        print(f"[{stage.upper()}] Device: {self.device}  |  "
              f"In: {in_chans}  Out: {out_chans}  |  "
              f"MC+Refine: {use_mc_refine}  |  "
              f"Train: {len(train_ds)}  Val: {len(val_ds)}  |  "
              f"Start epoch: {self.start_epoch}")

    # ----------------------------------------------------------------
    #  Checkpoint
    # ----------------------------------------------------------------
    def _save_checkpoint(self, epoch: int):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_dice": self.best_dice,
            "no_improve": self.no_improve,
            "history": self.history,
        }, self.ckpt_path)

    def _load_model_weights(self):
        """仅加载模型权重（无优化器状态，从头开始训练）。"""
        dummy = torch.randn(1, self.model.in_chans, 32, 32, 32, device=self.device)
        with torch.no_grad():
            self.model(dummy)
        state = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        print(f"[Resume] 加载模型权重 {self.model_path}（优化器从头开始）")

    def _load_checkpoint(self):
        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        # 先跑一次 dummy forward，触发 PosFuse 等 lazy 模块初始化
        dummy = torch.randn(1, self.model.in_chans, 32, 32, 32, device=self.device)
        with torch.no_grad():
            self.model(dummy)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.best_dice = ckpt["best_dice"]
        self.history   = ckpt.get("history", self.history)
        self.no_improve = ckpt.get("no_improve", 0)
        self.start_epoch = ckpt["epoch"] + 1
        print(f"[Resume] 从 epoch {self.start_epoch} 继续, best_dice={self.best_dice:.4f}")

    def _save_history(self):
        path = os.path.join(self.log_dir, "history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    # ----------------------------------------------------------------
    #  Metrics
    # ----------------------------------------------------------------
    def compute_metrics(self, pred: np.ndarray, target: np.ndarray) -> dict:
        metrics = {}
        for c in range(1, self.out_chans):
            pm = (pred == c); tm = (target == c)
            inter = (pm & tm).sum(); union = pm.sum() + tm.sum()
            metrics[f"dice_{c}"] = 2.0 * inter / (union + 1e-5)
        return metrics

    # ----------------------------------------------------------------
    #  Train / Val
    # ----------------------------------------------------------------
    def _get_loss_mask(self, batch):
        """肿瘤阶段返回肝脏 mask 用于 Dice 过滤"""
        if self.stage == "tumor" and "liver_mask" in batch:
            return batch["liver_mask"].squeeze(1).to(self.device)
        return None

    def _masked_dice(self, out, lbls, mask):
        """Dice loss 只在 mask>0 区域计算。"""
        prob = F.softmax(out, dim=1)
        target = F.one_hot(lbls.long(), num_classes=self.out_chans).permute(0,4,1,2,3).float()
        m = mask.unsqueeze(1)  # (B,1,D,H,W)
        prob, target = prob * m, target * m
        # 逐类 Dice
        dice = 0.0; count = 0
        for c in range(1, self.out_chans):
            inter = (prob[:,c] * target[:,c]).sum()
            union = prob[:,c].sum() + target[:,c].sum()
            if union > 0:
                dice += 1 - (2*inter + 1e-5) / (union + 1e-5)
                count += 1
        return dice / max(count, 1)

    def _compute_metrics_masked(self, pred, tgt, batch):
        """肿瘤阶段：只在肝脏区域算 dice"""
        if self.stage == "tumor" and "liver_mask" in batch:
            mask = batch["liver_mask"].squeeze(1).cpu().numpy().astype(bool)
            # 对 batch 中每个样本
            metrics = {}
            for b in range(pred.shape[0]):
                m = mask[b] if mask.ndim == 3 else mask
                for c in range(1, self.out_chans):
                    pm = (pred[b] == c) & m; tm = (tgt[b] == c) & m
                    inter = (pm & tm).sum(); union = pm.sum() + tm.sum()
                    k = f"dice_{c}"
                    metrics.setdefault(k, []).append(2.0 * inter / (union + 1e-5))
            return metrics
        else:
            metrics = {}
            for b in range(pred.shape[0]):
                for c in range(1, self.out_chans):
                    pm = (pred[b] == c); tm = (tgt[b] == c)
                    inter = (pm & tm).sum(); union = pm.sum() + tm.sum()
                    k = f"dice_{c}"
                    metrics.setdefault(k, []).append(2.0 * inter / (union + 1e-5))
            return metrics

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        acc = {}
        pbar = tqdm(self.train_loader, desc=f"E{epoch} [Train]", leave=False)
        for batch in pbar:
            imgs = batch["image"].to(self.device)
            lbls = batch["label"].squeeze(1).to(self.device)
            liver_mask = self._get_loss_mask(batch)

            self.optimizer.zero_grad()
            with autocast():
                out = self.model(imgs)
                if liver_mask is not None:
                    # 肿瘤阶段：Dice 用 mask 过滤，CE 用 -100 ignore
                    loss_d = self._masked_dice(out, lbls, liver_mask)
                    lbls_ce = lbls.clone(); lbls_ce[liver_mask == 0] = -100
                    loss_ce = self.ce_loss(out, lbls_ce)
                else:
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
                m = self._compute_metrics_masked(pred, tgt, batch)
                for k, v in m.items():
                    acc.setdefault(k, []).extend(v)
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))
        avg_loss = total_loss / len(self.train_loader)
        avg_m = {k: float(np.mean(v)) for k, v in acc.items()}
        return {"loss": avg_loss, **avg_m}

    @torch.no_grad()
    def validate(self, epoch: int = None) -> dict:
        self.model.eval()
        acc = {}
        pbar = tqdm(self.val_loader, desc=f"E{epoch} [Val]  ", leave=False)
        for batch in pbar:
            imgs = batch["image"].to(self.device)
            lbls = batch["label"].squeeze(1).to(self.device)
            pred = torch.argmax(self.model(imgs), dim=1).cpu().numpy()
            tgt  = lbls.cpu().numpy()
            m = self._compute_metrics_masked(pred, tgt, batch)
            for k, v in m.items():
                acc.setdefault(k, []).extend(v)
        return {k: float(np.mean(v)) for k, v in acc.items()}

    # ----------------------------------------------------------------
    #  Main loop
    # ----------------------------------------------------------------
    def run(self):
        print(f"\n{'='*50}")
        print(f"Training Stage: {self.stage.upper()}")
        print(f"Model: {self.model_path}  |  Checkpoint: {self.ckpt_path}")
        print(f"{'='*50}\n")

        for epoch in range(self.start_epoch, self.epochs + 1):
            tr_res = self.train_epoch(epoch)
            val_res = self.validate(epoch)

            # ---- history ----
            fg_dice = [val_res[f"dice_{c}"] for c in range(1, self.out_chans)]
            avg_dice = float(np.mean(fg_dice)) if fg_dice else 0.0
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(tr_res["loss"])
            self.history["val_dice"].append(avg_dice)

            # ---- TensorBoard ----
            self.writer.add_scalar("Loss/train", tr_res["loss"], epoch)
            for k, v in tr_res.items():
                if k != "loss": self.writer.add_scalar(f"Train/{k}", v, epoch)
            for k, v in val_res.items():
                self.writer.add_scalar(f"Val/{k}", v, epoch)

            # ---- Console ----
            train_str = " | ".join(
                [f"loss:{tr_res['loss']:.4f}"] +
                [f"{k}:{v:.4f}" for k, v in tr_res.items() if k != "loss"])
            val_str = " | ".join([f"{k}:{v:.4f}" for k, v in val_res.items()])
            print(f"Epoch {epoch:3d}/{self.epochs}  Train [{train_str}]")
            print(f"              Val   [{val_str}]")

            # ---- Save best ----
            if avg_dice > self.best_dice + self.min_delta:
                self.best_dice = avg_dice
                self.no_improve = 0
                torch.save(self.model.state_dict(), self.model_path)
                print(f"  >> Best model saved (avg Dice: {avg_dice:.4f})")
            else:
                self.no_improve += 1

            # ---- 早停 ----
            if self.patience > 0 and self.no_improve >= self.patience:
                print(f"\n[EarlyStop] {self.patience} 个 epoch 无改善，停止训练。")
                break

            # ---- 每个 epoch 保存断点 + history ----
            self._save_checkpoint(epoch)
            self._save_history()

        self.writer.close()
        self._plot_history()
        print(f"\n[{self.stage.upper()}] Done. Best Dice: {self.best_dice:.4f}\n")

    def _plot_history(self):
        if not _HAS_MPL: return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1.plot(self.history["epoch"], self.history["train_loss"], "b-", lw=1)
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax1.set_title(f"[{self.stage.upper()}] Train Loss"); ax1.grid(True, alpha=0.3)
        ax2.plot(self.history["epoch"], self.history["val_dice"], "r-", lw=1)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Dice")
        ax2.set_title(f"[{self.stage.upper()}] Val Dice"); ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        fig.suptitle(f"Training Curves — {self.stage.upper()}", fontsize=13)
        fig.tight_layout()
        path = os.path.join(self.log_dir, "loss_curve.png")
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="LiTS 两阶段训练: Liver → Tumor")
    parser.add_argument("--data_dir", type=str, default="../training_data")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["liver", "tumor", "all"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="从断点续训")
    parser.add_argument("--no_mc_refine", action="store_true",
                        help="禁用 MC+Refine，允许 batch>1 (牺牲少量精度换速度)")
    parser.add_argument("--patience", type=int, default=50,
                        help="早停耐心值，0=禁用 (default: 50)")
    parser.add_argument("--min_delta", type=float, default=0.001,
                        help="早停最小改善阈值 (default: 0.001)")
    parser.add_argument("--roi_crop", type=int, nargs=4,
                        default=[20, 428, 92, 418])
    parser.add_argument("--liver_model", type=str,
                        default="../models/best_liver_model.pth")
    parser.add_argument("--tumor_model", type=str,
                        default="../models/best_tumor_model.pth")
    args = parser.parse_args()

    roi_crop = tuple(args.roi_crop) if args.roi_crop else None
    common = dict(data_dir=args.data_dir, epochs=args.epochs, lr=args.lr,
                  batch_size=args.batch_size, alpha=args.alpha,
                  num_workers=args.num_workers, val_ratio=args.val_ratio,
                  seed=args.seed, roi_crop=roi_crop, resume=args.resume,
                  patience=args.patience, min_delta=args.min_delta,
                  use_mc_refine=not args.no_mc_refine)

    if args.stage in ("liver", "all"):
        StageTrainer(**common, **STAGE1_CONFIG).run()
    if args.stage in ("tumor", "all"):
        StageTrainer(**common, **STAGE2_CONFIG).run()


if __name__ == "__main__":
    main()
