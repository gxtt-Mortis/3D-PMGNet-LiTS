#!/usr/bin/env python3
# train_two_stage.py  — LiTS 肝脏→肿瘤两阶段训练
#
# Stage 1 (liver):  CT → 肝脏 mask（肝脏+肿瘤合并为前景）
# Stage 2 (tumor):  [CT, liver_mask] → 肿瘤 mask（仅在肝脏区域内预测）
#
# 用法:
#   python train_two_stage.py --stage liver   # 仅训练肝脏模型
#   python train_two_stage.py --stage tumor   # 仅训练肿瘤模型
#   python train_two_stage.py --stage all     # 依次训练两个模型

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
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

# matplotlib 可选（无 GUI 环境也能用 Agg backend）
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# ================================================================
#  配置：可按需修改
# ================================================================
STAGE1_CONFIG = {
    "in_chans": 1,          # CT 单通道
    "out_chans": 2,         # 背景 + 肝脏
    "stage": "liver",
    "model_path": "../models/best_liver_model.pth",
    "log_dir": "../runs/liver",
}

STAGE2_CONFIG = {
    "in_chans": 2,          # CT + liver_mask
    "out_chans": 2,         # 背景 + 肿瘤
    "stage": "tumor",
    "model_path": "../models/best_tumor_model.pth",
    "log_dir": "../runs/tumor",
}


class StageTrainer:
    """通用训练器 — stage 参数决定输入通道数和标签处理方式。"""

    def __init__(
        self,
        data_dir: str,
        in_chans: int = 1,
        out_chans: int = 2,
        stage: str = "liver",
        epochs: int = 300,
        lr: float = 1e-4,
        batch_size: int = 2,
        alpha: float = 0.5,             # Dice vs CE 权重
        num_workers: int = 4,
        target_shape: tuple = DEFAULT_TARGET_SHAPE,
        val_ratio: float = 0.2,
        seed: int = 42,
        model_path: str = "best_model.pth",
        log_dir: str = "runs/default",
        roi_crop: tuple = (20, 428, 92, 418),
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stage = stage
        self.out_chans = out_chans
        self.epochs = epochs
        self.alpha = alpha
        self.model_path = model_path
        self.log_dir = log_dir

        # ---- Model ----
        self.model = PMGNet(
            in_chans=in_chans,
            out_chans=out_chans,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            spatial_dims=3,
        ).to(self.device)

        # ---- Datasets ----
        ds_kwargs = dict(
            data_dir=data_dir, stage=stage, val_ratio=val_ratio,
            target_shape=target_shape, roi_crop=roi_crop, seed=seed,
        )
        train_ds = LiTSDataset(phase="train", **ds_kwargs)
        val_ds   = LiTSDataset(phase="val",   **ds_kwargs)

        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        # ---- Losses ----
        self.dice_loss = DiceLoss(
            to_onehot_y=True, softmax=True, include_background=False,
        )
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

        # ---- Optimizer & AMP ----
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scaler = GradScaler()

        # ---- Logging ----
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.best_dice = 0.0
        self.history = {
            "stage": stage,
            "epoch": [],
            "train_loss": [],
            "val_dice": [],
        }

        print(f"[{stage.upper()}] Device: {self.device}  |  "
              f"In: {in_chans}  Out: {out_chans}  |  "
              f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ----------------------------------------------------------------
    #  Metrics
    # ----------------------------------------------------------------
    def compute_metrics(self, pred: np.ndarray, target: np.ndarray) -> dict:
        """计算每个前景类的 Dice。pred/target: (D, H, W) int"""
        metrics = {}
        for c in range(1, self.out_chans):
            pm = (pred == c)
            tm = (target == c)
            inter = (pm & tm).sum()
            union = pm.sum() + tm.sum()
            metrics[f"dice_{c}"] = 2.0 * inter / (union + 1e-5)
        return metrics

    # ----------------------------------------------------------------
    #  Train / Val one epoch
    # ----------------------------------------------------------------
    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        acc = {f"dice_{c}": [] for c in range(1, self.out_chans)}
        pbar = tqdm(self.train_loader, desc=f"E{epoch} [Train]", leave=False)

        for batch in pbar:
            imgs = batch["image"].to(self.device)
            lbls = batch["label"].squeeze(1).to(self.device)   # (B, D, H, W)

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

    @torch.no_grad()
    def validate(self, epoch: int = None) -> dict:
        self.model.eval()
        acc = {f"dice_{c}": [] for c in range(1, self.out_chans)}
        pbar = tqdm(self.val_loader, desc=f"E{epoch} [Val]  ", leave=False)

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

    # ----------------------------------------------------------------
    #  Main loop
    # ----------------------------------------------------------------
    def run(self):
        print(f"\n{'='*50}")
        print(f"Training Stage: {self.stage.upper()}")
        print(f"Model will be saved to: {self.model_path}")
        print(f"{'='*50}\n")

        for epoch in range(1, self.epochs + 1):
            tr_res = self.train_epoch(epoch)
            val_res = self.validate(epoch)

            # ---- 记录 history ----
            fg_dice = [val_res[f"dice_{c}"] for c in range(1, self.out_chans)]
            avg_dice = float(np.mean(fg_dice)) if fg_dice else 0.0
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(tr_res["loss"])
            self.history["val_dice"].append(avg_dice)

            # ---- TensorBoard ----
            self.writer.add_scalar("Loss/train", tr_res["loss"], epoch)
            for k, v in tr_res.items():
                if k != "loss":
                    self.writer.add_scalar(f"Train/{k}", v, epoch)
            for k, v in val_res.items():
                self.writer.add_scalar(f"Val/{k}", v, epoch)

            # ---- Console ----
            train_str = " | ".join(
                [f"loss:{tr_res['loss']:.4f}"]
                + [f"{k}:{v:.4f}" for k, v in tr_res.items() if k != "loss"]
            )
            val_str = " | ".join([f"{k}:{v:.4f}" for k, v in val_res.items()])
            print(f"Epoch {epoch:3d}/{self.epochs}  Train [{train_str}]")
            print(f"              Val   [{val_str}]")

            # ---- Save best ----
            if avg_dice > self.best_dice:
                self.best_dice = avg_dice
                torch.save(self.model.state_dict(), self.model_path)
                print(f"  >> Best model saved (avg Dice: {avg_dice:.4f})")

        self.writer.close()

        # ---- 保存 history JSON ----
        history_path = os.path.join(self.log_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved to: {history_path}")

        # ---- 绘制 loss 曲线 ----
        self._plot_history()

        print(f"\n[{self.stage.upper()}] Training done. Best Dice: {self.best_dice:.4f}\n")

    def _plot_history(self):
        """绘制 train loss + val dice 曲线，保存到 log_dir。"""
        if not _HAS_MPL:
            print("[Warn] matplotlib 未安装，跳过绘图。")
            return

        epochs = self.history["epoch"]
        train_loss = self.history["train_loss"]
        val_dice = self.history["val_dice"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        # --- Train Loss ---
        ax1.plot(epochs, train_loss, "b-", linewidth=1)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"[{self.stage.upper()}] Train Loss")
        ax1.grid(True, alpha=0.3)

        # --- Val Dice ---
        ax2.plot(epochs, val_dice, "r-", linewidth=1)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Dice")
        ax2.set_title(f"[{self.stage.upper()}] Val Dice")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        fig.suptitle(f"Training Curves — {self.stage.upper()}", fontsize=13)
        fig.tight_layout()

        plot_path = os.path.join(self.log_dir, "loss_curve.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Loss curve saved to: {plot_path}")


# ================================================================
#  Main
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="LiTS 两阶段训练: Liver → Tumor"
    )
    parser.add_argument("--data_dir", type=str, default="../training_data",
                        help="数据目录")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["liver", "tumor", "all"],
                        help="训练阶段 (default: all)")
    parser.add_argument("--epochs", type=int, default=300,
                        help="每个阶段的 epoch 数")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--roi_crop", type=int, nargs=4,
                        default=[20, 428, 92, 418],
                        help="ROI 裁剪: row_start row_end col_start col_end")
    parser.add_argument("--liver_model", type=str, default="../models/best_liver_model.pth",
                        help="Stage 1 输出的肝脏模型路径")
    parser.add_argument("--tumor_model", type=str, default="../models/best_tumor_model.pth",
                        help="Stage 2 输出的肿瘤模型路径")

    args = parser.parse_args()
    roi_crop = tuple(args.roi_crop) if args.roi_crop else None

    common = dict(
        data_dir=args.data_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        alpha=args.alpha,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        roi_crop=roi_crop,
    )

    if args.stage in ("liver", "all"):
        trainer1 = StageTrainer(
            **common,
            in_chans=STAGE1_CONFIG["in_chans"],
            out_chans=STAGE1_CONFIG["out_chans"],
            stage=STAGE1_CONFIG["stage"],
            model_path=args.liver_model,
            log_dir=STAGE1_CONFIG["log_dir"],
        )
        trainer1.run()

    if args.stage in ("tumor", "all"):
        trainer2 = StageTrainer(
            **common,
            in_chans=STAGE2_CONFIG["in_chans"],
            out_chans=STAGE2_CONFIG["out_chans"],
            stage=STAGE2_CONFIG["stage"],
            model_path=args.tumor_model,
            log_dir=STAGE2_CONFIG["log_dir"],
        )
        trainer2.run()


if __name__ == "__main__":
    main()
