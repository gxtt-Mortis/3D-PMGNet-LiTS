# dataset_cadic.py  — LiTS 肝脏肿瘤分割数据集

import os
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import nibabel as nib
from torch.utils.data import Dataset

DEFAULT_TARGET_SHAPE = (1, 128, 128, 128)

# --- 预处理参数 ---
HU_MIN, HU_MAX = -160, 240          # HU 裁剪范围（肝脏软组织窗）


class LiTSDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        phase: str = "train",
        stage: str = "liver",              # "liver" | "tumor"
        val_ratio: float = 0.2,
        target_shape: tuple = DEFAULT_TARGET_SHAPE,
        roi_crop: tuple = (20, 428, 92, 418),  # (row_start, row_end, col_start, col_end) 或 None
        seed: int = 42,
    ):
        """
        LiTS 肝脏肿瘤分割数据集 — 支持两阶段训练。

        stage="liver": label 将肝脏(1)和肿瘤(2)合并为前景, 输出 1 通道 CT
        stage="tumor": label 仅肿瘤(2)为前景, 输出 2 通道 [CT, liver_mask]

        支持两种目录结构：

        【结构 A — 扁平（自动划分）】
        data_dir/
          volume-0.nii
          segmentation-0.nii
          ...

        【结构 B — 预划分（split_data.py 生成）】
        data_dir/
          train/
            volume-0.nii
            segmentation-0.nii
          val/
            volume-1.nii
            segmentation-1.nii

        Parameters
        ----------
        data_dir : str
            数据根目录。
        phase : str
            "train" 或 "val"。
        stage : str
            "liver" (阶段一) 或 "tumor" (阶段二)。
        val_ratio : float
            验证集比例（仅结构 A 自动划分时生效）。
        target_shape : tuple
            (C, D, H, W) 输出统一尺寸。
        roi_crop : tuple or None
            (row_start, row_end, col_start, col_end)，None 则不裁剪。
        seed : int
            划分 train/val 的随机种子（仅结构 A 生效）。
        """
        self.target_shape = target_shape
        self.roi_crop = roi_crop
        self.stage = stage

        # 检测目录结构：有 train/ 子目录 → 预划分模式
        train_subdir = os.path.join(data_dir, "train")
        val_subdir   = os.path.join(data_dir, "val")

        if os.path.isdir(train_subdir) and os.path.isdir(val_subdir):
            # ---- 结构 B：预划分模式 ----
            self.data_dir = os.path.join(data_dir, phase)
            self.case_ids = self._scan_cases(self.data_dir)
        else:
            # ---- 结构 A：扁平 + 自动划分 ----
            self.data_dir = data_dir
            all_ids = self._scan_cases(data_dir)

            rng = np.random.RandomState(seed)
            indices = rng.permutation(len(all_ids))
            n_val = max(1, int(len(all_ids) * val_ratio)) if len(all_ids) > 1 else 0
            val_indices = set(indices[:n_val])

            if phase == "val":
                self.case_ids = [all_ids[i] for i in range(len(all_ids)) if i in val_indices]
            else:
                self.case_ids = [all_ids[i] for i in range(len(all_ids)) if i not in val_indices]

    @staticmethod
    def _scan_cases(directory: str) -> list:
        """扫描目录中的 volume-*.nii[.gz]，返回排序后的 case_id 列表。"""
        files = set(os.listdir(directory))
        ids = []
        for fname in files:
            if fname.startswith("volume-") and fname.endswith((".nii", ".nii.gz")):
                case_id = fname.replace("volume-", "").split(".nii")[0]
                seg_nii = f"segmentation-{case_id}.nii"
                seg_gz  = f"segmentation-{case_id}.nii.gz"
                if seg_nii in files or seg_gz in files:
                    ids.append(case_id)
        ids.sort()
        return ids

    def __len__(self):
        return len(self.case_ids)

    def _find_file(self, prefix: str, case_id: str) -> str:
        """查找 volume-{id}.nii[.gz] 或 segmentation-{id}.nii[.gz]"""
        for ext in (".nii.gz", ".nii"):
            path = os.path.join(self.data_dir, f"{prefix}-{case_id}{ext}")
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"未找到 {prefix}-{case_id}.nii[.gz]")

    def __getitem__(self, idx: int):
        case = self.case_ids[idx]
        img_path = self._find_file("volume", case)
        seg_path = self._find_file("segmentation", case)

        img = self._load_nifti(img_path)   # (D, H, W), float32
        seg = self._load_nifti(seg_path)   # (D, H, W), int

        # --- 预处理1: HU 裁剪 + 归一化 ---
        img = self._hu_clip_normalize(img)   # [-160, 240] → [0, 1]

        # --- 预处理2: 可选 ROI 裁剪 ---
        if self.roi_crop is not None:
            rs, re, cs, ce = self.roi_crop
            img = img[:, rs:re, cs:ce]
            seg = seg[:, rs:re, cs:ce]

        # --- 根据 stage 构建 label 和输入 ---
        if self.stage == "liver":
            # 合并肝脏(1)和肿瘤(2) → 前景(1)
            seg = (seg > 0).astype(np.int64)
            img_t = torch.from_numpy(img[None]).float()         # (1, D, H, W)
            seg_t = torch.from_numpy(seg[None]).long()

            img_t = self._pad_or_crop(img_t, self.target_shape)
            seg_t = self._pad_or_crop(seg_t, self.target_shape)
            return {"image": img_t, "label": seg_t, "case_id": case}

        else:  # stage == "tumor"
            # 肝脏 mask（前景=1）：合并肝脏和肿瘤区域
            liver_mask = (seg > 0).astype(np.float32)
            # 肿瘤 label：仅肿瘤(2)为 1
            tumor_label = (seg == 2).astype(np.int64)

            # 2 通道输入: [CT, liver_mask]
            img_2ch = np.stack([img, liver_mask], axis=0)       # (2, D, H, W)
            img_t = torch.from_numpy(img_2ch).float()
            seg_t = torch.from_numpy(tumor_label[None]).long()  # (1, D, H, W)

            img_t = self._pad_or_crop(img_t, self.target_shape)
            seg_t = self._pad_or_crop(seg_t, self.target_shape)
            # liver_mask 在第 1 通道中，已随 img_t 一起 crop/pad
            return {"image": img_t, "label": seg_t, "case_id": case}

    # ----------------------------------------------------------------
    #  预处理静态方法
    # ----------------------------------------------------------------
    def _hu_clip_normalize(self, img: np.ndarray) -> np.ndarray:
        """HU 裁剪 + 归一化: [HU_MIN, HU_MAX] → [0, 1]"""
        img = np.clip(img, HU_MIN, HU_MAX)
        img = (img - HU_MIN) / (HU_MAX - HU_MIN)
        return img.astype(np.float32)

    def _load_nifti(self, path: str) -> np.ndarray:
        """加载 NIfTI，统一输出 (D, H, W) float32"""
        try:
            arr = sitk.GetArrayFromImage(sitk.ReadImage(path))  # (D, H, W)
        except Exception:
            nb = nib.load(path)
            data = nb.get_fdata(dtype=np.float32)
            arr = data.transpose(2, 1, 0)  # (X, Y, Z) → (Z, Y, X) = (D, H, W)
        return arr

    def _pad_or_crop(self, x: torch.Tensor, target_shape: tuple):
        """中心裁剪 + 对称 pad 到 target_shape。x: (C, D, H, W)"""
        _, d0, h0, w0 = x.shape
        _, dt, ht, wt = target_shape

        # 中心裁剪
        if d0 > dt:
            sd = (d0 - dt) // 2
            x = x[:, sd:sd + dt, :, :]
        if h0 > ht:
            sh = (h0 - ht) // 2
            x = x[:, :, sh:sh + ht, :]
        if w0 > wt:
            sw = (w0 - wt) // 2
            x = x[:, :, :, sw:sw + wt]

        # 对称 pad
        pad_d = max(dt - x.shape[1], 0)
        pad_h = max(ht - x.shape[2], 0)
        pad_w = max(wt - x.shape[3], 0)
        pad_cfg = (
            pad_w // 2, pad_w - pad_w // 2,
            pad_h // 2, pad_h - pad_h // 2,
            pad_d // 2, pad_d - pad_d // 2,
            0, 0,
        )
        if any((pad_d, pad_h, pad_w)):
            x = F.pad(x, pad_cfg, mode='constant', value=0)
        return x


# 向后兼容别名
CTCACSimpleDataset = LiTSDataset


if __name__ == "__main__":
    data_dir = "../training_data"

    # ---- 测试 Stage 1: Liver ----
    print("\n=== STAGE 1: LIVER (train) ===")
    ds = LiTSDataset(data_dir, phase="train", stage="liver", val_ratio=0.2)
    print(f"train cases: {len(ds)}")
    if len(ds) > 0:
        sample = ds[0]
        print("img.shape :", sample['image'].shape)   # (1, 128, 128, 128)
        print("seg.shape :", sample['label'].shape)
        print("case_id   :", sample['case_id'])
        print("seg unique:", torch.unique(sample['label']))  # [0, 1]

    # ---- 测试 Stage 2: Tumor ----
    print("\n=== STAGE 2: TUMOR (train) ===")
    ds2 = LiTSDataset(data_dir, phase="train", stage="tumor", val_ratio=0.2)
    print(f"train cases: {len(ds2)}")
    if len(ds2) > 0:
        sample2 = ds2[0]
        print("img.shape :", sample2['image'].shape)   # (2, 128, 128, 128): CT + liver_mask
        print("seg.shape :", sample2['label'].shape)
        print("case_id   :", sample2['case_id'])
        print("seg unique:", torch.unique(sample2['label']))  # [0, 1] (only tumor)
        # 验证 liver_mask 通道有内容
        liver_ch = sample2['image'][1]
        print("liver_mask sum:", liver_ch.sum().item())
