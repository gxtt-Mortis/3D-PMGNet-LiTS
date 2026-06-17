# dataset_cadic.py  — LiTS 肝脏肿瘤分割数据集

import os
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import nibabel as nib
from torch.utils.data import Dataset

DEFAULT_TARGET_SHAPE = (1, 96, 96, 96)       # 96³ patch，比128³省近一半显存

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

        # 加载（保留原始 spacing，不做重采样以避免厚层切片暴增）
        img = self._load_nifti(img_path)   # (D, H, W), float32
        seg = self._load_nifti(seg_path)   # (D, H, W)

        # --- 预处理1: HU 裁剪 + 归一化 ---
        img = self._hu_clip_normalize(img)   # [-160, 240] → [0, 1]

        # --- 预处理2: ROI 裁剪（固定范围，可选） + 肝脏 bbox 动态裁剪 ---
        if self.roi_crop is not None:
            rs, re, cs, ce = self.roi_crop
            img = img[:, rs:re, cs:ce]
            seg = seg[:, rs:re, cs:ce]

        # 肝脏动态裁剪: 找到肝脏 bbox + 边距，保证 focus 在肝脏区域
        img, seg = self._liver_bbox_crop(img, seg)

        # --- 根据 stage 构建 label 和输入 ---
        if self.stage == "liver":
            seg = (seg > 0).astype(np.int64)
            img_t = torch.from_numpy(img[None]).float()
            seg_t = torch.from_numpy(seg[None]).long()

            img_t = self._pad_or_crop(img_t, self.target_shape)
            seg_t = self._pad_or_crop(seg_t, self.target_shape)
            return {"image": img_t, "label": seg_t, "case_id": case}

        else:  # stage == "tumor"
            # 肝脏区域 mask
            liver_mask = (seg > 0).astype(np.float32)
            # 肿瘤 label（仅肿瘤=1）
            tumor_label = (seg == 2).astype(np.int64)

            # CT 在肝脏外用 0 填充（过滤无关背景信息）
            img_masked = img * liver_mask

            # 2 通道输入: [masked_CT, liver_mask]
            img_2ch = np.stack([img_masked, liver_mask], axis=0)
            img_t = torch.from_numpy(img_2ch).float()
            seg_t = torch.from_numpy(tumor_label[None]).long()
            # liver_mask 额外返回，供 trainer 做 loss 过滤
            mask_t = torch.from_numpy(liver_mask[None]).float()

            img_t = self._pad_or_crop(img_t, self.target_shape)
            seg_t = self._pad_or_crop(seg_t, self.target_shape)
            mask_t = self._pad_or_crop(mask_t, self.target_shape)
            return {"image": img_t, "label": seg_t, "liver_mask": mask_t, "case_id": case}

    # ----------------------------------------------------------------
    #  预处理静态方法
    # ----------------------------------------------------------------
    def _liver_bbox_crop(self, img: np.ndarray, seg: np.ndarray,
                          margin: int = 20) -> (np.ndarray, np.ndarray):
        """基于肝脏 mask 动态裁剪，保证肝脏区域在视野内。无肝脏时返回原图。"""
        fg = np.where(seg > 0)
        if len(fg[0]) == 0:
            return img, seg

        d0, d1 = fg[0].min(), fg[0].max() + 1
        h0, h1 = fg[1].min(), fg[1].max() + 1
        w0, w1 = fg[2].min(), fg[2].max() + 1

        D, H, W = img.shape
        d0 = max(0, d0 - margin);  d1 = min(D, d1 + margin)
        h0 = max(0, h0 - margin);  h1 = min(H, h1 + margin)
        w0 = max(0, w0 - margin);  w1 = min(W, w1 + margin)

        return img[d0:d1, h0:h1, w0:w1], seg[d0:d1, h0:h1, w0:w1]

    def _hu_clip_normalize(self, img: np.ndarray) -> np.ndarray:
        """HU 裁剪 + 归一化: [HU_MIN, HU_MAX] → [0, 1]"""
        img = np.clip(img, HU_MIN, HU_MAX)
        img = (img - HU_MIN) / (HU_MAX - HU_MIN)
        return img.astype(np.float32)

    def _load_sitk(self, path: str):
        """加载为 SimpleITK 对象（保留 spacing 信息用于重采样）。"""
        try:
            return sitk.ReadImage(path)
        except Exception:
            nb = nib.load(path)
            data = nb.get_fdata(dtype=np.float32)
            data = data.transpose(2, 1, 0)  # → (D, H, W)
            img = sitk.GetImageFromArray(data.astype(np.float32))
            # 从 nibabel header 读取 spacing
            zooms = nb.header.get_zooms()[:3]  # (X, Y, Z)
            img.SetSpacing((float(zooms[2]), float(zooms[1]), float(zooms[0])))
            return img

    def _resample_isotropic(self, img_sitk, seg_sitk, target_spacing):
        """重采样到各向同性间距。CT 用线性插值，标签用最近邻。"""
        orig_spacing = img_sitk.GetSpacing()      # (Z, Y, X)
        orig_size = img_sitk.GetSize()             # (Z, Y, X)

        if orig_spacing == target_spacing:
            return img_sitk, seg_sitk

        new_size = [int(osz * osp / tsp + 0.5)
                    for osz, osp, tsp in zip(orig_size, orig_spacing, target_spacing)]

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetOutputOrigin(img_sitk.GetOrigin())
        resampler.SetOutputDirection(img_sitk.GetDirection())

        resampler.SetInterpolator(sitk.sitkLinear)
        new_img = resampler.Execute(img_sitk)

        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        new_seg = resampler.Execute(seg_sitk)

        return new_img, new_seg

    def _sitk_to_array(self, sitk_img) -> np.ndarray:
        """SimpleITK → numpy (D, H, W) float32"""
        return sitk.GetArrayFromImage(sitk_img)

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
        """随机裁剪 + 对称 pad 到 target_shape (默认96³)。小体积补零，大体积每epoch随机裁不同区域。"""
        _, d0, h0, w0 = x.shape
        _, dt, ht, wt = target_shape

        # 随机裁剪
        if d0 > dt:
            sd = np.random.randint(0, d0 - dt + 1)
            x = x[:, sd:sd + dt, :, :]
        if h0 > ht:
            sh = np.random.randint(0, h0 - ht + 1)
            x = x[:, :, sh:sh + ht, :]
        if w0 > wt:
            sw = np.random.randint(0, w0 - wt + 1)
            x = x[:, :, :, sw:sw + wt]

        # 对称 pad
        pad_d = max(dt - x.shape[1], 0); pad_h = max(ht - x.shape[2], 0); pad_w = max(wt - x.shape[3], 0)
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
