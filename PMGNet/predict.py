#!/usr/bin/env python3
# predict.py  — LiTS 两阶段推理
#
# Stage 1: CT → 肝脏 mask
# Stage 2: [CT, liver_mask] → 肿瘤 mask
# 合并输出: 0=背景, 1=肝脏, 2=肿瘤（LiTS 提交格式）
#
# 用法:
#   python predict.py --input_dir ./test_data --output_dir ./predictions

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import SimpleITK as sitk
from tqdm.auto import tqdm

from PMGNet.MC_network_backbone import PMGNet

# ---- 预处理参数（需与训练一致） ----
HU_MIN, HU_MAX = -160, 240
ROI_ROW = (20, 428)
ROI_COL = (92, 418)
PATCH_SIZE = (96, 96, 96)          # (D, H, W) — 与训练一致
OVERLAP = 0.5                       # 滑动窗口重叠比例
TARGET_SPACING = (1.0, 1.0, 1.0)   # 各向同性重采样


def load_nifti(path: str) -> np.ndarray:
    """加载 NIfTI → (D, H, W) float32"""
    try:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
    except Exception:
        nb = nib.load(path)
        data = nb.get_fdata(dtype=np.float32)
        arr = data.transpose(2, 1, 0)  # (X,Y,Z) → (Z,Y,X)
    return arr.astype(np.float32)


def save_nifti(data: np.ndarray, ref_path: str, save_path: str):
    """
    将 (D, H, W) int 数组保存为 NIfTI，几何信息从 ref_path 复制。
    LiTS 要求 int16 标签。
    """
    data = data.astype(np.int16)
    # 转回 (X, Y, Z) = (W, H, D) → nibabel 格式
    data_xyz = data.transpose(2, 1, 0)    # (D,H,W) → (W,H,D)
    try:
        ref_img = sitk.ReadImage(ref_path)
        out_img = sitk.GetImageFromArray(data)  # (D,H,W) → SimpleITK
        out_img.CopyInformation(ref_img)
        sitk.WriteImage(out_img, save_path)
    except Exception:
        ref_nb = nib.load(ref_path)
        out_nb = nib.Nifti1Image(data_xyz, ref_nb.affine, ref_nb.header)
        out_nb.header.set_data_dtype(np.int16)
        nib.save(out_nb, save_path)


def preprocess(img: np.ndarray, roi_crop: bool = True) -> np.ndarray:
    """HU 裁剪 + 归一化 + ROI 裁剪。img: (D, H, W)"""
    img = np.clip(img, HU_MIN, HU_MAX)
    img = (img - HU_MIN) / (HU_MAX - HU_MIN)
    img = img.astype(np.float32)

    if roi_crop:
        rs, re = ROI_ROW
        cs, ce = ROI_COL
        img = img[:, rs:re, cs:ce]
    return img


def _pad_to_patch(vol: np.ndarray, patch_size: tuple) -> (np.ndarray, tuple):
    """
    将 vol (D,H,W) 对称 pad 到至少 patch_size。
    返回 (padded_vol, (pad_left, pad_right, ...)) 用于后续裁掉。
    """
    D, H, W = vol.shape
    pD, pH, pW = patch_size

    pad_d = max(0, pD - D)
    pad_h = max(0, pH - H)
    pad_w = max(0, pW - W)

    pd0, pd1 = pad_d // 2, pad_d - pad_d // 2
    ph0, ph1 = pad_h // 2, pad_h - pad_h // 2
    pw0, pw1 = pad_w // 2, pad_w - pad_w // 2

    padded = np.pad(vol, ((pd0, pd1), (ph0, ph1), (pw0, pw1)),
                    mode="constant", constant_values=0)
    pad_info = (pd0, pd1, ph0, ph1, pw0, pw1)
    return padded, pad_info


def _unpad(vol: np.ndarray, pad_info: tuple) -> np.ndarray:
    """反向移除对称 padding。"""
    pd0, pd1, ph0, ph1, pw0, pw1 = pad_info
    D, H, W = vol.shape
    d_end = D - pd1 if pd1 > 0 else D
    h_end = H - ph1 if ph1 > 0 else H
    w_end = W - pw1 if pw1 > 0 else W
    return vol[pd0:d_end, ph0:h_end, pw0:w_end]


def sliding_window(model: torch.nn.Module,
                   volume: np.ndarray,             # (C, D, H, W) or (D, H, W)
                   patch_size: tuple,
                   overlap: float = 0.5,
                   device: torch.device = None,
                   batch_size: int = 8) -> np.ndarray:
    """
    滑动窗口推理，返回概率图 (out_chans, D, H, W)。
    volume: (C, D, H, W) 或 (D, H, W)
    """
    if device is None:
        device = next(model.parameters()).device

    if volume.ndim == 3:
        volume = volume[None]  # (1, D, H, W)
    C, D, H, W = volume.shape
    pD, pH, pW = patch_size

    stride_D = max(1, int(pD * (1 - overlap)))
    stride_H = max(1, int(pH * (1 - overlap)))
    stride_W = max(1, int(pW * (1 - overlap)))

    # 起点列表
    starts_D = list(range(0, max(1, D - pD + 1), stride_D))
    starts_H = list(range(0, max(1, H - pH + 1), stride_H))
    starts_W = list(range(0, max(1, W - pW + 1), stride_W))
    # 确保最后一个 patch 覆盖到边缘
    if D > pD and starts_D[-1] + pD < D:
        starts_D.append(D - pD)
    if H > pH and starts_H[-1] + pH < H:
        starts_H.append(H - pH)
    if W > pW and starts_W[-1] + pW < W:
        starts_W.append(W - pW)
    if D < pD:
        starts_D = [0]
    if H < pH:
        starts_H = [0]
    if W < pW:
        starts_W = [0]

    model.eval()

    # 收集所有 patch 位置，分批推理
    positions = [(sd, sh, sw) for sd in starts_D for sh in starts_H for sw in starts_W]

    # 先跑一次获取 out_chans
    dummy = volume[:, :pD, :pH, :pW]
    # pad dummy if needed
    pad_d = max(0, pD - dummy.shape[1])
    pad_h = max(0, pH - dummy.shape[2])
    pad_w = max(0, pW - dummy.shape[3])
    if any([pad_d, pad_h, pad_w]):
        dummy = np.pad(dummy, ((0,0),(0,pad_d),(0,pad_h),(0,pad_w)), mode='constant')
    with torch.no_grad():
        dummy_t = torch.from_numpy(dummy[None]).float().to(device)
        out_chans = model(dummy_t).shape[1]

    # 累积输出和权重
    accum = np.zeros((out_chans, D, H, W), dtype=np.float32)
    weight = np.zeros((1, D, H, W), dtype=np.float32)

    # 批量处理
    batch_patches = []
    batch_positions = []
    for sd, sh, sw in positions:
        patch = volume[:, sd:sd + pD, sh:sh + pH, sw:sw + pW]
        pad_d = max(0, pD - patch.shape[1])
        pad_h = max(0, pH - patch.shape[2])
        pad_w = max(0, pW - patch.shape[3])
        if any([pad_d, pad_h, pad_w]):
            patch = np.pad(patch, ((0,0),(0,pad_d),(0,pad_h),(0,pad_w)), mode='constant')
        batch_patches.append(patch)
        batch_positions.append((sd, sh, sw))

        if len(batch_patches) >= batch_size:
            _process_batch(model, batch_patches, batch_positions, accum, weight, device)
            batch_patches, batch_positions = [], []

    if batch_patches:
        _process_batch(model, batch_patches, batch_positions, accum, weight, device)

    # 加权平均
    weight = np.maximum(weight, 1e-8)
    prob = accum / weight
    return prob


def _process_batch(model, patches, positions, accum, weight, device):
    """处理一个 batch 的 patch 并累加到 accum。"""
    batch_t = torch.from_numpy(np.stack(patches, axis=0)).float().to(device)
    pD, pH, pW = patches[0].shape[1:]

    with torch.no_grad():
        out = model(batch_t)                      # (B, C, pD, pH, pW)
        prob = F.softmax(out, dim=1).cpu().numpy()

    # 高斯权重（中心权重高，边缘低）
    gauss_w = _gaussian_weight((pD, pH, pW)).astype(np.float32)

    for i, (sd, sh, sw) in enumerate(positions):
        _, d_len, h_len, w_len = patches[i].shape  # 实际 patch 尺寸（可能小于 patch_size）
        p = prob[i, :, :d_len, :h_len, :w_len]
        w = gauss_w[:d_len, :h_len, :w_len]
        accum[:, sd:sd+d_len, sh:sh+h_len, sw:sw+w_len] += p * w[np.newaxis]
        weight[0, sd:sd+d_len, sh:sh+h_len, sw:sw+w_len] += w


def _gaussian_weight(shape: tuple) -> np.ndarray:
    """生成 3D 高斯权重图，用于平滑拼接。"""
    D, H, W = shape
    d = np.linspace(-1, 1, D)
    h = np.linspace(-1, 1, H)
    w = np.linspace(-1, 1, W)
    dd, hh, ww = np.meshgrid(d, h, w, indexing="ij")
    g = np.exp(-(dd**2 + hh**2 + ww**2) / 0.5)
    return g / g.max()


def predict_single(ct_path: str,
                   liver_model: torch.nn.Module,
                   tumor_model: torch.nn.Module,
                   device: torch.device,
                   roi_crop: bool = True,
                   save_path: str = None) -> np.ndarray:
    """
    对单个 CT 执行两阶段推理。

    返回: (D, H, W) int, 0=背景 1=肝脏 2=肿瘤（原始图像空间）
    """
    # 1. 加载原始图像（保留几何信息用于保存）
    img_raw = load_nifti(ct_path)                     # (D, H, W)
    D_raw, H_raw, W_raw = img_raw.shape

    # 2. 预处理
    img = preprocess(img_raw, roi_crop=roi_crop)       # (D, H_crop, W_crop)
    img_input = img.copy()                              # 保留裁剪后尺寸

    # 3. Pad 到至少 patch_size
    img_pad, pad_info = _pad_to_patch(img_input, PATCH_SIZE)  # (D', H', W')

    # ---- Stage 1: 肝脏 ----
    prob_liver = sliding_window(liver_model, img_pad, PATCH_SIZE,
                                overlap=OVERLAP, device=device)
    # prob_liver: (2, D', H', W'), channel 1 = 肝脏前景概率
    liver_mask = (prob_liver[1] > 0.5).astype(np.float32)   # (D', H', W')

    # ---- Stage 2: 肿瘤 ----
    # 2 通道输入: [CT_pad, liver_mask]
    img_2ch = np.stack([img_pad, liver_mask], axis=0)       # (2, D', H', W')
    prob_tumor = sliding_window(tumor_model, img_2ch, PATCH_SIZE,
                                overlap=OVERLAP, device=device)
    # prob_tumor: (2, D', H', W'), channel 1 = 肿瘤前景概率
    tumor_mask = (prob_tumor[1] > 0.5).astype(np.float32)    # (D', H', W')

    # 4. 组合最终标签（在 pad 空间中）
    final_pad = np.zeros(liver_mask.shape, dtype=np.int64)   # (D', H', W')
    final_pad[(liver_mask > 0) & (tumor_mask == 0)] = 1     # 肝脏
    final_pad[tumor_mask > 0] = 2                            # 肿瘤

    # 5. 反向移除 padding → 回到 ROI 裁剪后尺寸
    final_crop = _unpad(final_pad, pad_info)                  # (D, H_crop, W_crop)
    liver_crop = _unpad(liver_mask, pad_info)

    # 6. 如果用了 ROI 裁剪，贴回原始图像空间
    if roi_crop:
        final_full = np.zeros((D_raw, H_raw, W_raw), dtype=np.int64)
        rs, re = ROI_ROW
        cs, ce = ROI_COL
        # 确保尺寸匹配
        d_crop = min(final_crop.shape[0], D_raw)
        final_full[:d_crop, rs:re, cs:ce] = final_crop[:d_crop]
    else:
        final_full = final_crop

    # 7. 保存
    if save_path:
        save_nifti(final_full, ct_path, save_path)
        print(f"  Saved → {save_path}")

    return final_full


def main():
    parser = argparse.ArgumentParser(
        description="LiTS 两阶段推理: CT → 肝脏 + 肿瘤分割"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="测试 CT 目录 (volume-*.nii)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="预测输出目录")
    parser.add_argument("--liver_model", type=str, default="../models/best_liver_model.pth",
                        help="肝脏模型权重路径")
    parser.add_argument("--tumor_model", type=str, default="../models/best_tumor_model.pth",
                        help="肿瘤模型权重路径")
    parser.add_argument("--no_roi_crop", action="store_true",
                        help="不使用 ROI 裁剪（使用完整 512×512 图像）")
    parser.add_argument("--liver_thresh", type=float, default=0.5,
                        help="肝脏概率阈值 (default: 0.5)")
    parser.add_argument("--tumor_thresh", type=float, default=0.5,
                        help="肿瘤概率阈值 (default: 0.5)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"ROI crop: {not args.no_roi_crop}")

    # ---- 加载模型 ----
    print("\nLoading liver model...")
    liver_model = PMGNet(in_chans=1, out_chans=2, depths=[2,2,2,2],
                         feat_size=[48,96,192,384], spatial_dims=3).to(device)
    liver_model.load_state_dict(torch.load(args.liver_model, map_location=device,
                                           weights_only=True))
    liver_model.eval()

    print("Loading tumor model...")
    tumor_model = PMGNet(in_chans=2, out_chans=2, depths=[2,2,2,2],
                         feat_size=[48,96,192,384], spatial_dims=3).to(device)
    tumor_model.load_state_dict(torch.load(args.tumor_model, map_location=device,
                                           weights_only=True))
    tumor_model.eval()

    # ---- 扫描测试文件 (支持 volume- 和 test-volume- 两种前缀) ----
    test_files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.endswith((".nii", ".nii.gz"))
        and (f.startswith("test-volume-") or f.startswith("volume-"))
    ])

    if not test_files:
        print(f"[ERROR] 在 {args.input_dir} 中未找到 volume-*.nii 或 test-volume-*.nii 文件")
        return

    print(f"\n找到 {len(test_files)} 个测试文件:")
    for f in test_files:
        print(f"  {f}")

    # ---- 推理 ----
    print("\n开始推理...")
    for fname in tqdm(test_files, desc="Predicting"):
        ct_path = os.path.join(args.input_dir, fname)
        # 提取 case_id: test-volume-0.nii / volume-0.nii → "0"
        base = fname.replace("test-volume-", "").replace("volume-", "")
        case_id = base.split(".nii")[0]
        out_name = f"test-pre-segmentation-{case_id}.nii"
        save_path = os.path.join(args.output_dir, out_name)

        predict_single(
            ct_path=ct_path,
            liver_model=liver_model,
            tumor_model=tumor_model,
            device=device,
            roi_crop=not args.no_roi_crop,
            save_path=save_path,
        )

    print(f"\nDone! 预测结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
