#!/usr/bin/env python3
# split_data.py  — LiTS 数据集划分脚本
# 将扁平的 volume-*.nii / segmentation-*.nii 按比例划分到 train/ val/ 子目录。
# 自包含，无项目内依赖，可直接复制到服务器运行。

import os
import sys
import shutil
import argparse
import numpy as np


def find_cases(src_dir: str) -> list:
    """扫描 volume-*.nii[.gz]，返回存在的 case_id 列表（已排序）。"""
    cases = []
    files = set(os.listdir(src_dir))
    for fname in sorted(files):
        if not fname.startswith("volume-"):
            continue
        # volume-0.nii / volume-0.nii.gz → "0"
        base = fname.replace("volume-", "", 1)
        for ext in (".nii.gz", ".nii"):
            if base.endswith(ext):
                case_id = base[:-len(ext)]
                break
        else:
            continue
        # 确认对应的 segmentation 文件存在
        seg_nii = f"segmentation-{case_id}.nii"
        seg_gz  = f"segmentation-{case_id}.nii.gz"
        if seg_nii in files or seg_gz in files:
            cases.append(case_id)
    return cases


def copy_case(src: str, dst_root: str, case_id: str, subset: str):
    """将一个 case 的 volume 和 segmentation 复制到 dst_root/subset/ 下。"""
    dst_dir = os.path.join(dst_root, subset)
    os.makedirs(dst_dir, exist_ok=True)

    for prefix in ("volume", "segmentation"):
        for ext in (".nii.gz", ".nii"):
            fname = f"{prefix}-{case_id}{ext}"
            src_path = os.path.join(src, fname)
            if os.path.exists(src_path):
                shutil.copy2(src_path, os.path.join(dst_dir, fname))
                break


def main():
    parser = argparse.ArgumentParser(
        description="划分 LiTS 数据集为 train/val，复制文件到目标目录"
    )
    parser.add_argument(
        "--src", required=True,
        help="源目录，包含 volume-*.nii 和 segmentation-*.nii"
    )
    parser.add_argument(
        "--dst", required=True,
        help="目标目录，将创建 train/ 和 val/ 子目录"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.2,
        help="验证集比例 (default: 0.2)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子 (default: 42)"
    )
    parser.add_argument(
        "--move", action="store_true",
        help="移动文件而非复制（默认复制）"
    )
    args = parser.parse_args()

    cases = find_cases(args.src)
    if not cases:
        print(f"[ERROR] 在 {args.src} 中未找到任何 volume-*.nii 文件")
        sys.exit(1)

    print(f"找到 {len(cases)} 个 cases: {cases}")

    # 随机划分
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(cases))
    n_val = max(1, int(len(cases) * args.val_ratio)) if len(cases) > 1 else 0
    val_set = set(indices[:n_val])

    train_ids = [cases[i] for i in range(len(cases)) if i not in val_set]
    val_ids   = [cases[i] for i in range(len(cases)) if i in val_set]

    print(f"Train: {len(train_ids)} cases → {train_ids}")
    print(f"Val:   {len(val_ids)} cases   → {val_ids}")

    copy_func = shutil.move if args.move else shutil.copy2
    action = "移动" if args.move else "复制"

    for cid in train_ids:
        copy_case(args.src, args.dst, cid, "train")
    for cid in val_ids:
        copy_case(args.src, args.dst, cid, "val")

    # 保存划分记录
    split_log = os.path.join(args.dst, "split_info.txt")
    with open(split_log, "w", encoding="utf-8") as f:
        f.write(f"seed={args.seed}  val_ratio={args.val_ratio}\n")
        f.write(f"train ({len(train_ids)}): {', '.join(train_ids)}\n")
        f.write(f"val   ({len(val_ids)}):   {', '.join(val_ids)}\n")
    print(f"\n完成 — {action}了 {len(cases)} 个 cases 到 {args.dst}")
    print(f"划分记录已保存到 {split_log}")


if __name__ == "__main__":
    main()
