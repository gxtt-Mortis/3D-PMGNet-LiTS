# 3D-PMGNet — LiTS 肝脏肿瘤两阶段分割

## 目录

- [整体架构](#整体架构)
- [数据格式与预处理](#数据格式与预处理)
- [模型架构](#模型架构)
- [两阶段训练策略](#两阶段训练策略)
- [代码文件说明](#代码文件说明)
  - [dataset_cadic.py](#dataset_cadicpy)
  - [train_two_stage.py](#train_two_stagepy)
  - [predict.py](#predictpy)
  - [split_data.py](#split_datapy)
- [使用方式](#使用方式)
- [输出文件说明](#输出文件说明)

---

## 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      训练阶段                                │
│                                                             │
│  原始标注 {0,1,2}                                           │
│       │                                                     │
│       ├──► 阶段1 (liver): 合并 1+2→1, 训练肝脏分割模型       │
│       │       CT(1ch) → PMGNet → liver_mask                 │
│       │                                                      │
│       └──► 阶段2 (tumor): 仅 2→1, 训练肿瘤分割模型           │
│               [CT, liver_mask](2ch) → PMGNet → tumor_mask   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                      推理阶段                                │
│                                                             │
│  测试 CT (volume-*.nii)                                     │
│       │                                                     │
│       ├──► Stage 1: 滑动窗口 → liver_mask                   │
│       │                                                     │
│       └──► Stage 2: [CT, liver_mask] → tumor_mask           │
│                    │                                        │
│                    合成: bg=0, liver=1, tumor=2              │
│                    │                                        │
│                    输出: segmentation-*.nii (LiTS 提交格式)   │
└─────────────────────────────────────────────────────────────┘
```

## 数据格式与预处理

### 输入数据

LiTS 数据集，每个 case 包含两个 NIfTI 文件：

| 文件 | 内容 | 形状 | 数据类型 |
|------|------|------|----------|
| `volume-{id}.nii` | CT 扫描 | (512, 512, D) → 转置为 (D, 512, 512) | int16, HU 值 |
| `segmentation-{id}.nii` | 标注 | (512, 512, D) | int16, {0,1,2} |

标签含义：`0`=背景, `1`=肝脏, `2`=肿瘤

### 预处理流水线

```
原始 CT (D, 512, 512)  [HU: -3024 ~ 1410]
        │
        ▼
① HU 裁剪 + 归一化:    clip[-160, 240] → normalize → [0, 1]
        │
        ▼
② ROI 裁剪 (可选):     [:, 20:428, 92:418] → (D, 408, 326)
        │
        ▼
③ 中心裁剪/对称 Pad:   → (1, 128, 128, 128) 固定尺寸
```

- **HU 窗口 [-160, 240]**：肝脏软组织窗，滤除骨骼和空气
- **ROI 裁剪**：去除 CT 图像边缘的无效区域
- **128³ 固定尺寸**：PMGNet 要求的输入尺寸，超过则中心裁剪，不足则对称补零

### 目录结构

支持两种模式，自动检测：

```
# 结构 A — 扁平（内置随机划分，固定 seed）
data_dir/
  volume-0.nii
  segmentation-0.nii
  volume-1.nii
  ...

# 结构 B — 预划分（split_data.py 生成）
data_dir/
  train/
    volume-0.nii, segmentation-0.nii, ...
  val/
    volume-1.nii, segmentation-1.nii, ...
```

---

## 模型架构

### PMGNet (Progressive Multi-scale Grouping Network)

```
输入: (B, in_chans, 128, 128, 128)
        │
        ├──► uxnet_conv (ConvNeXt 风格 3D Encoder)
        │      depths=[2,2,2,2]
        │      feat_size=[48, 96, 192, 384]
        │      输出 4 个尺度的特征图
        │
        ├──► Encoder 分支 (UnetrBasicBlock × 5)
        │      enc1: in_chans → 48    (原始分辨率)
        │      enc2: 48 → 96          (1/2)
        │      enc3: 96 → 192         (1/4)
        │      enc4: 192 → 384        (1/8)
        │      enc5: 384 → 768        (1/16, bottleneck)
        │
        ├──► MC + Refine 模块
        │      对每个 encoder 输出做 MC Dropout 采样
        │      → Softmax 得到概率图
        │      → 平均 → RefineSegmentation 精调
        │      → ProbPromptFusion 融合原始特征与概率
        │
        ├──► Decoder (Fuseblock × 4)
        │      转置卷积上采样 + Skip Connection
        │      decoder5: 768 → 384
        │      decoder4: 384 → 192
        │      decoder3: 192 → 96
        │      decoder2: 96 → 48
        │
        └──► Output: UnetOutBlock
               (B, out_chans, 128, 128, 128)
```

**关键模块：**

| 模块 | 位置 | 作用 |
|------|------|------|
| `uxnet_conv` | `pmg_encoder.py` | 3D ConvNeXt 骨干网络，提取多尺度特征 |
| `ProbPromptFusion` | `PosFuse.py` | 基于位置编码的特征-概率交叉注意力融合 |
| `RefineSegmentation` | `mc_refine.py` | MC Dropout 不确定性估计 + 自适应阈值精调 |
| `Fuseblock` | `MC_network_backbone.py` | 转置卷积上采样 + skip connection 融合 |

---

## 两阶段训练策略

### 阶段 1：肝脏分割

| 项目 | 设置 |
|------|------|
| 输入通道 | `in_chans=1` (CT 单通道) |
| 输出通道 | `out_chans=2` (背景, 肝脏) |
| 标签处理 | 原始标注 `{0,1,2}` → 合并为 `{0,1}` |
| 模型保存 | `best_liver_model.pth` |

**设计思路**：肝脏和肿瘤在 CT 上都呈现为软组织密度，阶段 1 将两者合并为 "肝脏区域" 统一学习，降低任务难度，保证肝脏召回率。

### 阶段 2：肿瘤分割

| 项目 | 设置 |
|------|------|
| 输入通道 | `in_chans=2` (CT + liver_mask) |
| 输出通道 | `out_chans=2` (背景, 肿瘤) |
| 标签处理 | 原始标注仅 `{2}` → `{1}`，其余为 0 |
| 模型保存 | `best_tumor_model.pth` |

**设计思路**：
- 第二通道的 liver_mask 直接告诉模型 "肝脏在哪里"
- 模型只需在肝脏区域内区分肿瘤 vs 正常肝组织
- 这比从全图直接找肿瘤（前景占比极小）要容易得多

### 损失函数

```python
loss = α × DiceLoss + (1-α) × CrossEntropyLoss
# α = 0.5, CE 的 ignore_index = -100
```

- **DiceLoss** (MONAI): `include_background=False`，只计算前景类的 Dice
- **CrossEntropyLoss**: 标准多分类交叉熵
- **混合权重 α=0.5**: 两个损失等权重

### 训练配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| epoch | 300 | 3D 医学分割常用范围（200-500） |
| lr | 1e-4 | AdamW 初始学习率 |
| batch_size | 2 | 单张 128³ 约需 2-4GB 显存 |
| optimizer | AdamW | weight_decay=1e-5 |
| AMP | GradScaler | 混合精度训练，节省显存 |

---

## 代码文件说明

### `dataset_cadic.py`

**核心类: `LiTSDataset(Dataset)`**

```
__init__(data_dir, phase, stage, val_ratio, target_shape, roi_crop, seed)
  │
  ├── 检测 data_dir 下是否有 train/ val/ 子目录
  │   ├── 有 → 结构B：直接读对应 phase 的目录
  │   └── 无 → 结构A：扫描所有 case，按 seed 随机划分
  │
  └── __getitem__(idx)
        │
        ├── _load_nifti()    → 加载 CT + 标注 (SimpleITK → nibabel 备用)
        ├── _hu_clip_normalize() → HU 裁剪 + 归一化
        ├── ROI 裁剪 (可选)
        │
        ├── stage="liver":
        │     标签: (seg > 0) → {0,1}
        │     输入: img[None] → (1, D, H, W)
        │
        ├── stage="tumor":
        │     标签: (seg == 2) → {0,1}
        │     输入: stack([img, liver_mask]) → (2, D, H, W)
        │
        └── _pad_or_crop() → (C, 128, 128, 128)
```

**关键设计**：
- `seed=42` 固定随机划分，保证每次 train/val 划分结果一致
- `stage` 参数控制标签合并策略，同一个类服务于两个训练阶段
- `_pad_or_crop` 对图像和标签做相同的空间变换

### `train_two_stage.py`

**核心类: `StageTrainer`**

```
__init__()
  ├── PMGNet 模型初始化 (in_chans/out_chans 按 stage 配置)
  ├── LiTSDataset × 2 (train + val)
  ├── DiceLoss + CrossEntropyLoss
  ├── AdamW + GradScaler (AMP)
  └── SummaryWriter (TensorBoard) + history 字典

run()
  └── for epoch in 1..300:
        ├── train_epoch()
        │     ├── forward → loss = 0.5×Dice + 0.5×CE
        │     ├── backward (AMP)
        │     └── 计算 batch dice
        │
        ├── validate()
        │     └── 计算 val dice
        │
        ├── 记录 history (epoch, train_loss, val_dice)
        ├── TensorBoard 写入
        └── 保存最佳模型 (按 val avg_dice)
  
  训练结束后:
  ├── 保存 history.json
  └── 绘制 loss_curve.png (matplotlib, 双栏: Loss + Dice)
```

**命令行参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_dir` | `../../training_data` | 数据路径 |
| `--stage` | `all` | liver / tumor / all |
| `--epochs` | 300 | 训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--batch_size` | 2 | 批次大小 |
| `--alpha` | 0.5 | Dice vs CE 权重 |
| `--val_ratio` | 0.2 | 验证集比例 |
| `--seed` | 42 | 随机种子 |
| `--roi_crop` | `20 428 92 418` | ROI 裁剪范围 |

### `predict.py`

**推理流水线: `predict_single()`**

```
CT 文件 (.nii)
    │
    ▼
① load_nifti()             → (D_raw, 512, 512)
    │
    ▼
② preprocess()             → HU clip + normalize + ROI crop → (D, 408, 326)
    │
    ▼
③ _pad_to_patch()          → 对称 pad 到 ≥ 128³ → (D', H', W')
    │
    ▼
④ sliding_window() × 2
    │
    ├── Stage 1: liver_model(CT) → liver_prob (2, D', H', W')
    │                                        │
    │                              threshold=0.5 → liver_mask
    │
    └── Stage 2: tumor_model([CT, liver_mask]) → tumor_prob (2, D', H', W')
                                                   │
                                         threshold=0.5 → tumor_mask
    │
    ▼
⑤ 标签合成                    liver=1, tumor=2 (肿瘤优先)
    │
    ▼
⑥ _unpad()                  移除 padding → 回到 ROI 尺寸
    │
    ▼
⑦ 贴回原始空间                将 ROI 区域放回 (D_raw, 512, 512)
    │
    ▼
⑧ save_nifti()              保留原始几何信息 (affine/origin/spacing)
                             segmentation-{id}.nii (int16)
```

**滑动窗口推理 (`sliding_window`)**：
- Patch 大小: 128×128×128
- 重叠比例: 50% (stride = 64)
- 高斯权重融合: 窗口中心权重高，边缘低，平滑拼接
- 批量处理: 一次推理最多 8 个 patch，加速推理

### `split_data.py`

可选的辅助脚本，将扁平目录按指定比例复制到 `train/` `val/` 子目录。

```
python split_data.py --src ./training_data --dst ./data_split --val_ratio 0.2 --seed 42
```

**注意**：日常训练不需要此脚本，`dataset_cadic.py` 内置的自动划分已足够。此脚本用于需要手动检查划分结果或迁移数据的场景。

---

## 使用方式

### 1. 训练

```bash
# 完整两阶段训练（最常用）
python train_two_stage.py \
    --data_dir /path/to/training_data \
    --stage all \
    --epochs 300 \
    --batch_size 2

# 仅训练肝脏模型
python train_two_stage.py --stage liver --epochs 300

# 仅训练肿瘤模型
python train_two_stage.py --stage tumor --epochs 300
```

### 2. 推理 & 提交

```bash
python predict.py \
    --input_dir /path/to/test_ct \
    --output_dir /path/to/submission \
    --liver_model best_liver_model.pth \
    --tumor_model best_tumor_model.pth
```

将 `submission/` 目录打包为 `.zip`，上传至 LiTS 评测平台。

### 3. 监控训练

```bash
# 终端实时输出
Epoch   1/300  Train [loss:0.9485 | dice_1:0.0027]
              Val   [dice_1:0.0003]
  >> Best model saved (avg Dice: 0.0003)

# TensorBoard
tensorboard --logdir runs/
```

### 4. 训练结束后

```
runs/
├── liver/
│   ├── history.json        ← 每 epoch 的 loss/dice 记录
│   ├── loss_curve.png       ← 训练曲线图
│   └── events.out.*         ← TensorBoard 日志
├── tumor/
│   ├── history.json
│   ├── loss_curve.png
│   └── events.out.*

best_liver_model.pth         ← 肝脏模型权重
best_tumor_model.pth         ← 肿瘤模型权重
```

---

## 关键参数速查

| 文件 | 参数 | 值 | 说明 |
|------|------|-----|------|
| `dataset_cadic.py` | `HU_MIN, HU_MAX` | -160, 240 | CT 软组织窗 |
| `dataset_cadic.py` | `ROI_ROW` | (20, 428) | 去除上下边缘 |
| `dataset_cadic.py` | `ROI_COL` | (92, 418) | 去除左右边缘 |
| `dataset_cadic.py` | `DEFAULT_TARGET_SHAPE` | (1, 128, 128, 128) | 模型输入尺寸 |
| `train_two_stage.py` | `lr` | 1e-4 | AdamW 学习率 |
| `train_two_stage.py` | `alpha` | 0.5 | Dice/CE 损失权重 |
| `train_two_stage.py` | `val_ratio` | 0.2 | 验证集占比 |
| `train_two_stage.py` | `seed` | 42 | 随机种子 |
| `predict.py` | `PATCH_SIZE` | (128,128,128) | 滑动窗口大小 |
| `predict.py` | `OVERLAP` | 0.5 | 窗口重叠比例 |
