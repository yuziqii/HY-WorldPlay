# 数据预处理管线使用说明

## 📖 概述

本工具用于将 **MP4 视频 + NPZ 相机位姿数据** 预处理为训练所需的格式。支持：

### 🎯 核心功能
- ✅ **预处理模式**：处理视频和相机数据
  - 视频编码为 VAE latent
  - 相机位姿转换
  - 文本 prompt 编码
  - 首帧条件提取

- ✅ **检查模式**：数据质量验证
  - 检查文件完整性
  - 验证 latent 帧数
  - 自动过滤低质量数据

---

## 📁 目录结构

```
hy_preprocess/
├── preprocess.py                   # 主预处理脚本（双模式）
└── utils/                          # 共享工具模块
    ├── model_loading.py            # 模型加载
    ├── encoding.py                 # 特征编码
    ├── video_utils.py              # 视频处理
    └── pose_utils.py               # 相机位姿处理
```

---

## 📥 输入数据格式

### 1. CSV 标注文件

创建一个 CSV 文件（如 `data.csv`），包含以下列：

| 列名 | 说明 | 示例 |
|------|------|------|
| `videoFile` | MP4 文件名 | `video_001.mp4` |
| `cameraFile` | NPZ 相机文件名 | `camera_001.npz` |
| `caption` | 文本描述 | `A drone flying over the city` |

**示例 CSV:**
```csv
videoFile,cameraFile,caption
video_001.mp4,camera_001.npz,A drone flying over the city
video_002.mp4,camera_002.npz,A car driving on the highway
```

### 2. NPZ 相机文件格式

每个 NPZ 文件应包含：

```python
{
    'intrinsic': np.array shape (3, 3),    # 相机内参矩阵
    'extrinsic': np.array shape (N, 4, 4)  # N帧的相机外参矩阵 (c2w)
}
```

**说明：**
- `intrinsic`: 相机内参矩阵 (3x3)
- `extrinsic`: 每帧的相机到世界坐标变换矩阵 (N × 4 × 4)
- 系统会自动将 c2w 转换为训练所需的 w2c 格式

### 3. 目录结构示例

```
your_data/
├── data.csv              # 标注文件
├── video_001.mp4         # 视频文件
├── camera_001.npz        # 相机文件
├── video_002.mp4
├── camera_002.npz
└── ...
```

---

## 🚀 快速开始

### 预处理模式

```bash
python datasets/preprocess/hy_preprocess/preprocess.py \
    --data_root /path/to/your_data/data.csv \
    --output_dir ./preprocessed_data \
    --model_path /path/to/HunyuanVideo-1.5
```

### 检查模式

```bash
python datasets/preprocess/hy_preprocess/preprocess.py \
    --check_only \
    --input_json ./preprocessed_data/dataset_index.json
```

### 完整参数示例

```bash
# 预处理模式
python datasets/preprocess/hy_preprocess/preprocess.py \
    --data_root /path/to/data.csv \
    --output_dir ./output \
    --output_json dataset_index.json \
    --model_path /path/to/HunyuanVideo-1.5 \
    --target_height 480 \
    --target_width 832 \
    --target_num_frames 129 \
    --device cuda \
    --num_samples 10

# 检查模式
python datasets/preprocess/hy_preprocess/preprocess.py \
    --check_only \
    --input_json ./output/dataset_index.json \
    --min_frames 7
```

---

## ⚙️ 参数说明

### 预处理模式参数

**必需参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `--data_root` | str | CSV 文件路径（必须是 `.csv` 结尾） |
| `--output_dir` | str | 输出目录 |
| `--model_path` | str | HunyuanVideo-1.5 模型路径 |

**可选参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--output_json` | str | `dataset_index.json` | 输出索引文件名 |
| `--target_height` | int | 480 | 目标高度（必须能被16整除） |
| `--target_width` | int | 832 | 目标宽度（必须能被16整除） |
| `--target_num_frames` | int | None | 重采样帧数（None=保持原样） |
| `--device` | str | `cuda` | 计算设备 |
| `--num_samples` | int | None | 处理样本数量（用于测试） |

### 检查模式参数

**必需参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `--check_only` | flag | 启用检查模式 |
| `--input_json` | str | 输入的 dataset_index.json 路径 |

**可选参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--min_frames` | int | 7 | 最小 latent 帧数阈值 |
| `--output_dir` | str | None | 清洗后文件的输出目录 |

---

## 📤 输出格式

### 目录结构

```
output/
├── dataset_index.json              # 数据集索引
├── video_001/                      # 每个视频一个文件夹
│   ├── video_001_latent.pt         # latent + embeddings
│   ├── video_001_pose.json         # 相机位姿
│   └── video_001_action.json       # 动作数据（空）
├── video_002/
│   └── ...
```

### latent.pt 文件内容

```python
{
    'latent': Tensor,              # [1, C, T, H, W] - VAE编码结果
    'prompt_embeds': Tensor,       # [1, seq_len, dim] - 文本编码
    'prompt_mask': Tensor,         # [1, seq_len] - 文本mask
    'byt5_text_states': Tensor,    # [1, 256, 1472] - byT5编码
    'byt5_text_mask': Tensor,      # [1, 256] - byT5 mask
    'image_cond': Tensor,          # [1, C, 1, H, W] - 首帧条件
    'vision_states': Tensor        # [1, seq_len, dim] - 视觉特征
}
```

### pose.json 文件格式

```json
{
    "0": {
        "w2c": [[...]],      // 4x4 世界到相机变换矩阵
        "intrinsic": [[...]] // 3x3 相机内参
    },
    "1": { ... },
    ...
}
```

### dataset_index.json 格式

```json
[
    {
        "segment_id": "video_001",
        "video_name": "video_001.mp4",
        "video_path": "/path/to/video_001.mp4",
        "latent_path": "output/video_001/video_001_latent.pt",
        "pose_path": "output/video_001/video_001_pose.json",
        "action_path": "output/video_001/video_001_action.json",
        "prompt": "A drone flying over the city"
    },
    ...
]
```

---

## 💡 使用示例

### 示例 1: 预处理单个测试样本

```bash
python datasets/preprocess/hy_preprocess/preprocess.py \
    --data_root ./my_data/data.csv \
    --output_dir ./test_output \
    --model_path ./models/HunyuanVideo-1.5 \
    --num_samples 1
```

### 示例 2: 完整数据集预处理

```bash
python datasets/preprocess/hy_preprocess/preprocess.py \
    --data_root ./dataset/train.csv \
    --output_dir ./preprocessed/train \
    --model_path ./models/HunyuanVideo-1.5 \
    --target_num_frames 129
```

### 示例 3: 自定义分辨率

```bash
python datasets/preprocess/hy_preprocess/preprocess.py \
    --data_root ./data.csv \
    --output_dir ./output \
    --model_path ./models/HunyuanVideo-1.5 \
    --target_height 720 \
    --target_width 1280
```

### 示例 4: 数据质量检查

```bash
# 处理完成后检查数据质量
python datasets/preprocess/hy_preprocess/preprocess.py \
    --check_only \
    --input_json ./output/dataset_index.json \
    --min_frames 7
```

### 示例 5: 使用 Bash 脚本

```bash
# 方式 1: 修改脚本配置后运行
bash scripts/preprocess_data.sh

# 方式 2: 通过参数传递
bash scripts/preprocess_data.sh ./data/train.csv ./output
```

---

## 🧹 数据清洗

处理完成后，可以使用检查模式清洗数据：

```bash
python datasets/preprocess/hy_preprocess/preprocess.py \
    --check_only \
    --input_json ./output/dataset_index.json \
    --min_frames 7
```

**检查内容：**
- ✅ 过滤帧数过少的样本（< min_frames）
- ✅ 检查文件完整性（latent 文件是否存在）
- ✅ 验证 latent 文件格式（能否正常加载）
- ✅ 移除损坏或无效的数据

**输出：**
- 生成 `dataset_index_clean.json`（清洗后的索引）
- 打印统计信息（总数、过滤数、保留数）

---

## 🔧 常见问题

### Q1: 内存不足怎么办？

**A:**
- 使用 `--num_samples` 先处理少量样本测试
- 降低分辨率（`--target_height` / `--target_width`）
- 使用 `--target_num_frames` 限制帧数

### Q2: NPZ 文件格式错误？

**A:** 确保 NPZ 文件包含：
- `intrinsic`: shape (3, 3)
- `extrinsic`: shape (N, 4, 4)

检查方法：
```python
import numpy as np
data = np.load('camera.npz')
print(data['intrinsic'].shape)   # 应为 (3, 3)
print(data['extrinsic'].shape)   # 应为 (N, 4, 4)
```

### Q3: 视频和相机帧数不匹配？

**A:** 系统会自动处理：
- 如果视频帧数 > NPZ 帧数：截取视频前 N 帧
- 如果视频帧数 < NPZ 帧数：只使用视频实际帧数

### Q4: 如何验证处理结果？

**A:** 检查输出文件：
```python
import torch
data = torch.load('output/video_001/video_001_latent.pt')
print(data['latent'].shape)        # 检查 latent 形状
print(data['prompt_embeds'].shape) # 检查文本编码
```

### Q5: 可以并行处理吗？

**A:** 目前是串行处理。如需加速：
- 可以将 CSV 拆分为多个子文件
- 在多个终端分别处理不同的子集
- 最后合并 `dataset_index.json`

---

## 📊 性能参考

**处理速度（参考）：**
- 单个 480p 视频 (5秒, 24fps): ~30秒
- 包含：视频编码、文本编码、位姿处理、保存

**显存需求：**
- 最小: 12GB (float16)
- 推荐: 24GB+ (float32)

---

## 🎯 最佳实践

### 1. 数据准备
- ✅ 确保视频和 NPZ 文件一一对应
- ✅ NPZ 帧数 ≥ 视频帧数
- ✅ 相机内参和外参格式正确

### 2. 分步处理
```bash
# 第1步：处理1-2个样本测试
python preprocess_game_dataset.py ... --num_samples 2

# 第2步：检查输出是否正确
ls output/
python -c "import torch; print(torch.load('output/xxx/xxx_latent.pt').keys())"

# 第3步：全量处理
python preprocess_game_dataset.py ...
```

### 3. 数据清洗
```bash
# 处理完成后清洗数据
python check_and_clean_latents.py \
    --input_json output/dataset_index.json \
    --output_json output/dataset_index_clean.json \
    --min_frames 7
```

---

## 📝 代码集成

如果需要在 Python 代码中调用：

```python
import sys
sys.path.insert(0, '/path/to/worldmodel')

from datasets.preprocess.hy_preprocess.utils import (
    load_vae_model,
    load_text_encoder,
    encode_video_to_latent,
    encode_prompt,
    load_video_segment,
    resample_video_frames,
    convert_npz_to_pose_and_actions,
)

# 加载模型
vae = load_vae_model(model_path, device="cuda")
text_encoders = load_text_encoder(model_path, device="cuda")

# 处理视频
frames = load_video_segment(video_path, num_frames=100)
frames, indices = resample_video_frames(frames, target_num_frames=65)

# 编码
latent = encode_video_to_latent(vae, frames, 480, 832, device="cuda")
prompt_embeds = encode_prompt("Your prompt", text_encoders, device="cuda")

# 处理相机位姿
pose_dict, action_dict = convert_npz_to_pose_and_actions(npz_path, indices)
```

---

## 📞 问题反馈

如遇到问题，请检查：
1. ✅ 数据格式是否正确
2. ✅ 模型路径是否存在
3. ✅ 显存是否充足
4. ✅ Python 依赖是否安装完整

---

**祝使用愉快！🎉**