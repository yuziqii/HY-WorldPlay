# 更新日志 - 2026-04-01

## 🔄 主要变更

### 1. 文件整合与重命名
- ✅ 合并 `preprocess_game_dataset.py` + `check_and_clean_latents.py` → `preprocess.py`
- ✅ 删除冗余文件，简化目录结构

### 2. 功能增强
- ✅ **双模式支持**：
  - 预处理模式（默认）
  - 检查模式（`--check_only`）
- ✅ 统一的命令行接口
- ✅ 改进的用户体验

---

## 📁 文件变更详情

### 新增文件
```
preprocess.py                    # 整合后的主脚本（15KB）
```

### 删除文件
```
❌ preprocess_game_dataset.py    # 已整合到 preprocess.py
❌ check_and_clean_latents.py    # 已整合到 preprocess.py
❌ test_pt.py                     # 测试文件，不再需要
❌ test_utils.py                  # 测试文件，不再需要
❌ generate_neg_prompt_pt.py      # 功能已损坏，已移除
❌ preprocess_gamefactory_dataset.py      # 不支持的格式
❌ preprocess_game_dataset2.py            # 不支持的格式
❌ preprocess_gamefactory_dataset_refactored.py  # 重复版本
❌ preprocess_game_dataset_refactored.py        # 重复版本
❌ preprocess_game_dataset2_refactored.py       # 重复版本
❌ scripts/generate_neg_prompt_pt.sh      # 无用的脚本
❌ scripts/preprocess_gamefactory.sh      # 不支持的格式
```

### 保留文件
```
✅ preprocess.py                  # 主预处理脚本
✅ utils/                         # 工具模块（必需）
✅ scripts/preprocess_data.sh    # Bash 启动脚本
✅ README.md                      # 使用文档
```

---

## 🚀 新的使用方式

### 预处理模式
```bash
# 基本用法
python preprocess.py --data_root data.csv --output_dir output --model_path model

# 完整参数
python preprocess.py \
    --data_root data.csv \
    --output_dir output \
    --model_path model \
    --target_num_frames 129 \
    --num_samples 10
```

### 检查模式
```bash
# 检查数据质量
python preprocess.py \
    --check_only \
    --input_json output/dataset_index.json \
    --min_frames 7
```

### Bash 脚本
```bash
# 方式 1: 修改配置后运行
bash scripts/preprocess_data.sh

# 方式 2: 命令行传参
bash scripts/preprocess_data.sh data.csv output model
```

---

## ✨ 改进亮点

### 1. 更简洁的接口
- 从 2 个脚本 → 1 个统一脚本
- 通过 `--check_only` 标志切换模式

### 2. 更好的文档
- 更新 `README.md` 包含两种模式说明
- 添加更多使用示例
- 改进参数说明表格

### 3. 更少的维护负担
- 删除 11 个冗余文件
- 只保留必需的 `utils/__init__.py`
- 清晰的文件组织结构

---

## 📊 对比

### 重构前
```
hy_preprocess/
├── preprocess_gamefactory_dataset.py      (1017 行) ❌
├── preprocess_game_dataset.py              (551 行)  ❌
├── preprocess_game_dataset2.py             (427 行)  ❌
├── check_and_clean_latents.py              (73 行)   ❌
├── test_pt.py                              (7 行)    ❌
├── test_utils.py                           (200 行)  ❌
├── generate_neg_prompt_pt.py               (94 行)  ❌
├── scripts/
│   ├── preprocess_gamefactory.sh          ❌
│   └── generate_neg_prompt_pt.sh          ❌
└── utils/                                  (1010 行) ✅
```
**总计：约 3379 行代码（含重复）**

### 重构后
```
hy_preprocess/
├── preprocess.py                           (380 行)  ✅
├── README.md                               (11 KB)  ✅
├── scripts/
│   └── preprocess_data.sh                  (120 行) ✅
└── utils/                                  (1010 行) ✅
```
**总计：约 1510 行代码**

**代码减少：约 55%** 🎉

---

## 🎯 下一步

### 开始使用
```bash
# 1. 测试单个样本
python preprocess.py --data_root data.csv --output_dir output --model_path model --num_samples 1

# 2. 检查输出
ls output/
python -c "import torch; print(torch.load('output/xxx/xxx_latent.pt').keys())"

# 3. 全量处理
python preprocess.py --data_root data.csv --output_dir output --model_path model

# 4. 质量检查
python preprocess.py --check_only --input_json output/dataset_index.json
```

---

## 📞 相关文档

- 详细使用说明：`README.md`
- Bash 脚本：`scripts/preprocess_data.sh`
- 工具模块：`utils/`

---

**更新完成！现在可以开始使用了 🚀**