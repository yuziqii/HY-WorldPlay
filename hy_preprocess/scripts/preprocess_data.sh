#!/bin/bash

#######################################################################
# 数据预处理脚本 - MP4 + NPZ 数据格式
#
# 使用方法:
#   方式 1: 修改脚本内的配置参数，然后运行
#           bash scripts/preprocess_data.sh
#
#   方式 2: 通过命令行传参
#           bash scripts/preprocess_data.sh <csv_file> <output_dir> [model_path]
#
# 示例:
#   bash scripts/preprocess_data.sh ./data/train.csv ./output
#   bash scripts/preprocess_data.sh ./data/train.csv ./output ./models/HunyuanVideo-1.5
#
# 注意:
#   处理完成后可运行检查模式：
#   python preprocess.py --check_only --input_json ./output/dataset_index.json
#######################################################################

set -e  # 遇到错误立即退出

# ============================================================================
# 配置参数（可通过命令行覆盖）
# ============================================================================

# 数据路径 - CSV 标注文件
DATA_CSV="${1:-./datasets/sekai-game-drone/sekai-game-drone.csv}"

# 输出目录
OUTPUT_DIR="${2:-./preprocessed_data}"

# 模型路径 - HunyuanVideo-1.5 根目录
MODEL_PATH="${3:-./model_ckpts/HunyuanVideo-1.5}"

# 输出索引文件名
OUTPUT_JSON="dataset_index.json"

# 目标分辨率（必须能被16整除）
TARGET_HEIGHT=480
TARGET_WIDTH=832

# 目标帧数（None表示不重采样，可设置为如129、65等）
TARGET_NUM_FRAMES=""  # 留空表示保持原样

# 计算设备
DEVICE="cuda"

# 处理样本数量（空表示处理全部，用于测试可设置为如：10）
NUM_SAMPLES=""  # 留空表示处理全部

# ============================================================================
# 脚本开始
# ============================================================================

echo "======================================================================"
echo "  数据预处理管线 - MP4 + NPZ 格式"
echo "======================================================================"
echo ""

# 激活虚拟环境（如果存在）
if [ -d ".venv" ]; then
    echo "[INFO] 激活虚拟环境..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "[INFO] 激活虚拟环境..."
    source venv/bin/activate
fi

# 设置 PYTHONPATH
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "[INFO] PYTHONPATH: $PYTHONPATH"
echo ""

# 参数验证
echo "======================================================================"
echo "  配置信息"
echo "======================================================================"
echo "  CSV 文件:        $DATA_CSV"
echo "  输出目录:        $OUTPUT_DIR"
echo "  模型路径:        $MODEL_PATH"
echo "  输出索引:        $OUTPUT_JSON"
echo "  目标分辨率:      ${TARGET_WIDTH}x${TARGET_HEIGHT}"
echo "  目标帧数:        ${TARGET_NUM_FRAMES:-保持原样}"
echo "  计算设备:        $DEVICE"
echo "  处理数量:        ${NUM_SAMPLES:-全部}"
echo "======================================================================"
echo ""

# 检查 CSV 文件
if [ ! -f "$DATA_CSV" ]; then
    echo "[ERROR] CSV 文件不存在: $DATA_CSV"
    echo ""
    echo "使用方法:"
    echo "  bash scripts/preprocess_data.sh <csv_file> <output_dir> [model_path]"
    echo ""
    echo "示例:"
    echo "  bash scripts/preprocess_data.sh ./data/train.csv ./output"
    echo "  bash scripts/preprocess_data.sh ./data/train.csv ./output ./models/HunyuanVideo-1.5"
    exit 1
fi

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] 模型路径不存在: $MODEL_PATH"
    echo "请修改脚本中的 MODEL_PATH 变量，或通过参数传递正确路径"
    exit 1
fi

# 检查 CSV 格式
if [[ ! "$DATA_CSV" == *.csv ]]; then
    echo "[ERROR] DATA_CSV 必须指向 .csv 文件，当前: $DATA_CSV"
    exit 1
fi

echo "[INFO] 开始预处理..."
echo ""

# ============================================================================
# 构建命令参数
# ============================================================================
export CUDA_VISIBLE_DEVICES=1

CMD="python preprocess.py \
    --data_root $DATA_CSV \
    --output_dir $OUTPUT_DIR \
    --output_json $OUTPUT_JSON \
    --model_path $MODEL_PATH \
    --target_height $TARGET_HEIGHT \
    --target_width $TARGET_WIDTH \
    --device $DEVICE"

# 添加可选参数
if [ -n "$TARGET_NUM_FRAMES" ]; then
    CMD="$CMD --target_num_frames $TARGET_NUM_FRAMES"
fi

if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD --num_samples $NUM_SAMPLES"
fi

# ============================================================================
# 执行预处理
# ============================================================================

echo "[INFO] 执行命令:"
echo "$CMD"
echo ""

eval $CMD

# ============================================================================
# 完成提示
# ============================================================================

echo ""
echo "======================================================================"
echo "  预处理完成！"
echo "======================================================================"
echo ""
echo "输出文件:"
echo "  - 索引文件: $OUTPUT_DIR/$OUTPUT_JSON"
echo "  - 数据文件: $OUTPUT_DIR/<segment_id>/<segment_id>_latent.pt"
echo ""
echo "下一步:"
echo "  1. 检查输出: cat $OUTPUT_DIR/$OUTPUT_JSON"
echo "  2. 验证数据: python -c \"import torch; print(torch.load('$OUTPUT_DIR/<segment_id>/<segment_id>_latent.pt').keys())\""
echo "  3. 清洗数据（可选）: python datasets/hy_preprocess/preprocess.py --check_only --input_json $OUTPUT_DIR/$OUTPUT_JSON"
echo ""
echo "======================================================================"
