#!/bin/bash

# TinyVLA-Cache评估运行脚本
# 用法: ./scripts/run_eval_vla_cache.sh <model_path> [task_name] [output_dir]

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <model_path> [task_name] [output_dir]"
    echo "  <model_path>: TinyVLA模型路径"
    echo "  [task_name]: 评估任务名称 (默认: sim_transfer_cube)"
    echo "  [output_dir]: 评估结果保存目录 (默认: vla_cache_eval_results)"
    exit 1
fi

# 设置参数
MODEL_PATH=$1
TASK_NAME=${2:-"sim_transfer_cube"}
OUTPUT_DIR=${3:-"vla_cache_eval_results_${TASK_NAME}"}

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行评估 - 启用缓存
echo "运行评估 (启用VLA-Cache)..."
python run_eval_vla_cache.py \
    --model_path $MODEL_PATH \
    --task_name $TASK_NAME \
    --output_dir "${OUTPUT_DIR}/with_cache" \
    --cache_size 0.3 \
    --importance_threshold 0.7 \
    --cache_update_freq 10 \
    --cache_warmup_steps 5

# 运行评估 - 禁用缓存
echo "运行评估 (禁用VLA-Cache)..."
python run_eval_vla_cache.py \
    --model_path $MODEL_PATH \
    --task_name $TASK_NAME \
    --output_dir "${OUTPUT_DIR}/no_cache" \
    --no_cache

# 比较结果
echo "评估完成！"
echo "结果保存在: $OUTPUT_DIR"
echo "- 启用缓存的结果: ${OUTPUT_DIR}/with_cache"
echo "- 禁用缓存的结果: ${OUTPUT_DIR}/no_cache"

# 显示平均结果
echo "平均结果比较:"
echo "启用缓存:"
cat "${OUTPUT_DIR}/with_cache/avg_results.txt"
echo ""
echo "禁用缓存:"
cat "${OUTPUT_DIR}/no_cache/avg_results.txt" 