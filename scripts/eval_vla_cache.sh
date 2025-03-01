#!/bin/bash

# TinyVLA-Cache评估脚本
# 用法: bash scripts/eval_vla_cache.sh MODEL_PATH TASK_NAME

# 检查参数
if [ "$#" -lt 2 ]; then
    echo "用法: bash scripts/eval_vla_cache.sh MODEL_PATH TASK_NAME"
    exit 1
fi

MODEL_PATH=$1
TASK_NAME=$2
OUTPUT_DIR="vla_cache_results_${TASK_NAME}"

# 执行评估
python eval_vla_cache.py \
    --model_path ${MODEL_PATH} \
    --task_name ${TASK_NAME} \
    --num_samples 50 \
    --output_dir ${OUTPUT_DIR} \
    --cache_size 0.3 \
    --importance_threshold 0.7 \
    --cache_update_freq 10 \
    --cache_warmup_steps 5

echo "评估完成！结果保存在: ${OUTPUT_DIR}" 