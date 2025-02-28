#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TinyVLA-Cache评估脚本

该脚本用于评估VLA-Cache机制对TinyVLA模型性能的影响。
比较启用和禁用缓存时的推理时间、内存使用和准确性。
"""

import os
import time
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import torch.nn.functional as F

# 从TinyVLA导入必要的模块
from aloha_scripts.constants import TASK_CONFIGS
from data_utils.datasets import load_data
from llava_pythia.model.builder import load_pretrained_model

def parse_args():
    parser = argparse.ArgumentParser(description='评估VLA-Cache对TinyVLA性能的影响')
    parser.add_argument('--model_path', type=str, required=True,
                        help='TinyVLA模型路径')
    parser.add_argument('--task_name', type=str, default='real_robot_eval',
                        help='评估任务名称')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='评估的批次大小')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='评估的样本数量')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='计算设备')
    parser.add_argument('--output_dir', type=str, default='vla_cache_eval_results',
                        help='评估结果保存目录')
    # 缓存相关参数
    parser.add_argument('--no_cache', action='store_true',
                        help='不使用VLA-Cache')
    parser.add_argument('--cache_size', type=float, default=0.3,
                        help='缓存大小占总Token数的比例')
    parser.add_argument('--importance_threshold', type=float, default=0.7,
                        help='Token重要性阈值')
    parser.add_argument('--cache_update_freq', type=int, default=10,
                        help='缓存更新频率')
    parser.add_argument('--cache_warmup_steps', type=int, default=5,
                        help='缓存预热步数')
    
    return parser.parse_args()

def set_seed(seed):
    """设置随机种子以确保结果可重复"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(args):
    """加载模型和分词器"""
    print(f"Loading model from {args.model_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # 加载模型并配置VLA-Cache
    model_args = {
        "model_name_or_path": args.model_path,
        "use_cache_mechanism": not args.no_cache,
        "cache_size": args.cache_size,
        "importance_threshold": args.importance_threshold,
        "cache_update_freq": args.cache_update_freq,
        "cache_warmup_steps": args.cache_warmup_steps,
    }
    
    model, _ = load_pretrained_model(model_args)
    model.to(args.device)
    model.eval()
    
    return model, tokenizer

def prepare_data(args):
    """准备评估数据"""
    print(f"Preparing evaluation data for task: {args.task_name}")
    
    task_config = TASK_CONFIGS[args.task_name]
    dataset_dir = task_config['dataset_dir']
    camera_names = task_config['camera_names']
    episode_len = task_config.get('episode_len', 1000)
    
    # 加载数据
    train_data, val_data = load_data(
        dataset_dir=dataset_dir,
        camera_names=camera_names,
        episode_len=episode_len,
        skip_mirrored_data=True,
    )
    
    # 使用验证集进行评估
    eval_data = val_data
    
    # 限制样本数量
    if args.num_samples > 0 and args.num_samples < len(eval_data):
        indices = np.random.choice(len(eval_data), args.num_samples, replace=False)
        eval_data = [eval_data[i] for i in indices]
    
    print(f"Loaded {len(eval_data)} evaluation samples")
    return eval_data

def evaluate_model(model, tokenizer, eval_data, args):
    """评估模型性能"""
    print("Starting evaluation...")
    
    # 性能指标
    results = {
        "inference_times": [],
        "memory_usage": [],
        "action_errors": [],
        "cache_hit_ratios": [] if not args.no_cache else None,
    }
    
    # 测量GPU内存使用的函数
    def get_gpu_memory():
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        return 0
    
    for i, sample in enumerate(tqdm(eval_data, desc="Evaluating")):
        # 准备输入
        language_instruction = sample["language_instruction"]
        images = sample.get("images", None)  # [N, C, H, W]
        if images is not None:
            images = torch.tensor(images).to(args.device)
        states = sample.get("states", None)
        if states is not None:
            states = torch.tensor(states).to(args.device)
        
        # 编码文本
        inputs = tokenizer(
            language_instruction,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(args.device)
        
        # 记录初始内存
        initial_memory = get_gpu_memory()
        
        # 开始计时
        start_time = time.time()
        
        # 模型推理
        with torch.no_grad():
            outputs = model(
                **inputs,
                images=images,
                states=states,
                eval=True  # 启用评估模式
            )
        
        # 结束计时
        inference_time = time.time() - start_time
        
        # 记录最终内存
        final_memory = get_gpu_memory()
        memory_usage = final_memory - initial_memory
        
        # 记录推理结果
        results["inference_times"].append(inference_time)
        results["memory_usage"].append(memory_usage)
        
        # 如果有真实动作，计算误差
        if "actions" in sample:
            true_actions = torch.tensor(sample["actions"]).to(args.device)
            if isinstance(outputs, torch.Tensor):  # 如果输出是动作张量
                pred_actions = outputs
            else:  # 如果输出是模型输出对象
                # 根据模型输出类型获取预测动作
                if hasattr(outputs, "logits"):
                    pred_actions = outputs.logits
                else:
                    # 对于没有logits属性的输出，尝试直接使用输出
                    pred_actions = outputs
            
            # 计算MSE误差
            action_error = F.mse_loss(pred_actions, true_actions).item()
            results["action_errors"].append(action_error)
        
        # 如果启用了缓存，记录缓存命中率
        if not args.no_cache and hasattr(model, 'token_cache'):
            cache_stats = model.token_cache.get_cache_stats()
            results["cache_hit_ratios"].append(cache_stats["hit_ratio"])
    
    # 计算平均值
    avg_results = {
        "avg_inference_time": np.mean(results["inference_times"]),
        "avg_memory_usage": np.mean(results["memory_usage"]),
    }
    
    if results["action_errors"]:
        avg_results["avg_action_error"] = np.mean(results["action_errors"])
    
    if results["cache_hit_ratios"]:
        avg_results["avg_cache_hit_ratio"] = np.mean(results["cache_hit_ratios"])
    
    return results, avg_results

def plot_results(results_cache, results_no_cache, args):
    """绘制对比图表"""
    print("Plotting results...")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 准备数据
    times_cache = results_cache["inference_times"]
    times_no_cache = results_no_cache["inference_times"]
    
    memory_cache = results_cache["memory_usage"]
    memory_no_cache = results_no_cache["memory_usage"]
    
    # 绘制推理时间对比
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.boxplot([times_cache, times_no_cache], labels=["With Cache", "No Cache"])
    plt.title("Inference Time Comparison")
    plt.ylabel("Time (seconds)")
    
    # 绘制内存使用对比
    plt.subplot(1, 2, 2)
    plt.boxplot([memory_cache, memory_no_cache], labels=["With Cache", "No Cache"])
    plt.title("Memory Usage Comparison")
    plt.ylabel("Memory (MB)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "performance_comparison.png"))
    
    # 如果有动作误差数据，绘制误差对比
    if "action_errors" in results_cache and "action_errors" in results_no_cache:
        errors_cache = results_cache["action_errors"]
        errors_no_cache = results_no_cache["action_errors"]
        
        plt.figure(figsize=(8, 6))
        plt.boxplot([errors_cache, errors_no_cache], labels=["With Cache", "No Cache"])
        plt.title("Action Error Comparison")
        plt.ylabel("MSE")
        plt.savefig(os.path.join(args.output_dir, "error_comparison.png"))
    
    # 如果有缓存命中率数据，绘制随时间变化的命中率
    if "cache_hit_ratios" in results_cache and results_cache["cache_hit_ratios"]:
        plt.figure(figsize=(8, 6))
        plt.plot(results_cache["cache_hit_ratios"])
        plt.title("Cache Hit Ratio Over Time")
        plt.xlabel("Sample")
        plt.ylabel("Hit Ratio")
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "cache_hit_ratio.png"))

def save_results(results_cache, results_no_cache, avg_results_cache, avg_results_no_cache, args):
    """保存评估结果"""
    print("Saving results...")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存详细结果
    np.save(os.path.join(args.output_dir, "results_cache.npy"), results_cache)
    np.save(os.path.join(args.output_dir, "results_no_cache.npy"), results_no_cache)
    
    # 保存摘要结果
    with open(os.path.join(args.output_dir, "summary_results.txt"), "w") as f:
        f.write("===== Results with VLA-Cache =====\n")
        for key, value in avg_results_cache.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n===== Results without VLA-Cache =====\n")
        for key, value in avg_results_no_cache.items():
            f.write(f"{key}: {value}\n")
        
        # 计算性能改进
        if "avg_inference_time" in avg_results_cache and "avg_inference_time" in avg_results_no_cache:
            time_improv = (1 - avg_results_cache["avg_inference_time"] / avg_results_no_cache["avg_inference_time"]) * 100
            f.write(f"\nInference time improvement: {time_improv:.2f}%\n")
        
        if "avg_memory_usage" in avg_results_cache and "avg_memory_usage" in avg_results_no_cache:
            mem_improv = (1 - avg_results_cache["avg_memory_usage"] / avg_results_no_cache["avg_memory_usage"]) * 100
            f.write(f"Memory usage improvement: {mem_improv:.2f}%\n")
        
        if "avg_action_error" in avg_results_cache and "avg_action_error" in avg_results_no_cache:
            error_diff = (avg_results_cache["avg_action_error"] - avg_results_no_cache["avg_action_error"]) / avg_results_no_cache["avg_action_error"] * 100
            f.write(f"Action error change: {error_diff:.2f}%\n")
        
        if "avg_cache_hit_ratio" in avg_results_cache:
            f.write(f"Average cache hit ratio: {avg_results_cache['avg_cache_hit_ratio']:.2f}\n")

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 准备评估数据
    eval_data = prepare_data(args)
    
    # 评估启用缓存的模型
    print("\n=== Evaluating with VLA-Cache ===")
    args.no_cache = False
    model_cache, tokenizer = load_model_and_tokenizer(args)
    results_cache, avg_results_cache = evaluate_model(model_cache, tokenizer, eval_data, args)
    
    # 清除模型以释放内存
    del model_cache
    torch.cuda.empty_cache()
    
    # 评估不使用缓存的模型
    print("\n=== Evaluating without VLA-Cache ===")
    args.no_cache = True
    model_no_cache, _ = load_model_and_tokenizer(args)
    results_no_cache, avg_results_no_cache = evaluate_model(model_no_cache, tokenizer, eval_data, args)
    
    # 绘制结果
    plot_results(results_cache, results_no_cache, args)
    
    # 保存结果
    save_results(results_cache, results_no_cache, avg_results_cache, avg_results_no_cache, args)
    
    print(f"\nEvaluation completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 