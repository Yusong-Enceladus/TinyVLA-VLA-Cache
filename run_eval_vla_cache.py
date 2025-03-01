#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TinyVLA-Cache简化评估脚本

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
from llava_pythia.model.builder import load_pretrained_model

def parse_args():
    parser = argparse.ArgumentParser(description='评估VLA-Cache对TinyVLA性能的影响')
    parser.add_argument('--model_path', type=str, required=True,
                        help='TinyVLA模型路径')
    parser.add_argument('--task_name', type=str, default='sim_transfer_cube',
                        help='评估任务名称')
    parser.add_argument('--num_samples', type=int, default=10,
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

def prepare_dummy_data(args, num_samples=10):
    """准备模拟评估数据"""
    print(f"Preparing dummy evaluation data for testing")
    
    # 创建模拟数据
    eval_data = []
    for i in range(num_samples):
        sample = {
            "language_instruction": f"Pick up the cube and place it on the target location {i}",
            "images": torch.randn(5, 3, 224, 224),  # 模拟图像数据
            "states": torch.randn(5, 16),  # 模拟状态数据
            "actions": torch.randn(5, 14)  # 模拟动作数据
        }
        eval_data.append(sample)
    
    print(f"Created {len(eval_data)} dummy evaluation samples")
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
            images = images.to(args.device)
        states = sample.get("states", None)
        if states is not None:
            states = states.to(args.device)
        
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
            true_actions = sample["actions"].to(args.device)
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

def plot_results(results, args):
    """绘制结果图表"""
    print("Plotting results...")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 准备数据
    times = results["inference_times"]
    memory = results["memory_usage"]
    
    # 绘制推理时间
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(times)), times)
    plt.title("Inference Time")
    plt.xlabel("Sample")
    plt.ylabel("Time (seconds)")
    
    # 绘制内存使用
    plt.subplot(1, 2, 2)
    plt.plot(range(len(memory)), memory)
    plt.title("Memory Usage")
    plt.xlabel("Sample")
    plt.ylabel("Memory (MB)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "performance.png"))
    
    # 如果有缓存命中率数据，绘制缓存命中率
    if "cache_hit_ratios" in results and results["cache_hit_ratios"]:
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(results["cache_hit_ratios"])), results["cache_hit_ratios"])
        plt.title("Cache Hit Ratio")
        plt.xlabel("Sample")
        plt.ylabel("Hit Ratio")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "cache_hit_ratio.png"))

def save_results(results, avg_results, args):
    """保存评估结果"""
    print("Saving results...")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存详细结果
    np.save(os.path.join(args.output_dir, "results.npy"), results)
    
    # 保存平均结果
    with open(os.path.join(args.output_dir, "avg_results.txt"), "w") as f:
        f.write("Average Results:\n")
        for key, value in avg_results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Results saved to {args.output_dir}")

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(args)
    
    # 准备评估数据
    eval_data = prepare_dummy_data(args, args.num_samples)
    
    # 评估模型
    results, avg_results = evaluate_model(model, tokenizer, eval_data, args)
    
    # 绘制结果
    plot_results(results, args)
    
    # 保存结果
    save_results(results, avg_results, args)
    
    print("Evaluation complete!")
    print("Average Results:")
    for key, value in avg_results.items():
        print(f"- {key}: {value}")

if __name__ == "__main__":
    main() 