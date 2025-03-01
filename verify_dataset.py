#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据集验证脚本

该脚本用于验证ALOHA模拟数据集是否可以正确加载。
"""

import os
import sys
import numpy as np
from aloha_scripts.constants import TASK_CONFIGS

# 检查数据集目录是否存在
def check_dataset_dir(task_name):
    task_config = TASK_CONFIGS[task_name]
    dataset_dirs = task_config['dataset_dir']
    
    for dir_path in dataset_dirs:
        if not os.path.exists(dir_path):
            print(f"错误: 数据集目录 '{dir_path}' 不存在!")
            return False
        else:
            print(f"成功: 找到数据集目录 '{dir_path}'")
            # 检查目录中的文件
            files = os.listdir(dir_path)
            print(f"目录中包含 {len(files)} 个文件")
            if len(files) > 0:
                print(f"示例文件: {files[:5]}")
    
    return True

# 尝试加载数据
def try_load_data(task_name):
    try:
        # 动态导入，避免在验证目录存在性之前就导入
        from data_utils.datasets import load_data
        import os
        
        task_config = TASK_CONFIGS[task_name]
        dataset_dir = task_config['dataset_dir']
        camera_names = task_config['camera_names']
        episode_len = task_config.get('episode_len', 1000)
        
        print(f"尝试加载数据集: {task_name}")
        print(f"- 数据集目录: {dataset_dir}")
        print(f"- 相机名称: {camera_names}")
        print(f"- 片段长度: {episode_len}")
        
        # 定义一个简单的name_filter函数
        def name_filter(filename):
            return True  # 接受所有文件
        
        # 创建一个简单的配置字典
        config = {
            'training_args': type('obj', (object,), {
                'pretrain_image_size': 480
            })
        }
        
        # 加载数据
        train_data, val_data = load_data(
            dataset_dir_l=dataset_dir,
            name_filter=name_filter,
            camera_names=camera_names,
            batch_size_train=1,
            batch_size_val=1,
            chunk_size=100,
            config=config,
            skip_mirrored_data=True,
            policy_class="ACT",
            return_dataset=True
        )
        
        print(f"成功: 加载了 {len(train_data)} 个训练样本和 {len(val_data)} 个验证样本")
        
        # 检查数据结构
        if len(train_data) > 0:
            sample = train_data[0]
            print("\n数据样本结构:")
            for key, value in sample.items():
                if isinstance(value, np.ndarray):
                    print(f"- {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"- {key}: {type(value)}")
        
        return True
    except Exception as e:
        print(f"错误: 加载数据时出现异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    task_name = 'sim_transfer_cube'
    
    print(f"验证数据集: {task_name}")
    print("-" * 50)
    
    # 检查数据集目录
    if not check_dataset_dir(task_name):
        print("验证失败: 数据集目录不存在")
        return
    
    # 尝试加载数据
    if try_load_data(task_name):
        print("\n验证成功: 数据集可以正确加载")
    else:
        print("\n验证失败: 无法加载数据集")

if __name__ == "__main__":
    main() 