"""
一键运行脚本 - YOLO11训练工作流程
按顺序执行：数据转换 -> YOLO11训练
数据来源: image5/train/
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, cwd=None):
    """运行命令并实时显示输出"""
    print(f"\n{'='*60}")
    print(f"执行: {cmd}")
    print('='*60)
    
    process = subprocess.Popen(
        cmd,
        shell=True,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def main():
    scripts_dir = Path(__file__).parent
    project_dir = scripts_dir.parent
    
    print("\n" + "="*60)
    print("YOLO11 商品检测 - 训练工作流程")
    print("="*60)
    
    print("\n当前项目结构:")
    print(f"  项目目录: {project_dir}")
    print(f"  脚本目录: {scripts_dir}")
    
    print("\n" + "-"*60)
    print("步骤 1: 转换标注格式 (JSON -> YOLO)")
    print("-"*60)
    
    cmd = f'python "{scripts_dir / "01_convert_json_to_yolo.py"}" --source /kaggle/input/models/cartiliya/2/pytorch/default/1/image5 --output /kaggle/working/yolo_dataset'
    ret = run_command(cmd, cwd=scripts_dir)
    if ret != 0:
        print("转换失败!")
        return
    
    print("\n" + "-"*60)
    print("步骤 2: 训练 YOLO11 模型")
    print("-"*60)
    
    response = input("\n是否开始训练? (y/n): ").strip().lower()
    if response == 'y':
        epochs = input("训练轮数 (默认100): ").strip()
        epochs = int(epochs) if epochs else 100
        
        model_size = input("模型大小 n/s/m/l/x (默认s): ").strip().lower()
        model_size = model_size if model_size in ['n', 's', 'm', 'l', 'x'] else 's'
        
        cmd = f'python "{scripts_dir / "02_train_yolo11.py"}" --mode train --model_size {model_size} --epochs {epochs}'
        ret = run_command(cmd, cwd=scripts_dir)
        if ret != 0:
            print("训练失败!")
            return
    else:
        print("跳过训练步骤")
    
    print("\n" + "="*60)
    print("YOLO11训练工作流程完成!")
    print("="*60)
    print("\n后续步骤:")
    print("1. 查看训练结果: runs/detect/product_detector/")
    print("2. 使用训练好的模型进行推理测试")
    print("3. 根据需要调整超参数重新训练")

if __name__ == "__main__":
    main()
