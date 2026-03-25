"""
完整 pipeline 脚本 - 商品识别全流程
阶段 A: JSON -> YOLO 格式转换
阶段 B: 裁剪 GT-ROI (真实框区域)
阶段 C: 训练 ConvNeXt V2 分类器
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, cwd=None, description=""):
    """运行命令并实时显示输出"""
    if description:
        print(f"\n{'='*60}")
        print(f"{description}")
        print('='*60)
    print(f"执行: {cmd}")
    
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
    
    print("\n" + "="*70)
    print("商品识别完整 Pipeline")
    print("="*70)
    print("流程: JSON转YOLO -> 裁剪GT-ROI -> 训练ConvNeXt V2")
    print(f"项目目录: {project_dir}")
    print(f"脚本目录: {scripts_dir}")
    
    # ============== 阶段 A: JSON 转 YOLO ==============
    print("\n" + "-"*70)
    print("阶段 A: JSON 标签转换为 YOLO 格式")
    print("-"*70)
    print("选项 1: 自动划分 - 从单个目录按比例划分 train/val")
    print("选项 2: 分别处理 - 已手动划分好 train/val 目录")
    
    response = input("\n是否执行阶段 A (JSON -> YOLO)? (y/n): ").strip().lower()
    if response == 'y':
        mode = input("选择模式 (1=自动划分, 2=分别处理, 默认=2): ").strip()
        
        # 默认使用模式2（分别处理）
        if not mode:
            mode = '2'
        
        output_dir = os.path.abspath("yolo_dataset")
        
        if mode == '1':
            # 自动划分模式
            source_dir = "image5/train"
            cmd = f'python "{scripts_dir / "01_convert_json_to_yolo.py"}" --source {source_dir} --output {output_dir} --train_ratio 0.8'
            ret = run_command(cmd, cwd=project_dir, description="转换 JSON 到 YOLO 格式 (自动划分)")
        else:
            # 分别处理模式：分别转换 train 和 val
            train_dir = "image5/train"
            val_dir = "image5/val"
            
            # 先转换 train (100%)
            cmd = f'python "{scripts_dir / "01_convert_json_to_yolo.py"}" --source {train_dir} --output {output_dir} --train_ratio 1.0'
            ret = run_command(cmd, cwd=project_dir, description="转换 Train 集")
            
            if ret == 0:
                # 再转换 val (作为第二个批次)
                cmd = f'python "{scripts_dir / "01_convert_json_to_yolo.py"}" --source {val_dir} --output {output_dir} --train_ratio 1.0'
                ret = run_command(cmd, cwd=project_dir, description="转换 Val 集")
        
        if ret != 0:
            print("阶段 A 失败!")
            return
        print("阶段 A 完成!")
        print(f"输出路径: {output_dir}")
    else:
        print("跳过阶段 A")
    
    # ============== 阶段 B: 裁剪 GT-ROI ==============
    print("\n" + "-"*70)
    print("阶段 B: 裁剪 GT-ROI (真实框区域)")
    print("-"*70)
    print("从 JSON 标签中读取边界框，裁剪出商品 ROI，按 8 个类别分类保存")
    
    response = input("\n是否执行阶段 B (裁剪 GT-ROI)? (y/n): ").strip().lower()
    if response == 'y':
        cmd = f'python "{scripts_dir / "crop_gt_roi.py"}"'
        ret = run_command(cmd, cwd=project_dir, description="裁剪 GT-ROI")
        if ret != 0:
            print("阶段 B 失败!")
            return
        print("阶段 B 完成!")
    else:
        print("跳过阶段 B")
    
    # ============== 阶段 C: 训练 ConvNeXt V2 ==============
    print("\n" + "-"*70)
    print("阶段 C: 训练 ConvNeXt V2 分类模型")
    print("-"*70)
    print("使用裁剪后的 GT-ROI 图像训练分类器")
    
    response = input("\n是否执行阶段 C (训练 ConvNeXt)? (y/n): ").strip().lower()
    if response == 'y':
        epochs = input("训练轮数 (默认50): ").strip()
        epochs = int(epochs) if epochs else 50
        
        batch_size = input("批次大小 (默认32): ").strip()
        batch_size = int(batch_size) if batch_size else 32
        
        model_name = input("模型名称 (默认convnextv2_atto): ").strip()
        model_name = model_name if model_name else 'convnextv2_atto'
        
        device = input("训练设备 (默认cuda): ").strip()
        device = device if device else 'cuda'
        
        roi_dir = "gt_roi_dataset"
        output_dir = "convnext_models"
        
        cmd = (f'python "{scripts_dir / "train_convnext_roi.py"}" '
               f'--roi_dir {roi_dir} '
               f'--output_dir {output_dir} '
               f'--model_name {model_name} '
               f'--epochs {epochs} '
               f'--batch_size {batch_size} '
               f'--device {device}')
        
        ret = run_command(cmd, cwd=project_dir, 
                         description=f"训练 ConvNeXt V2 ({model_name}, {epochs} epochs)")
        if ret != 0:
            print("阶段 C 失败!")
            return
        print("阶段 C 完成!")
    else:
        print("跳过阶段 C")
    
    # ============== 完成总结 ==============
    print("\n" + "="*70)
    print("Pipeline 执行完成!")
    print("="*70)
    print("\n输出文件:")
    print(f"  1. YOLO 数据集: yolo_dataset/")
    print(f"  2. GT-ROI 数据集: gt_roi_dataset/")
    print(f"  3. 训练好的模型: convnext_models/")

if __name__ == "__main__":
    main()
