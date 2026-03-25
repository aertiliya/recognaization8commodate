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
    
    response = input("\n是否执行阶段 A (JSON -> YOLO)? (y/n): ").strip().lower()
    if response == 'y':
        source_dir = "image5/train"
        # 使用绝对路径以匹配 yolo_train.yaml 中的配置
        output_dir = os.path.abspath("yolo_dataset")
        
        cmd = f'python "{scripts_dir / "01_convert_json_to_yolo.py"}" --source {source_dir} --output {output_dir}'
        ret = run_command(cmd, cwd=project_dir, description="转换 JSON 到 YOLO 格式")
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
