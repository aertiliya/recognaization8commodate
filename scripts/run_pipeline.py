"""
完整 pipeline 脚本 - 商品识别全流程 (Kaggle 非交互式完美版)
阶段 A: JSON -> YOLO 格式转换
阶段 B: 训练 YOLO11 目标检测器
阶段 C: 裁剪 GT-ROI (真值框区域)
阶段 D: 训练 ConvNeXt V2 分类器
阶段 E: 端到端测试集联合评估 (出报告指标)
"""

import os
import subprocess
import argparse
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
    parser = argparse.ArgumentParser(description="商品识别完整 Pipeline (非交互式)")
    
    # 核心控制开关：通过 --stages A C D E 来控制执行哪些阶段
    parser.add_argument('--stages', nargs='+', choices=['A', 'B', 'C', 'D', 'E'], default=[], 
                        help="指定要执行的阶段，例如: --stages C D E")
    
    # 路径配置 (使用 Kaggle 的绝对路径作为默认值)
    parser.add_argument('--test_images', default='/kaggle/input/recognaization8commodate/yolo_dataset/test/images')
    parser.add_argument('--test_labels', default='/kaggle/input/recognaization8commodate/yolo_dataset/test/labels')
    parser.add_argument('--yolo_pt', default='/kaggle/input/recognaization8commodate/scripts/best.pt')
    parser.add_argument('--conv_pth', default='/kaggle/input/recognaization8commodate/convnext_models/best_model.pth')
    parser.add_argument('--epochs', default='50', help="ConvNeXt 的训练轮数")
    
    args = parser.parse_args()
    
    scripts_dir = Path(__file__).parent
    project_dir = scripts_dir.parent
    
    print("\n" + "🚀"*20)
    print(" 商品识别完整 Pipeline (YOLO11 + ConvNeXt V2) - 非交互式版")
    print(f" 当前计划执行的阶段: {args.stages if args.stages else '无 (请在命令中加 --stages)'}")
    print("🚀"*20)
    
    # ============== 阶段 A: JSON 转 YOLO ==============
    if 'A' in args.stages:
        output_dir = os.path.abspath("yolo_dataset")
        cmd1 = f'python "{scripts_dir / "01_convert_json_to_yolo.py"}" --source image5/train --output {output_dir} --train_ratio 1.0'
        run_command(cmd1, cwd=project_dir, description="[阶段 A] 转换 Train 集")
        
        cmd2 = f'python "{scripts_dir / "01_convert_json_to_yolo.py"}" --source image5/val --output {output_dir} --train_ratio 1.0'
        run_command(cmd2, cwd=project_dir, description="[阶段 A] 转换 Val 集")
    else:
        print("\n⏭️ 跳过阶段 A")

    # ============== 阶段 B: YOLO 训练提示 ==============
    if 'B' in args.stages:
        print("\n[阶段 B] 提示: 建议在 Kaggle 中直接新建一个 Cell 运行 YOLO 训练代码 (yolo task=detect mode=train...)。")
    else:
        print("⏭️ 跳过阶段 B")

    # ============== 阶段 C: 裁剪 GT-ROI ==============
    if 'C' in args.stages:
        cmd = f'python "{scripts_dir / "crop_gt_roi.py"}"'
        run_command(cmd, cwd=project_dir, description="[阶段 C] 裁剪 GT-ROI")
    else:
        print("⏭️ 跳过阶段 C")

    # ============== 阶段 D: 训练 ConvNeXt V2 ==============
    if 'D' in args.stages:
        cmd = (f'python "{scripts_dir / "train_convnext_roi.py"}" '
               f'--roi_dir gt_roi_dataset '
               f'--output_dir convnext_models '
               f'--epochs {args.epochs}')
        run_command(cmd, cwd=project_dir, description=f"[阶段 D] 训练 ConvNeXt V2 ({args.epochs} epochs)")
    else:
        print("⏭️ 跳过阶段 D")

    # ============== 阶段 E: 端到端联合评估 ==============
    if 'E' in args.stages:
        cmd = (f'python "{scripts_dir / "evaluate_end2end.py"}" '
               f'--test_images {args.test_images} '
               f'--test_labels {args.test_labels} '
               f'--yolo_model {args.yolo_pt} '
               f'--convnext_model {args.conv_pth}')
        run_command(cmd, cwd=project_dir, description="[阶段 E] 执行端到端性能评估")
    else:
        print("⏭️ 跳过阶段 E")

    print("\n" + "🎉"*20)
    print(" 全部指定流程结束！")
    print("🎉"*20)

if __name__ == "__main__":
    main()