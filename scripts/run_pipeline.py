"""
完整 pipeline 脚本 - 商品识别全流程 (上机作业完美版)
阶段 A: JSON -> YOLO 格式转换
阶段 B: 训练 YOLO11 目标检测器
阶段 C: 裁剪 GT-ROI (真值框区域)
阶段 D: 训练 ConvNeXt V2 分类器
阶段 E: 端到端测试集联合评估 (出报告指标)
"""

import os
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
    
    print("\n" + "🚀"*20)
    print(" 商品识别完整 Pipeline (YOLO11 + ConvNeXt V2)")
    print("🚀"*20)
    
    # ============== 阶段 A: JSON 转 YOLO ==============
    print("\n" + "-"*70)
    print("阶段 A: JSON 标签转换为 YOLO 格式")
    response = input("\n是否执行阶段 A? (y/n, 默认 n): ").strip().lower()
    if response == 'y':
        # 默认使用分别处理模式
        output_dir = os.path.abspath("yolo_dataset")
        train_dir = "image5/train"
        val_dir = "image5/val"
        
        cmd1 = f'python "{scripts_dir / "01_convert_json_to_yolo.py"}" --source {train_dir} --output {output_dir} --train_ratio 1.0'
        run_command(cmd1, cwd=project_dir, description="转换 Train 集")
        
        cmd2 = f'python "{scripts_dir / "01_convert_json_to_yolo.py"}" --source {val_dir} --output {output_dir} --train_ratio 1.0'
        run_command(cmd2, cwd=project_dir, description="转换 Val 集")
    else:
        print("⏭️ 跳过阶段 A")

    # ============== 阶段 B: YOLO 训练提示 ==============
    print("\n" + "-"*70)
    print("阶段 B: 训练 YOLO11 目标检测器")
    print("请确认你已经使用 ultralytics 训练了 YOLO 模型并得到了 best.pt")
    response = input("是否要在此时启动 YOLO 训练? (y/n, 默认 n): ").strip().lower()
    if response == 'y':
        yaml_path = input("请输入你的 data.yaml 路径: ").strip()
        epochs = input("训练轮数 (默认 50): ").strip() or "50"
        cmd = f'yolo task=detect mode=train model=yolo11n.pt data={yaml_path} epochs={epochs} imgsz=640'
        run_command(cmd, cwd=project_dir, description="训练 YOLO11")
    else:
        print("⏭️ 跳过 YOLO 训练 (假设已准备好 best.pt)")

    # ============== 阶段 C: 裁剪 GT-ROI ==============
    print("\n" + "-"*70)
    print("阶段 C: 裁剪 GT-ROI (为 ConvNeXt 准备干净数据)")
    response = input("\n是否执行阶段 C? (y/n, 默认 n): ").strip().lower()
    if response == 'y':
        cmd = f'python "{scripts_dir / "crop_gt_roi.py"}"'
        run_command(cmd, cwd=project_dir, description="裁剪 GT-ROI")
    else:
        print("⏭️ 跳过阶段 C")

    # ============== 阶段 D: 训练 ConvNeXt V2 ==============
    print("\n" + "-"*70)
    print("阶段 D: 训练 ConvNeXt V2 细分类模型")
    response = input("\n是否执行阶段 D? (y/n, 默认 n): ").strip().lower()
    if response == 'y':
        epochs = input("训练轮数 (默认 50): ").strip() or "50"
        cmd = (f'python "{scripts_dir / "train_convnext_roi.py"}" '
               f'--roi_dir gt_roi_dataset '
               f'--output_dir convnext_models '
               f'--epochs {epochs}')
        run_command(cmd, cwd=project_dir, description=f"训练 ConvNeXt V2 ({epochs} epochs)")
    else:
        print("⏭️ 跳过阶段 D")

    # ============== 阶段 E: 端到端联合评估 ==============
    print("\n" + "-"*70)
    print("阶段 E: 端到端测试集联合评估 (生成实验报告指标) 🌟")
    response = input("\n是否执行阶段 E (联合算分)? (y/n, 默认 y): ").strip().lower()
    if response != 'n':
        test_images = input("请输入 Test 集图片目录 (默认 yolo_dataset/test/images): ").strip() or "yolo_dataset/test/images"
        test_labels = input("请输入 Test 集标签目录 (默认 yolo_dataset/test/labels): ").strip() or "yolo_dataset/test/labels"
        yolo_pt = input("请输入 YOLO best.pt 路径 (默认 runs/detect/train/weights/best.pt): ").strip() or "runs/detect/train/weights/best.pt"
        conv_pth = input("请输入 ConvNeXt best_model.pth 路径 (默认 convnext_models/best_model.pth): ").strip() or "convnext_models/best_model.pth"
        
        cmd = (f'python "{scripts_dir / "evaluate_end2end.py"}" '
               f'--test_images {test_images} '
               f'--test_labels {test_labels} '
               f'--yolo_model {yolo_pt} '
               f'--convnext_model {conv_pth}')
        run_command(cmd, cwd=project_dir, description="执行端到端性能评估")
    else:
        print("⏭️ 跳过阶段 E")

    print("\n" + "🎉"*20)
    print(" 全部 Pipeline 流程结束！可以去写上机报告了！")
    print("🎉"*20)

if __name__ == "__main__":
    main()