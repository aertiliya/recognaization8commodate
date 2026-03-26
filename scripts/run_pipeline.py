import os, subprocess, argparse
from pathlib import Path

def run_cmd(cmd, cwd, desc=""):
    print(f"\n{'='*60}\n{desc}\n{'='*60}\n执行: {cmd}")
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout: print(line, end='')
    proc.wait()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stages', nargs='+', choices=['A', 'B', 'C', 'D', 'E'], default=[])
    
    BASE = '/kaggle/working/recognaization8commodate'
    parser.add_argument('--test_images', default=f'{BASE}/yolo_dataset/test/images')
    parser.add_argument('--test_labels', default=f'{BASE}/yolo_dataset/test/labels')
    parser.add_argument('--yolo_pt', default=f'{BASE}/scripts/best.pt')
    parser.add_argument('--conv_pth', default=f'{BASE}/convnext_models/best_model.pth')
    parser.add_argument('--epochs', default='20')
    parser.add_argument('--multi_gpu', action='store_true', help='开启双卡/多卡训练模式')
    
    args = parser.parse_args()
    scripts = Path(__file__).parent
    
    # 构造多卡参数后缀
    gpu_flag = "--multi_gpu" if args.multi_gpu else ""
    
    if 'A' in args.stages:
        run_cmd(f'python "{scripts/"01_convert_json_to_yolo.py"}" --source image5/train --output "{BASE}/yolo_dataset" --train_ratio 1.0', BASE, "[A] 转换 Train")
        run_cmd(f'python "{scripts/"01_convert_json_to_yolo.py"}" --source image5/val --output "{BASE}/yolo_dataset" --train_ratio 1.0', BASE, "[A] 转换 Val")
        run_cmd(f'python "{scripts/"01_convert_json_to_yolo.py"}" --source image5/test --output "{BASE}/yolo_dataset" --train_ratio 1.0', BASE, "[A] 转换 Test")
    
    if 'C' in args.stages:
        run_cmd(f'python "{scripts/"crop_gt_roi.py"}" --input "{BASE}/image5/train" --output "{BASE}/gt_roi_dataset/train"', BASE, "[C] 裁剪 Train 集")
        run_cmd(f'python "{scripts/"crop_gt_roi.py"}" --input "{BASE}/image5/val" --output "{BASE}/gt_roi_dataset/val"', BASE, "[C] 裁剪 Val 集")
    
    if 'D' in args.stages:
        run_cmd(f'python "{scripts/"train_convnext_roi.py"}" --roi_dir "{BASE}/gt_roi_dataset" --output_dir "{BASE}/convnext_models" --epochs {args.epochs} --lr 0.0001 {gpu_flag}', BASE, "[D] 训练分类器")
        
    if 'E' in args.stages:
        run_cmd(f'python "{scripts/"evaluate_end2end.py"}" --test_images "{BASE}/yolo_dataset/test/images" --test_labels "{BASE}/yolo_dataset/test/labels" --yolo_model "{args.yolo_pt}" --convnext_model "{args.conv_pth}"', BASE, "[E] 端到端评估")

if __name__ == "__main__":
    main()