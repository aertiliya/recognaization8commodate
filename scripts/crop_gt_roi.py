"""
第二步：裁剪 GT-ROI（真实框区域）
从 JSON 标签中读取边界框，裁剪出商品 ROI，并按类别分类保存
(已修复 Linux/Kaggle 跨平台路径问题，并支持命令行传参)
"""

import cv2
import json
import os
import glob
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path

# 类别映射（8个类别）
CLASS_NAMES = {
    'hks_large': 0,
    'hks_small': 1,
    'hn_can': 2,
    'jlb_can': 3,
    'kkkl_can': 4,
    'wlj_can': 5,
    'xb_wt': 6,
    'xb': 7
}

ROI_SIZE = (224, 224)

def parse_json_label(json_path):
    """解析 X-AnyLabeling 的 JSON 标签文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    objects = []
    for shape in data.get('shapes', []):
        label = shape.get('label', '')
        points = shape.get('points', [])
        
        if label not in CLASS_NAMES:
            continue
        
        if len(points) >= 2:
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            objects.append((label, [x_min, y_min, x_max, y_max]))
    
    return objects

def crop_roi(image, bbox, target_size=(224, 224)):
    """从图像中裁剪 ROI 并调整尺寸"""
    h, w = image.shape[:2]
    x_min, y_min, x_max, y_max = map(int, bbox)
    
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)
    
    if x_max <= x_min or y_max <= y_min:
        return None
    
    roi = image[y_min:y_max, x_min:x_max]
    roi_resized = cv2.resize(roi, target_size, interpolation=cv2.INTER_CUBIC)
    
    return roi_resized

def process_dataset(input_dir, output_dir):
    """处理数据集：读取所有图片和对应的JSON标签，裁剪ROI并分类保存"""
    print("=" * 60)
    print(f"第二步：裁剪 GT-ROI (从 {input_dir} 到 {output_dir})")
    print("=" * 60)
    
    input_path = Path(input_dir)
    # 使用 Pathlib 解决跨平台斜杠问题
    json_files = list(input_path.glob("*.json"))
    
    print(f"找到 {len(json_files)} 个标签文件")
    if len(json_files) == 0:
        print("❌ 警告：未找到任何 JSON 文件，请检查源路径是否正确！")
        return
    
    for class_name in CLASS_NAMES.keys():
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
    
    stats = {name: 0 for name in CLASS_NAMES.keys()}
    total_crops = 0
    
    for json_path in tqdm(json_files, desc="处理标签文件"):
        base_name = json_path.stem
        
        # 尝试寻找图片 (处理多种后缀)
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            temp_path = input_path / f"{base_name}{ext}"
            if temp_path.exists():
                img_path = temp_path
                break
                
        if img_path is None:
            continue
            
        image = cv2.imread(str(img_path))
        if image is None:
            continue
            
        objects = parse_json_label(str(json_path))
        
        for idx, (label, bbox) in enumerate(objects):
            roi = crop_roi(image, bbox, ROI_SIZE)
            if roi is not None:
                output_filename = f"{base_name}_roi{idx}.jpg"
                output_path = os.path.join(output_dir, label, output_filename)
                cv2.imwrite(output_path, roi)
                stats[label] += 1
                total_crops += 1
                
    print("\n" + "=" * 60)
    print("裁剪完成统计：")
    print("=" * 60)
    for class_name, count in stats.items():
        print(f"  {class_name:15s}: {count:5d} 张")
    print(f"  {'总计':15s}: {total_crops:5d} 张")
    print(f"\n✅ GT-ROI 数据集已保存到: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="裁剪 GT-ROI 脚本")
    # 默认使用 Linux 正斜杠路径
    parser.add_argument('--input', type=str, default='image5/train', help='输入图片和JSON目录')
    parser.add_argument('--output', type=str, default='gt_roi_dataset', help='输出目录')
    args = parser.parse_args()
    
    process_dataset(args.input, args.output)