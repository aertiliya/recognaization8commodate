"""
X-AnyLabeling JSON格式转换为YOLO格式
将标注的JSON文件转换为YOLO检测模型所需的txt格式
"""

import json
import os
import shutil
from pathlib import Path
import random

CLASSES = [
    "hks_large",
    "hks_small", 
    "hn_can",
    "jlb_can",
    "kkkl_can",
    "wlj_can",
    "xb_wt",
    "xb"
]

def convert_json_to_yolo(json_path, output_dir, class_mapping):
    """
    将单个JSON文件转换为YOLO格式
    YOLO格式: class_id center_x center_y width height (归一化到0-1)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    image_width = data['imageWidth']
    image_height = data['imageHeight']
    shapes = data.get('shapes', [])
    
    yolo_lines = []
    for shape in shapes:
        label = shape['label']
        if label not in class_mapping:
            print(f"警告: 未知标签 {label} 在文件 {json_path}")
            continue
        
        class_id = class_mapping[label]
        points = shape['points']
        
        if shape['shape_type'] == 'rectangle' and len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
        else:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
        
        center_x = ((x1 + x2) / 2) / image_width
        center_y = ((y1 + y2) / 2) / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        
        center_x = max(0, min(1, center_x))
        center_y = max(0, min(1, center_y))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        yolo_lines.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
    
    return yolo_lines

def process_dataset(source_dir, output_dir, train_ratio=0.8, val_ratio=0.2):
    """
    处理整个数据集，划分训练集和验证集
    """
    class_mapping = {cls: idx for idx, cls in enumerate(CLASSES)}
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    json_files = list(source_path.glob('*.json'))
    
    labeled_files = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data.get('shapes'):
            labeled_files.append(json_file)
    
    print(f"找到 {len(labeled_files)} 个已标注文件")
    
    random.shuffle(labeled_files)
    train_count = int(len(labeled_files) * train_ratio)
    train_files = labeled_files[:train_count]
    val_files = labeled_files[train_count:]
    
    def copy_files(files, split):
        for json_file in files:
            image_name = json_file.stem
            image_file = source_path / f"{image_name}.jpg"
            
            if not image_file.exists():
                for ext in ['.png', '.jpeg', '.JPG', '.PNG']:
                    alt_image = source_path / f"{image_name}{ext}"
                    if alt_image.exists():
                        image_file = alt_image
                        break
            
            if image_file.exists():
                shutil.copy(image_file, output_path / 'images' / split / image_file.name)
            
            yolo_lines = convert_json_to_yolo(json_file, output_path, class_mapping)
            
            label_file = output_path / 'labels' / split / f"{image_name}.txt"
            with open(label_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_lines))
    
    print(f"处理训练集: {len(train_files)} 个文件")
    copy_files(train_files, 'train')
    
    print(f"处理验证集: {len(val_files)} 个文件")
    copy_files(val_files, 'val')
    
    print("转换完成!")
    return len(train_files), len(val_files)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='将X-AnyLabeling JSON转换为YOLO格式')
    parser.add_argument('--source', type=str, default='../image5/train', help='源数据目录')
    parser.add_argument('--output', type=str, default='/kaggle/working/yolo_dataset', help='输出目录')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    
    args = parser.parse_args()
    
    train_count, val_count = process_dataset(args.source, args.output, args.train_ratio)
    print(f"\n数据集统计:")
    print(f"  训练集: {train_count} 张")
    print(f"  验证集: {val_count} 张")
