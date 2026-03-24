"""
合并审核后的标注到训练集
将审核修正后的自动标注合并到原始训练数据中
"""

import json
import shutil
from pathlib import Path
import argparse

def merge_labels(auto_label_dir, train_dir, reviewed_only=True):
    """
    合并标注文件
    
    参数:
        auto_label_dir: 自动标注目录
        train_dir: 训练集目录
        reviewed_only: 是否只合并已审核的文件（通过检查是否有修改）
    """
    auto_path = Path(auto_label_dir)
    train_path = Path(train_dir)
    
    json_files = list(auto_path.glob("*.json"))
    
    merged_count = 0
    skipped_count = 0
    
    print(f"找到 {len(json_files)} 个自动标注文件")
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data.get('shapes'):
            skipped_count += 1
            continue
        
        image_name = json_file.stem
        src_image = auto_path / f"{image_name}.jpg"
        
        if not src_image.exists():
            for ext in ['.png', '.jpeg', '.JPG', '.PNG']:
                alt = auto_path / f"{image_name}{ext}"
                if alt.exists():
                    src_image = alt
                    break
        
        dst_json = train_path / json_file.name
        dst_image = train_path / src_image.name
        
        if dst_json.exists():
            with open(dst_json, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            
            if existing.get('shapes') and not data.get('description', '').startswith('Auto-labeled'):
                skipped_count += 1
                continue
        
        shutil.copy(json_file, dst_json)
        
        if src_image.exists() and not dst_image.exists():
            shutil.copy(src_image, dst_image)
        
        merged_count += 1
    
    print(f"\n合并完成:")
    print(f"  已合并: {merged_count} 个文件")
    print(f"  已跳过: {skipped_count} 个文件")
    
    return merged_count

def check_label_quality(json_dir):
    """检查标注质量"""
    json_path = Path(json_dir)
    json_files = list(json_path.glob("*.json"))
    
    empty_count = 0
    multi_box_count = 0
    total_boxes = 0
    
    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        shapes = data.get('shapes', [])
        if not shapes:
            empty_count += 1
        elif len(shapes) > 1:
            multi_box_count += 1
        
        total_boxes += len(shapes)
    
    print(f"\n标注质量统计:")
    print(f"  总文件数: {len(json_files)}")
    print(f"  空标注: {empty_count}")
    print(f"  多框标注: {multi_box_count}")
    print(f"  总检测框: {total_boxes}")
    print(f"  平均框数: {total_boxes/len(json_files):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='合并标注文件')
    parser.add_argument('--auto', type=str, default='../auto_labels',
                        help='自动标注目录')
    parser.add_argument('--train', type=str, default='../image5/train',
                        help='训练集目录')
    parser.add_argument('--check', action='store_true',
                        help='仅检查标注质量')
    
    args = parser.parse_args()
    
    if args.check:
        check_label_quality(args.auto)
    else:
        merge_labels(args.auto, args.train)
