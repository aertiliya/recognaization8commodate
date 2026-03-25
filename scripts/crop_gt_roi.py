"""
第二步：裁剪 GT-ROI（真实框区域）
从 JSON 标签中读取边界框，裁剪出商品 ROI，并按类别分类保存
"""

import cv2
import json
import os
import glob
from tqdm import tqdm
import numpy as np

# --- 配置 ---
# 输入路径（原始图片和JSON标签）
input_image_dir = r"image5\train"  # 包含 .jpg 和 .json 文件的目录
# 输出路径（裁剪后的GT-ROI）
output_roi_dir = r"gt_roi_dataset"

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

# 反转映射：ID -> 名称
ID_TO_CLASS = {v: k for k, v in CLASS_NAMES.items()}

# ROI 输出尺寸（ConvNeXt 输入尺寸）
ROI_SIZE = (224, 224)


def parse_json_label(json_path):
    """
    解析 X-AnyLabeling 的 JSON 标签文件
    返回: [(label_name, bbox), ...]
    bbox格式: [x_min, y_min, x_max, y_max]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    objects = []
    for shape in data.get('shapes', []):
        label = shape.get('label', '')
        points = shape.get('points', [])
        
        # 只处理我们关心的8个类别
        if label not in CLASS_NAMES:
            continue
        
        # 计算边界框 [x_min, y_min, x_max, y_max]
        if len(points) >= 2:
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            objects.append((label, [x_min, y_min, x_max, y_max]))
    
    return objects


def crop_roi(image, bbox, target_size=(224, 224)):
    """
    从图像中裁剪 ROI 并调整尺寸
    bbox: [x_min, y_min, x_max, y_max]
    """
    h, w = image.shape[:2]
    x_min, y_min, x_max, y_max = map(int, bbox)
    
    # 边界检查，防止越界
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)
    
    # 确保有有效区域
    if x_max <= x_min or y_max <= y_min:
        return None
    
    # 裁剪
    roi = image[y_min:y_max, x_min:x_max]
    
    # 调整尺寸
    roi_resized = cv2.resize(roi, target_size, interpolation=cv2.INTER_CUBIC)
    
    return roi_resized


def process_dataset():
    """
    处理整个数据集：读取所有图片和对应的JSON标签，裁剪ROI并分类保存
    """
    print("=" * 60)
    print("第二步：裁剪 GT-ROI（真实框区域）")
    print("=" * 60)
    
    # 查找所有 JSON 文件
    json_pattern = os.path.join(input_image_dir, "*.json")
    json_files = glob.glob(json_pattern)
    
    print(f"找到 {len(json_files)} 个标签文件")
    
    # 创建输出目录结构
    for class_name in CLASS_NAMES.keys():
        class_dir = os.path.join(output_roi_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    # 统计信息
    stats = {name: 0 for name in CLASS_NAMES.keys()}
    total_crops = 0
    
    # 处理每个 JSON 文件
    for json_path in tqdm(json_files, desc="处理标签文件"):
        # 获取对应的图片路径
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        img_path = os.path.join(input_image_dir, base_name + ".jpg")
        
        # 检查图片是否存在
        if not os.path.exists(img_path):
            # 尝试其他扩展名
            for ext in ['.jpeg', '.png', '.bmp']:
                alt_path = os.path.join(input_image_dir, base_name + ext)
                if os.path.exists(alt_path):
                    img_path = alt_path
                    break
        
        if not os.path.exists(img_path):
            print(f"❌ 找不到图片: {base_name}")
            continue
        
        # 读取图片
        image = cv2.imread(img_path)
        if image is None:
            print(f"❌ 无法读取图片: {img_path}")
            continue
        
        # 解析 JSON 标签
        objects = parse_json_label(json_path)
        
        if not objects:
            continue
        
        # 裁剪每个目标
        for idx, (label, bbox) in enumerate(objects):
            roi = crop_roi(image, bbox, ROI_SIZE)
            
            if roi is not None:
                # 保存裁剪结果
                output_filename = f"{base_name}_roi{idx}.jpg"
                output_path = os.path.join(output_roi_dir, label, output_filename)
                cv2.imwrite(output_path, roi)
                
                stats[label] += 1
                total_crops += 1
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("裁剪完成统计：")
    print("=" * 60)
    for class_name, count in stats.items():
        print(f"  {class_name:15s}: {count:5d} 张")
    print(f"  {'总计':15s}: {total_crops:5d} 张")
    print(f"\n✅ GT-ROI 数据集已保存到: {output_roi_dir}")
    print("目录结构：")
    for class_name in CLASS_NAMES.keys():
        print(f"  {output_roi_dir}/{class_name}/")


if __name__ == "__main__":
    process_dataset()
