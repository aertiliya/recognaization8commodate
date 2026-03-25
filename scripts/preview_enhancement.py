"""
随机采样3张图片进行增强效果展示
"""

import cv2
import numpy as np
import os
import random
import glob

# --- 配置部分 ---
# 使用相对于当前工作目录的路径，避免中文路径编码问题
input_dir = r"image5\train"
output_dir = r"enhance_preview"

# 图像处理参数
TARGET_SIZE = (224, 224)
BLACK_THRESHOLD = 5
INPAINT_RADIUS = 3
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)
UNSHARP_SIGMA = 1.0
UNSHARP_WEIGHT = 1.5


def process_and_enhance_image(img):
    """
    核心处理流水线：
    1. 寻找并修复人为添加的黑色方块
    2. 增强对比度 (CLAHE)
    3. 自适应锐化
    4. 调整尺寸
    """
    if img is None:
        return None

    # --- 1. 去除黑色方块的影响 (图像修复) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, black_mask = cv2.threshold(gray, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(black_mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)
            
    if np.sum(final_mask) > 0:
        inpainted_img = cv2.inpaint(img, final_mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    else:
        inpainted_img = img

    # --- 2. 图像清晰度增强 ---
    lab = cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    l_enhanced = clahe.apply(l)

    gaussian = cv2.GaussianBlur(l_enhanced, (0, 0), UNSHARP_SIGMA)
    l_sharpened = cv2.addWeighted(l_enhanced, 1 + UNSHARP_WEIGHT, gaussian, -UNSHARP_WEIGHT, 0)

    enhanced_lab = cv2.merge((l_sharpened, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # --- 3. 尺寸调整 (Resize) ---
    resized_img = cv2.resize(enhanced_bgr, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)

    return resized_img


def create_comparison_image(original, enhanced, filename):
    """创建对比图：原图 vs 增强后"""
    # 统一尺寸用于显示
    orig_resized = cv2.resize(original, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
    
    # 添加文字标签
    orig_label = orig_resized.copy()
    enhanced_label = enhanced.copy()
    
    cv2.putText(orig_label, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(enhanced_label, "Enhanced", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 水平拼接
    comparison = np.hstack([orig_label, enhanced_label])
    return comparison


def main():
    print(f"随机采样3张图片进行增强预览...")
    print(f"源目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    # 获取所有图片文件 (使用glob避免编码问题)
    image_pattern = os.path.join(input_dir, "*.jpg")
    all_files = glob.glob(image_pattern)
    
    # 只保留文件名部分
    all_files = [os.path.basename(f) for f in all_files]
    
    if len(all_files) == 0:
        print(f"❌ 目录中没有找到图片文件: {input_dir}")
        print(f"   尝试查找: {image_pattern}")
        return
    
    print(f"共找到 {len(all_files)} 张图片")
    
    # 随机选择3张
    sample_files = random.sample(all_files, min(3, len(all_files)))
    print(f"随机选择的图片: {sample_files}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每张图片
    for i, filename in enumerate(sample_files, 1):
        src_path = os.path.join(input_dir, filename)
        
        # 读取原图
        original = cv2.imread(src_path)
        if original is None:
            print(f"❌ 无法读取图片: {filename}")
            continue
        
        # 增强处理
        enhanced = process_and_enhance_image(original)
        
        # 创建对比图
        comparison = create_comparison_image(original, enhanced, filename)
        
        # 保存对比图
        output_path = os.path.join(output_dir, f"comparison_{i:02d}_{filename}")
        cv2.imwrite(output_path, comparison)
        
        # 单独保存增强后的图片
        enhanced_path = os.path.join(output_dir, f"enhanced_{i:02d}_{filename}")
        cv2.imwrite(enhanced_path, enhanced)
        
        print(f"✅ 处理完成: {filename}")
        print(f"   对比图: {output_path}")
        print(f"   增强图: {enhanced_path}")
    
    print(f"\n🎉 预览完成！所有结果保存在: {output_dir}")
    print(f"包含对比图 (原图+增强图并排) 和单独的增强后图片")


if __name__ == "__main__":
    main()
