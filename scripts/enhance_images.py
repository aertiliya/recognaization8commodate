"""
高级图像增强脚本 - 专门处理黑色方块噪点和严重模糊
核心功能：
1. 图像修复(Inpainting)去除黑色方块
2. CLAHE对比度增强
3. 自适应锐化
4. 统一尺寸调整
"""

import cv2
import numpy as np
import os
from tqdm import tqdm

# --- 配置部分 ---
# 输入和输出路径
"""
input_dir = "/kaggle/working/recognaization8commodate/yolo_dataset"
output_dir = "/kaggle/working/convnext_data"
"""
input_dir = r"C:\Users\aertlia\Desktop\zako\AI_study\数据挖掘\recognaization8commodate\image5"
output_dir = "C:/Users/aertlia/Desktop/zako/AI_study/数据挖掘/recognaization8commodate/enhance_data"
# 图像处理参数
TARGET_SIZE = (224, 224)  # ConvNeXt 默认输入尺寸

# --- 修复(Inpainting)参数 ---
BLACK_THRESHOLD = 5       # 判定为"纯黑"的阈值（0-255）
INPAINT_RADIUS = 3        # 修复半径

# --- CLAHE (对比度增强)参数 ---
CLAHE_CLIP_LIMIT = 2.0    # 对比度限制因子
CLAHE_GRID_SIZE = (8, 8)   # 局部网格大小

# --- 锐化参数 ---
UNSHARP_SIGMA = 1.0       # 高斯模糊的Sigma
UNSHARP_WEIGHT = 1.5      # 锐化细节的权重

def get_class_name(cls_id):
    """
    根据类别 ID 获取类别名称。
    应根据数据集的 data.yaml 进行同步更新。
    """
    class_map = {
        '0': 'hks_large', '1': 'hks_small', '2': 'hn_can', '3': 'jlb_can',
        '4': 'kkkl_can', '5': 'wlj_can', '6': 'xb_wt', '7': 'xb'
    }
    return class_map.get(str(cls_id), 'unknown')

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
    # 将图像转换为灰度，用于创建黑色方块的掩膜
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 阈值化：找到极暗（几乎纯黑）的区域
    # 这里我们创建了一个二值掩膜：黑色方块区域为白色(255)，其他区域为黑色(0)
    _, black_mask = cv2.threshold(gray, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    
    # 形态学操作：去除掩膜中的微小噪点，确保方块区域的完整性
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel_close)

    # 找到所有黑色区域的轮廓，并过滤掉太小的区域（防止误伤正常的商品黑色纹理）
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(black_mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:  # 过滤掉面积小于100像素的噪声区域
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)
            
    # 执行修复操作 (Telea算法)，利用周围的商品信息填补黑色方块
    if np.sum(final_mask) > 0: # 只有当找到黑色方块时才进行修复，节省资源
        inpainted_img = cv2.inpaint(img, final_mask, INPAINT_RADIUS, cv2.INPAINT_TELEA)
    else:
        inpainted_img = img

    # --- 2. 图像清晰度增强 ---
    # 2.1 转换到 LAB 空间，只增强 L (亮度) 通道，避免颜色失真
    lab = cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 2.2 应用 CLAHE (限制对比度自适应直方图均衡化)
    # 这对严重的模糊和曝光不均非常有效，能显著提升局部细节
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE)
    l_enhanced = clahe.apply(l)

    # 2.3 局部自适应锐化 (基于增强后的L通道)
    # 使用高斯模糊创建基础图像
    gaussian = cv2.GaussianBlur(l_enhanced, (0, 0), UNSHARP_SIGMA)
    # 局部对比度 = 原始亮度 - 高斯模糊基础，从而提取细节
    # 新的亮度 = 原始亮度 + 权重 * 局部对比度
    l_sharpened = cv2.addWeighted(l_enhanced, 1 + UNSHARP_WEIGHT, gaussian, -UNSHARP_WEIGHT, 0)

    # 2.4 合并回 LAB 空间并转换回 BGR
    enhanced_lab = cv2.merge((l_sharpened, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # --- 3. 尺寸调整 (Resize) ---
    resized_img = cv2.resize(enhanced_bgr, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)

    return resized_img

def main():
    print(f"数据增强预处理开始...")
    print(f"源目录: {input_dir}")
    print(f"目标目录: {output_dir}")

    for mode in ['train', 'val']:
        src_mode_dir = os.path.join(input_dir, mode)
        # 注意：这里我们只处理 images 子目录
        src_img_dir = os.path.join(src_mode_dir, 'images') 
        
        if not os.path.exists(src_img_dir):
            print(f"❌ 找不到图片目录: {src_img_dir}，跳过该模式。")
            continue

        files = os.listdir(src_img_dir)
        print(f"\n正在处理 {mode} 集, 共找到 {len(files)} 张图片...")

        # 使用 tqdm 显示进度条
        for filename in tqdm(files):
            # 支持多种常用图片格式
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                continue

            # 完整的文件路径
            src_file_path = os.path.join(src_img_dir, filename)
            
            # --- 读取图片 ---
            img = cv2.imread(src_file_path)
            
            # --- 调用处理核心函数 ---
            enhanced_img = process_and_enhance_image(img)

            if enhanced_img is not None:
                # --- 保存图片 ---
                # 注意：为了 ConvNeXt 训练方便，我们将处理后的数据统一保存
                dst_mode_dir = os.path.join(output_dir, mode)
                os.makedirs(dst_mode_dir, exist_ok=True)
                dst_file_path = os.path.join(dst_mode_dir, filename)
                
                cv2.imwrite(dst_file_path, enhanced_img)

    print("\n✅ 数据增强和修复完成！")
    print(f"处理后的文件保存在: {output_dir}")
    print(f"现在你可以直接使用这个目录下的数据开始训练 ConvNeXt 模型了。")

if __name__ == "__main__":
    main()
