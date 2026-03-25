"""
端到端性能评估脚本
用于计算上机报告要求的：“检测正确且分类正确”的联合统计
"""

import os
import cv2
from pathlib import Path
from inference_yolo_convnext import YOLOConvNeXtPipeline, CLASSES

def calculate_iou(box1, box2):
    """计算两个框的交并比 (IoU)"""
    x1_int = max(box1[0], box2[0])
    y1_int = max(box1[1], box2[1])
    x2_int = min(box1[2], box2[2])
    y2_int = min(box1[3], box2[3])

    if x1_int >= x2_int or y1_int >= y2_int:
        return 0.0

    intersection = (x2_int - x1_int) * (y2_int - y1_int)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def parse_yolo_label(label_path, img_width, img_height):
    """解析 YOLO 的 txt 真值标签，还原为像素坐标"""
    gt_boxes = []
    if not os.path.exists(label_path):
        return gt_boxes
        
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:5])
                
                # 反归一化
                x1 = int((x_center - w / 2) * img_width)
                y1 = int((y_center - h / 2) * img_height)
                x2 = int((x_center + w / 2) * img_width)
                y2 = int((y_center + h / 2) * img_height)
                
                gt_boxes.append({
                    'class_name': CLASSES[class_id],
                    'bbox': [x1, y1, x2, y2]
                })
    return gt_boxes

def evaluate_test_set(images_dir, labels_dir, yolo_model, convnext_model, iou_threshold=0.5):
    """在测试集上进行端到端评估"""
    pipeline = YOLOConvNeXtPipeline(yolo_model, convnext_model)
    
    # 统计指标
    total_gt = 0          # 真实目标总数
    total_pred = 0        # 预测框总数
    
    stats = {
        'both_correct': 0,     # 检测对 + 分类也对 (上机报告核心指标)
        'yolo_missed': 0,      # YOLO 漏检了
        'yolo_wrong_bbox': 0,  # YOLO 框偏了 (IoU < 0.5)
        'convnext_wrong': 0    # YOLO 框对了，但 ConvNeXt 分类错了
    }

    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
    print(f"\n开始评估测试集，共找到 {len(image_files)} 张图片...")

    for img_path in image_files:
        label_path = labels_path / f"{img_path.stem}.txt"
        
        # 读取图像尺寸以解析标签
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        
        # 获取真实标签 (GT)
        gt_objects = parse_yolo_label(label_path, w, h)
        total_gt += len(gt_objects)
        
        # 获取两阶段流水线的预测结果
        # 注意：这里我们只要结果列表，不用保存可视化图以加快速度
        _, pred_objects = pipeline.inference(str(img_path), conf_threshold=0.25)
        total_pred += len(pred_objects)
        
        # 对比评估逻辑
        for gt in gt_objects:
            best_iou = 0
            best_pred = None
            
            # 找 IoU 最高的预测框
            for pred in pred_objects:
                iou = calculate_iou(gt['bbox'], pred['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred
            
            # 分类讨论错误类型（完美契合作业的误差传递分析）
            if best_pred is None:
                stats['yolo_missed'] += 1
            elif best_iou < iou_threshold:
                stats['yolo_wrong_bbox'] += 1
            else:
                # 框准了，看分类
                if best_pred['final_class'] == gt['class_name']:
                    stats['both_correct'] += 1
                else:
                    stats['convnext_wrong'] += 1

    # 打印给实验报告用的结果
    print("\n" + "="*50)
    print("上机报告：端到端评估结果 (Test 集)")
    print("="*50)
    print(f"真实目标总数 (GT): {total_gt}")
    print(f"模型预测总数 (Pred): {total_pred}")
    print("-" * 50)
    
    if total_gt > 0:
        joint_acc = (stats['both_correct'] / total_gt) * 100
        print(f"✅ 端到端联合正确率 (检测准 且 分类准): {joint_acc:.2f}%")
        print("\n🔻 误差传递分析 (失败原因分解):")
        print(f"  1. YOLO 漏检 (完全没发现): {stats['yolo_missed']} 个 ({(stats['yolo_missed']/total_gt)*100:.1f}%)")
        print(f"  2. YOLO 框偏移 (IoU < {iou_threshold}): {stats['yolo_wrong_bbox']} 个 ({(stats['yolo_wrong_bbox']/total_gt)*100:.1f}%)")
        print(f"  3. ConvNeXt 分类错误 (框对了但认错包装): {stats['convnext_wrong']} 个 ({(stats['convnext_wrong']/total_gt)*100:.1f}%)")
    else:
        print("未找到真实的测试集标签，无法评估。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='端到端联合性能评估')
    # 根据你的目录结构调整默认路径
    parser.add_argument('--test_images', type=str, default='yolo_dataset/test/images', help='Test集图片目录')
    parser.add_argument('--test_labels', type=str, default='yolo_dataset/test/labels', help='Test集标签目录')
    parser.add_argument('--yolo_model', type=str, default='runs/detect/train/weights/best.pt', help='训练好的YOLO模型')
    parser.add_argument('--convnext_model', type=str, default='convnext_models/best_model.pth', help='训练好的ConvNeXt模型')
    
    args = parser.parse_args()
    
    evaluate_test_set(
        images_dir=args.test_images,
        labels_dir=args.test_labels,
        yolo_model=args.yolo_model,
        convnext_model=args.convnext_model
    )