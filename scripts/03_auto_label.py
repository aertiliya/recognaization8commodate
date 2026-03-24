"""
YOLO11 模型辅助标注脚本
使用训练好的模型对未标注图像进行预测，生成X-AnyLabeling格式的JSON标注文件
支持置信度阈值调整，方便人工审核和修正
"""

import json
import os
from pathlib import Path
import cv2
import numpy as np

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

def load_model(model_path):
    """加载YOLO11模型"""
    from ultralytics import YOLO
    model = YOLO(model_path)
    print(f"模型加载成功: {model_path}")
    return model

def predict_and_save_json(
    model,
    image_path,
    output_dir,
    conf_threshold=0.5,
    iou_threshold=0.5,
    save_images=False
):
    """
    对单张图像进行预测并保存为X-AnyLabeling JSON格式
    
    参数:
        model: YOLO模型
        image_path: 图像路径
        output_dir: 输出目录
        conf_threshold: 置信度阈值
        iou_threshold: IOU阈值
        save_images: 是否保存带标注的图像
    """
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    height, width = image.shape[:2]
    
    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    
    shapes = []
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            box = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            
            x1, y1, x2, y2 = box
            label = CLASSES[cls_id]
            
            shape = {
                "label": label,
                "score": round(conf, 4),
                "points": [
                    [float(x1), float(y1)],
                    [float(x2), float(y2)]
                ],
                "group_id": None,
                "description": f"confidence: {conf:.3f}",
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
                "kie_linking": []
            }
            shapes.append(shape)
    
    json_data = {
        "version": "4.0.0-beta.2",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "description": f"Auto-labeled with YOLO11, conf_threshold={conf_threshold}"
    }
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    json_file = output_path / f"{image_path.stem}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    if save_images and shapes:
        for shape in shapes:
            points = shape['points']
            x1, y1 = int(points[0][0]), int(points[0][1])
            x2, y2 = int(points[1][0]), int(points[1][1])
            label = shape['label']
            conf = shape['score']
            
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            text = f"{label} {conf:.2f}"
            cv2.putText(image, text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        vis_path = output_path / f"{image_path.stem}_vis.jpg"
        cv2.imwrite(str(vis_path), image)
    
    return len(shapes)

def process_unlabeled_images(
    model_path,
    source_dir,
    output_dir,
    conf_threshold=0.5,
    iou_threshold=0.5,
    save_images=False,
    extensions=['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
):
    """
    处理目录中所有未标注的图像
    
    参数:
        model_path: 模型路径
        source_dir: 源图像目录
        output_dir: 输出目录
        conf_threshold: 置信度阈值
        iou_threshold: IOU阈值
        save_images: 是否保存可视化结果
        extensions: 图像扩展名列表
    """
    model = load_model(model_path)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    all_images = []
    for ext in extensions:
        all_images.extend(source_path.glob(f"*{ext}"))
    
    existing_jsons = set(p.stem for p in source_path.glob("*.json"))
    
    unlabeled_images = [
        img for img in all_images 
        if img.stem not in existing_jsons
    ]
    
    print(f"\n找到 {len(all_images)} 张图像")
    print(f"已标注: {len(existing_jsons)} 张")
    print(f"待标注: {len(unlabeled_images)} 张")
    print(f"\n置信度阈值: {conf_threshold}")
    print(f"IOU阈值: {iou_threshold}")
    print("-" * 50)
    
    total_detections = 0
    processed = 0
    
    for img_path in unlabeled_images:
        num_detections = predict_and_save_json(
            model, img_path, output_path, 
            conf_threshold, iou_threshold, save_images
        )
        if num_detections is not None:
            total_detections += num_detections
            processed += 1
            
            if processed % 50 == 0:
                print(f"已处理: {processed}/{len(unlabeled_images)}")
    
    print("\n" + "=" * 50)
    print("辅助标注完成!")
    print(f"处理图像: {processed} 张")
    print(f"总检测框: {total_detections} 个")
    print(f"平均每张: {total_detections/processed:.2f} 个检测框")
    print(f"\n标注文件保存在: {output_path}")
    print("=" * 50)

def batch_predict_with_different_thresholds(
    model_path,
    image_dir,
    output_base_dir,
    thresholds=[0.3, 0.5, 0.7]
):
    """
    使用不同置信度阈值批量预测，便于选择最佳阈值
    """
    for thresh in thresholds:
        output_dir = Path(output_base_dir) / f"conf_{thresh}"
        print(f"\n使用置信度阈值: {thresh}")
        process_unlabeled_images(
            model_path, image_dir, output_dir,
            conf_threshold=thresh
        )

def review_predictions(json_dir, min_conf=0.8):
    """
    统计预测结果，帮助确定置信度阈值
    """
    json_path = Path(json_dir)
    json_files = list(json_path.glob("*.json"))
    
    stats = {cls: [] for cls in CLASSES}
    empty_count = 0
    
    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data.get('shapes'):
            empty_count += 1
            continue
        
        for shape in data['shapes']:
            label = shape['label']
            conf = shape.get('score', 0)
            stats[label].append(conf)
    
    print("\n预测统计:")
    print("-" * 50)
    for label, confs in stats.items():
        if confs:
            avg_conf = sum(confs) / len(confs)
            print(f"{label}: {len(confs)} 个检测框, 平均置信度: {avg_conf:.3f}")
        else:
            print(f"{label}: 0 个检测框")
    
    print(f"\n空检测图像: {empty_count} 张")
    print(f"总图像数: {len(json_files)} 张")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO11 模型辅助标注工具')
    parser.add_argument('--mode', type=str, default='label', 
                        choices=['label', 'review', 'batch'],
                        help='运行模式: label(标注), review(统计), batch(批量测试)')
    parser.add_argument('--model', type=str, 
                        default='../runs/detect/product_detector/weights/best.pt',
                        help='模型路径')
    parser.add_argument('--source', type=str, default='../image5/train',
                        help='源图像目录')
    parser.add_argument('--output', type=str, default='../auto_labels',
                        help='输出目录')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.5,
                        help='IOU阈值')
    parser.add_argument('--save_vis', action='store_true',
                        help='保存可视化结果')
    
    args = parser.parse_args()
    
    if args.mode == 'label':
        process_unlabeled_images(
            model_path=args.model,
            source_dir=args.source,
            output_dir=args.output,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            save_images=args.save_vis
        )
    elif args.mode == 'review':
        review_predictions(args.output)
    elif args.mode == 'batch':
        batch_predict_with_different_thresholds(
            model_path=args.model,
            image_dir=args.source,
            output_base_dir=args.output
        )
