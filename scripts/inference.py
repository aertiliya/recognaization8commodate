"""
商品识别推理脚本
YOLO11 先检测 → ConvNeXt V2 再分类
"""

import os
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import timm
from torchvision import transforms

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


class ProductRecognizer:
    def __init__(self, yolo_model_path, convnext_model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        print("加载 YOLO11 检测模型...")
        self.yolo_model = YOLO(yolo_model_path)
        
        print("加载 ConvNeXt V2 分类模型...")
        checkpoint = torch.load(convnext_model_path, map_location=self.device)
        self.class_names = checkpoint.get('classes', CLASSES)
        
        self.convnext_model = timm.create_model(
            'convnextv2_atto',
            pretrained=False,
            num_classes=len(self.class_names)
        )
        self.convnext_model.load_state_dict(checkpoint['model_state_dict'])
        self.convnext_model = self.convnext_model.to(self.device)
        self.convnext_model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("模型加载完成!")
    
    def detect_and_classify(self, image_path, conf_threshold=0.25, iou_threshold=0.45):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None
        
        results = self.yolo_model.predict(
            source=str(image_path),
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                
                cropped = image[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue
                
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                cropped_pil = Image.fromarray(cropped_rgb)
                
                input_tensor = self.transform(cropped_pil).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.convnext_model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, pred = probs.max(1)
                
                class_name = self.class_names[pred.item()]
                class_conf = conf.item()
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'class': class_name,
                    'confidence': float(class_conf),
                    'detection_conf': float(boxes.conf[i].cpu().numpy())
                }
                detections.append(detection)
        
        return detections
    
    def process_image(self, image_path, output_dir=None, save_json=True, save_vis=True):
        detections = self.detect_and_classify(image_path)
        
        if detections is None:
            return None
        
        image_path = Path(image_path)
        
        if save_json and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            image = cv2.imread(str(image_path))
            height, width = image.shape[:2]
            
            shapes = []
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                shape = {
                    "label": det['class'],
                    "score": round(det['confidence'], 4),
                    "points": [[x1, y1], [x2, y2]],
                    "group_id": None,
                    "description": f"conf: {det['confidence']:.3f}",
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
                "description": f"Detected by YOLO11 + Classified by ConvNeXt V2"
            }
            
            json_file = output_path / f"{image_path.stem}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        if save_vis and output_dir:
            output_path = Path(output_dir)
            image = cv2.imread(str(image_path))
            
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                class_name = det['class']
                conf = det['confidence']
                
                color = (0, 255, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                text = f"{class_name} {conf:.2f}"
                cv2.putText(image, text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            vis_path = output_path / f"{image_path.stem}_result.jpg"
            cv2.imwrite(str(vis_path), image)
        
        return detections
    
    def process_directory(
        self, 
        image_dir, 
        output_dir,
        extensions=['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    ):
        image_path = Path(image_dir)
        all_images = []
        for ext in extensions:
            all_images.extend(image_path.glob(f"*{ext}"))
        
        print(f"找到 {len(all_images)} 张图像")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_results = {}
        for i, img_path in enumerate(all_images, 1):
            print(f"处理 [{i}/{len(all_images)}]: {img_path.name}")
            
            detections = self.process_image(
                img_path, 
                output_dir=output_dir,
                save_json=True,
                save_vis=True
            )
            
            if detections:
                all_results[img_path.name] = {
                    'num_detections': len(detections),
                    'detections': detections
                }
        
        results_file = output_path / 'all_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n处理完成!")
        print(f"结果保存在: {output_path}")
        
        return all_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='商品识别推理: YOLO11检测 + ConvNeXt V2分类')
    parser.add_argument('--yolo_model', type=str, default='last.pt',
                        help='YOLO11模型路径')
    parser.add_argument('--convnext_model', type=str, default='../convnext_models/best_model.pth',
                        help='ConvNeXt V2模型路径')
    parser.add_argument('--source', type=str, required=True,
                        help='源图像路径(文件或目录)')
    parser.add_argument('--output', type=str, default='../results',
                        help='输出目录')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='检测置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IOU阈值')
    parser.add_argument('--device', type=str, default='cuda',
                        help='推理设备')
    
    args = parser.parse_args()
    
    recognizer = ProductRecognizer(
        yolo_model_path=args.yolo_model,
        convnext_model_path=args.convnext_model,
        device=args.device
    )
    
    source_path = Path(args.source)
    
    if source_path.is_file():
        print(f"处理单张图像: {source_path}")
        detections = recognizer.process_image(
            source_path,
            output_dir=args.output,
            save_json=True,
            save_vis=True
        )
        
        if detections:
            print(f"\n检测结果:")
            for det in detections:
                print(f"  {det['class']}: {det['confidence']:.3f}")
    
    elif source_path.is_dir():
        print(f"处理目录: {source_path}")
        recognizer.process_directory(source_path, args.output)
    
    else:
        print(f"错误: 路径不存在 {source_path}")


if __name__ == "__main__":
    main()
