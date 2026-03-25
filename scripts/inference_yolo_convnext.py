"""
端到端推理脚本：YOLO11 检测 + ConvNeXt V2 细分类
两阶段架构：YOLO 负责定位，ConvNeXt 负责细分类
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import json
import argparse
from torchvision import transforms
import timm

# 8个商品类别
CLASSES = [
    "hks_large",   # 大红瓶王老吉
    "hks_small",   # 小红瓶王老吉
    "hn_can",      # 红牛罐
    "jlb_can",     # 劲凉冰红茶
    "kkkl_can",    # 可口可乐
    "wlj_can",     # 王老吉罐装
    "xb_wt",       # 雪碧无糖
    "xb"           # 雪碧
]

# 类别颜色映射（用于可视化）
COLORS = {
    "hks_large": (255, 0, 0),     # 红
    "hks_small": (0, 255, 0),     # 绿
    "hn_can": (0, 0, 255),        # 蓝
    "jlb_can": (255, 255, 0),     # 黄
    "kkkl_can": (255, 0, 255),    # 紫
    "wlj_can": (0, 255, 255),     # 青
    "xb_wt": (128, 0, 128),       # 深紫
    "xb": (255, 165, 0),          # 橙
}


class YOLOConvNeXtPipeline:
    """YOLO + ConvNeXt 两阶段流水线"""
    
    def __init__(self, yolo_model_path, convnext_model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载 YOLO 模型
        print(f"加载 YOLO 模型: {yolo_model_path}")
        self.yolo_model = self._load_yolo(yolo_model_path)
        
        # 加载 ConvNeXt 模型
        print(f"加载 ConvNeXt 模型: {convnext_model_path}")
        self.convnext_model = self._load_convnext(convnext_model_path)
        
        # ConvNeXt 预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("模型加载完成！")
    
    def _load_yolo(self, model_path):
        """加载 YOLO 模型"""
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            return model
        except ImportError:
            print("警告: ultralytics 未安装，使用备用加载方式")
            # 如果是 ONNX 模型
            if model_path.endswith('.onnx'):
                import onnxruntime as ort
                return ort.InferenceSession(model_path)
            return None
    
    def _load_convnext(self, model_path):
        """加载 ConvNeXt 模型"""
        # 创建模型架构
        model = timm.create_model('convnextv2_atto', pretrained=False, num_classes=8)
        
        # 加载训练好的权重
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"已加载模型权重")
        else:
            print(f"警告: 模型文件不存在 {model_path}，使用随机权重")
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def detect(self, image_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        第一阶段：YOLO 检测
        返回检测框列表 [(x1, y1, x2, y2, conf, class_id), ...]
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        results = []
        
        if self.yolo_model is None:
            print("YOLO 模型未加载，跳过检测阶段")
            return image, []
        
        # 使用 YOLO 进行推理
        yolo_results = self.yolo_model(image, conf=conf_threshold, iou=iou_threshold)
        
        for result in yolo_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    results.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'conf': float(conf),
                        'yolo_class': CLASSES[cls_id] if cls_id < len(CLASSES) else f"class_{cls_id}",
                        'yolo_class_id': cls_id
                    })
        
        return image, results
    
    def classify_roi(self, image, bbox):
        """
        第二阶段：ConvNeXt 细分类
        对裁剪的 ROI 进行分类
        """
        x1, y1, x2, y2 = bbox
        
        # 边界检查
        h, w = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # 裁剪 ROI
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None, 0.0
        
        # 转换为 PIL Image
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(roi_rgb)
        
        # 预处理
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            outputs = self.convnext_model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = probs.max(1)
        
        class_id = pred.item()
        confidence = conf.item()
        class_name = CLASSES[class_id]
        
        return class_name, confidence
    
    def inference(self, image_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        完整推理流程：检测 + 分类
        """
        print(f"\n处理图像: {image_path}")
        
        # 阶段 1：YOLO 检测
        image, detections = self.detect(image_path, conf_threshold, iou_threshold)
        print(f"YOLO 检测到 {len(detections)} 个目标")
        
        # 阶段 2：ConvNeXt 细分类
        results = []
        for det in detections:
            bbox = det['bbox']
            
            # ConvNeXt 细分类
            convnext_class, convnext_conf = self.classify_roi(image, bbox)
            
            result = {
                'bbox': bbox,
                'yolo_class': det['yolo_class'],
                'yolo_conf': det['conf'],
                'convnext_class': convnext_class,
                'convnext_conf': convnext_conf,
                'final_class': convnext_class  # 最终类别由 ConvNeXt 决定
            }
            results.append(result)
            
            print(f"  检测框 {bbox}:")
            print(f"    YOLO 预测: {det['yolo_class']} ({det['conf']:.3f})")
            print(f"    ConvNeXt 细分类: {convnext_class} ({convnext_conf:.3f})")
        
        return image, results
    
    def visualize(self, image, results, save_path=None):
        """可视化检测结果"""
        vis_image = image.copy()
        
        for i, result in enumerate(results):
            x1, y1, x2, y2 = result['bbox']
            final_class = result['final_class']
            conf = result['convnext_conf']
            
            # 获取颜色
            color = COLORS.get(final_class, (255, 255, 255))
            
            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{final_class}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 10
            
            cv2.rectangle(vis_image, (x1, label_y - label_size[1] - 5), 
                         (x1 + label_size[0], label_y + 5), color, -1)
            cv2.putText(vis_image, label, (x1, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 如果 YOLO 和 ConvNeXt 类别不同，标注差异
            if result['yolo_class'] != result['convnext_class']:
                diff_label = f"YOLO: {result['yolo_class']}"
                cv2.putText(vis_image, diff_label, (x1, y2 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 添加统计信息
        stats_text = f"Total: {len(results)} objects"
        cv2.putText(vis_image, stats_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
            print(f"可视化结果已保存: {save_path}")
        
        return vis_image
    
    def process_directory(self, input_dir, output_dir, conf_threshold=0.25):
        """批量处理目录"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 查找所有图像
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(input_path.glob(ext))
        
        print(f"\n找到 {len(image_files)} 张图像")
        
        all_results = {}
        
        for img_file in image_files:
            # 推理
            image, results = self.inference(img_file, conf_threshold)
            
            # 保存可视化结果
            vis_path = output_path / f"vis_{img_file.name}"
            self.visualize(image, results, str(vis_path))
            
            # 保存 JSON 结果
            all_results[img_file.name] = results
        
        # 保存汇总结果
        json_path = output_path / 'inference_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n处理完成！结果保存在: {output_path}")
        return all_results


def main():
    parser = argparse.ArgumentParser(description='YOLO11 + ConvNeXt V2 端到端推理')
    parser.add_argument('--input', type=str, required=True, 
                       help='输入图像路径或目录')
    parser.add_argument('--yolo_model', type=str, default='scripts/best.pt',
                       help='YOLO 模型路径')
    parser.add_argument('--convnext_model', type=str, default='convnext_models/best_model.pth',
                       help='ConvNeXt 模型路径')
    parser.add_argument('--output', type=str, default='inference_output',
                       help='输出目录')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 创建流水线
    pipeline = YOLOConvNeXtPipeline(
        yolo_model_path=args.yolo_model,
        convnext_model_path=args.convnext_model,
        device=args.device
    )
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 单张图像
        image, results = pipeline.inference(str(input_path), conf_threshold=args.conf)
        vis_path = Path(args.output) / f"vis_{input_path.name}"
        Path(args.output).mkdir(parents=True, exist_ok=True)
        pipeline.visualize(image, results, str(vis_path))
    else:
        # 目录
        pipeline.process_directory(args.input, args.output, conf_threshold=args.conf)


if __name__ == "__main__":
    main()
