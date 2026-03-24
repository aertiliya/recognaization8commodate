"""
YOLO11 商品检测模型训练脚本
用于训练8类商品的检测模型
"""

import os
import subprocess
import sys
from pathlib import Path

def check_ultralytics():
    """检查并安装ultralytics库"""
    try:
        import ultralytics
        print(f"ultralytics 版本: {ultralytics.__version__}")
    except ImportError:
        print("正在安装 ultralytics...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        import ultralytics
        print(f"ultralytics 安装完成，版本: {ultralytics.__version__}")

def download_weights(model_size='n'):
    """
    下载YOLO11预训练权重
    model_size: n(nano), s(small), m(medium), l(large), x(xlarge)
    """
    from ultralytics import YOLO
    import os
    
    # 检查本地是否已有yolo11权重文件
    local_weights_dir = os.path.dirname(__file__)
    possible_names = [
        f"yolo11{model_size}.pt",
        f"yolo11{model_size}_best.pt", 
        f"yolo11{model_size}_last.pt",
        f"yolo11s.pt" if model_size == 's' else None
    ]
    possible_names = [name for name in possible_names if name]  # 过滤None
    
    for weight_name in possible_names:
        weight_path = os.path.join(local_weights_dir, weight_name)
        if os.path.exists(weight_path):
            print(f"使用本地权重: {weight_path}")
            model = YOLO(weight_path)
            print(f"模型加载完成: {weight_name}")
            return model
    
    # 如果本地没有，再下载yolo11权重
    model_name = f"yolo11{model_size}.pt"
    print(f"本地未找到权重，正在下载 {model_name}...")
    model = YOLO(model_name)
    print(f"模型下载并加载完成: {model_name}")
    return model

def train_model(
    model_size='n',
    epochs=100,
    batch_size=16,
    img_size=640,
    data_yaml=None,
    project='runs/detect',
    name='product_detector',
    resume=False
):
    """
    训练YOLO11检测模型
    
    参数:
        model_size: 模型大小 (n/s/m/l/x)
        epochs: 训练轮数
        batch_size: 批次大小
        img_size: 图像大小
        data_yaml: 数据集配置文件路径
        project: 项目保存路径
        name: 实验名称
        resume: 是否继续训练
    """
    from ultralytics import YOLO
    
    if data_yaml is None:
        data_yaml = os.path.join(os.path.dirname(__file__), 'yolo_train.yaml')
    
    # 确保使用正确的YOLO训练配置文件
    if data_yaml.endswith('data.yaml'):
        yolo_train_path = os.path.join(os.path.dirname(__file__), 'yolo_train.yaml')
        if os.path.exists(yolo_train_path):
            print(f"检测到data.yaml，自动切换到yolo_train.yaml: {yolo_train_path}")
            data_yaml = yolo_train_path
    
    print(f"使用配置文件: {data_yaml}")
    
    if resume:
        last_pt = Path(project) / name / 'weights' / 'last.pt'
        if last_pt.exists():
            print(f"继续训练: {last_pt}")
            model = YOLO(str(last_pt))
        else:
            print("未找到last.pt，开始新训练")
            model = download_weights(model_size)
    else:
        model = download_weights(model_size)
    
    print("\n" + "="*50)
    print("开始训练 YOLO11 商品检测模型")
    print("="*50)
    print(f"模型大小: YOLO11{model_size}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"图像大小: {img_size}")
    print(f"数据配置: {data_yaml}")
    print("="*50 + "\n")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=project,
        name=name,
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True,
        device='',  # 自动选择设备
        amp=False,  # 禁用AMP以避免下载yolo26权重
        workers=4,
        save=True,
        save_period=10,  # 每10轮保存一次
        patience=50,  # 早停耐心值
        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,  # box loss gain
        cls=0.5,  # cls loss gain
        dfl=1.5,  # dfl loss gain
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.0,
        nbs=64,
        hsv_h=0.015,  # 图像HSV-Hue增强
        hsv_s=0.7,    # 图像HSV-Saturation增强
        hsv_v=0.4,    # 图像HSV-Value增强
        degrees=0.0,  # 图像旋转增强
        translate=0.1,  # 图像平移增强
        scale=0.5,    # 图像缩放增强
        shear=0.0,    # 图像剪切增强
        perspective=0.0,  # 图像透视增强
        flipud=0.0,   # 图像上下翻转概率
        fliplr=0.5,   # 图像左右翻转概率
        mosaic=1.0,   # 图像 mosaic 增强
        mixup=0.0,    # 图像 mixup 增强
        copy_paste=0.0,  # 分割 copy-paste 增强
    )
    
    print("\n训练完成!")
    print(f"最佳模型保存在: {Path(project) / name / 'weights' / 'best.pt'}")
    
    return model, results

def validate_model(model_path, data_yaml='data.yaml'):
    """验证模型性能"""
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    results = model.val(data=data_yaml)
    
    print("\n验证结果:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    
    return results

def export_model(model_path, format='onnx'):
    """导出模型到其他格式"""
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    model.export(format=format)
    print(f"模型已导出为 {format} 格式")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO11 商品检测模型训练')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'export'],
                        help='运行模式: train(训练), val(验证), export(导出)')
    parser.add_argument('--model_size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='模型大小: n(nano), s(small), m(medium), l(large), x(xlarge)')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--img_size', type=int, default=640, help='图像大小')
    parser.add_argument('--data', type=str, default='yolo_train.yaml', help='数据集配置文件')
    parser.add_argument('--weights', type=str, default=None, help='预训练权重路径')
    parser.add_argument('--resume', action='store_true', help='继续训练')
    parser.add_argument('--export_format', type=str, default='onnx', help='导出格式')
    
    args = parser.parse_args()
    
    check_ultralytics()
    
    if args.mode == 'train':
        # 如果没有指定data_yaml，使用相对于脚本的默认值
        if args.data in ['data.yaml', 'yolo_train.yaml']:
            import os
            args.data = os.path.join(os.path.dirname(__file__), args.data)
        
        # 确保使用正确的YOLO训练配置文件
        if args.data.endswith('data.yaml'):
            yolo_train_path = os.path.join(os.path.dirname(__file__), 'yolo_train.yaml')
            if os.path.exists(yolo_train_path):
                print(f"检测到data.yaml，自动切换到yolo_train.yaml: {yolo_train_path}")
                args.data = yolo_train_path
        train_model(
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            data_yaml=args.data,
            resume=args.resume
        )
    elif args.mode == 'val':
        # 如果没有指定data_yaml，使用相对于脚本的默认值
        if args.data in ['data.yaml', 'yolo_train.yaml']:
            import os
            args.data = os.path.join(os.path.dirname(__file__), args.data)
        
        # 确保使用正确的YOLO训练配置文件
        if args.data.endswith('data.yaml'):
            yolo_train_path = os.path.join(os.path.dirname(__file__), 'yolo_train.yaml')
            if os.path.exists(yolo_train_path):
                print(f"检测到data.yaml，自动切换到yolo_train.yaml: {yolo_train_path}")
                args.data = yolo_train_path
        model_path = args.weights or 'runs/detect/product_detector/weights/best.pt'
        validate_model(model_path, args.data)
    elif args.mode == 'export':
        model_path = args.weights or 'runs/detect/product_detector/weights/best.pt'
        export_model(model_path, args.export_format)
