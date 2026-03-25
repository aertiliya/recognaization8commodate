"""
第三步：训练 ConvNeXt V2（使用 GT-ROI 裁剪后的商品图像）
对裁剪出的商品局部图进行细分类
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import timm
from tqdm import tqdm
import json
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# 8个商品类别
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


class GTROIDataset(Dataset):
    """
    GT-ROI 数据集加载器
    从按类别组织的文件夹中加载裁剪后的ROI图像
    """
    def __init__(self, roi_dir, split='train', transform=None):
        self.roi_dir = Path(roi_dir)
        self.split = split
        self.transform = transform
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}
        
        # 每个类别一个文件夹
        for class_name in CLASSES:
            class_dir = self.roi_dir / class_name
            if not class_dir.exists():
                print(f"⚠️ 警告: 类别文件夹不存在 {class_dir}")
                continue
            
            # 获取该类别下的所有图像
            class_images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
            
            for img_path in class_images:
                self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        print(f"加载 {split} 数据集: {len(self.samples)} 张图像")
        for cls in CLASSES:
            count = sum(1 for _, label in self.samples if label == self.class_to_idx[cls])
            print(f"  {cls}: {count} 张")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(img_size=224):
    """数据增强和预处理"""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_model(num_classes=8, model_name='convnextv2_atto', pretrained=True):
    """创建 ConvNeXt V2 模型"""
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    print(f"创建模型: {model_name}")
    print(f"类别数量: {num_classes}")
    return model


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': running_loss / len(dataloader),
            'acc': 100. * correct / total
        })
    
    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc, all_preds, all_labels


def train_model(
    roi_dir='gt_roi_dataset',
    output_dir='convnext_models',
    model_name='convnextv2_atto',
    img_size=224,
    batch_size=32,
    epochs=50,
    lr=0.001,
    device='cuda'
):
    """
    主训练函数
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 转换为绝对路径
    roi_dir = os.path.abspath(roi_dir)
    output_dir = os.path.abspath(output_dir)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取数据变换
    train_transform, val_transform = get_transforms(img_size)
    
    # 创建数据集
    # 注意：由于GT-ROI数据集来自train目录，我们将其全部用于训练
    # 可以手动划分验证集，或者如果有val目录则分别加载
    print("\n" + "="*60)
    print("加载 GT-ROI 数据集")
    print("="*60)
    
    train_dataset = GTROIDataset(roi_dir, 'train', train_transform)
    
    # 如果没有找到图像，退出
    if len(train_dataset) == 0:
        print("❌ 错误: 没有找到训练图像，请先运行 crop_gt_roi.py")
        return None, None
    
    # 手动划分训练集和验证集 (80% 训练, 20% 验证)
    dataset_size = len(train_dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # 为验证集使用val_transform
    val_dataset = GTROIDataset(roi_dir, 'val', val_transform)
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    # 创建模型
    model = create_model(num_classes=len(CLASSES), model_name=model_name)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 训练记录
    best_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print("\n" + "="*60)
    print("开始训练 ConvNeXt V2 分类模型 (使用 GT-ROI)")
    print("="*60)
    print(f"模型: {model_name}")
    print(f"输入尺寸: {img_size}x{img_size}")
    print(f"批次大小: {batch_size}")
    print(f"训练轮数: {epochs}")
    print(f"学习率: {lr}")
    print(f"训练样本: {train_size}, 验证样本: {val_size}")
    print("="*60 + "\n")
    
    # 训练循环
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'class_to_idx': train_dataset.class_to_idx,
                'classes': CLASSES
            }, output_path / 'best_model.pth')
            print(f"💾 保存最佳模型，验证准确率: {best_acc:.2f}%")
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': train_dataset.class_to_idx,
        'classes': CLASSES
    }, output_path / 'final_model.pth')
    
    # 保存训练历史
    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # 最终报告
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print(f"最佳验证准确率: {best_acc:.2f}%")
    print(f"模型保存在: {output_path}")
    
    print("\n验证集分类报告:")
    print(classification_report(val_labels, val_preds, target_names=CLASSES))
    
    cm = confusion_matrix(val_labels, val_preds)
    print("\n混淆矩阵:")
    print(cm)
    
    return model, history


def predict_image(model, image_path, transform, device, class_names):
    """单张图像预测"""
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = probs.max(1)
    
    return class_names[pred.item()], conf.item()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ConvNeXt V2 商品分类模型训练 (GT-ROI)')
    parser.add_argument('--roi_dir', type=str, default='gt_roi_dataset',
                        help='GT-ROI 数据集目录 (包含8个类别子文件夹)')
    parser.add_argument('--output_dir', type=str, default='convnext_models',
                        help='模型输出目录')
    parser.add_argument('--model_name', type=str, default='convnextv2_atto',
                        choices=['convnextv2_atto', 'convnextv2_femto', 
                                'convnextv2_pico', 'convnextv2_nano',
                                'convnextv2_tiny', 'convnextv2_small',
                                'convnextv2_base', 'convnextv2_large'],
                        help='模型大小')
    parser.add_argument('--img_size', type=int, default=224,
                        help='输入图像大小')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    train_model(
        roi_dir=args.roi_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device
    )
