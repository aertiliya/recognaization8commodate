"""
ConvNeXt V2 商品分类模型训练脚本
使用预训练的ConvNeXt V2模型对检测到的商品进行分类
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

class ProductDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, train_ratio=0.8):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}
        
        # 直接从 image5/train 目录加载数据
        images_dir = self.data_dir / 'image5' / 'train'
        if not images_dir.exists():
            raise ValueError(f"目录不存在: {images_dir}")
        
        # 收集所有图像文件
        all_images = []
        for class_name in CLASSES:
            class_images = list(images_dir.glob(f"{class_name}*.jpg"))
            for img_path in class_images:
                all_images.append((str(img_path), self.class_to_idx[class_name]))
        
        # 划分训练集和验证集
        import random
        random.seed(42)
        random.shuffle(all_images)
        
        split_idx = int(len(all_images) * train_ratio)
        if split == 'train':
            self.samples = all_images[:split_idx]
        else:
            self.samples = all_images[split_idx:]
        
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
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    print(f"创建模型: {model_name}")
    print(f"类别数量: {num_classes}")
    return model


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
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
    data_dir='..',
    output_dir='../convnext_models',
    model_name='convnextv2_atto',
    img_size=224,
    batch_size=32,
    epochs=50,
    lr=0.001,
    device='cuda'
):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_transform, val_transform = get_transforms(img_size)
    
    train_dataset = ProductDataset(data_dir, 'train', train_transform)
    val_dataset = ProductDataset(data_dir, 'val', val_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    model = create_model(num_classes=len(CLASSES), model_name=model_name)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print("\n" + "="*60)
    print("开始训练 ConvNeXt V2 分类模型")
    print("="*60)
    print(f"模型: {model_name}")
    print(f"图像大小: {img_size}")
    print(f"批次大小: {batch_size}")
    print(f"训练轮数: {epochs}")
    print(f"学习率: {lr}")
    print("="*60 + "\n")
    
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
            print(f"保存最佳模型，准确率: {best_acc:.2f}%")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': train_dataset.class_to_idx,
        'classes': CLASSES
    }, output_path / 'final_model.pth')
    
    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print(f"最佳验证准确率: {best_acc:.2f}%")
    print(f"模型保存在: {output_path}")
    
    print("\n分类报告:")
    print(classification_report(val_labels, val_preds, target_names=CLASSES))
    
    cm = confusion_matrix(val_labels, val_preds)
    print("\n混淆矩阵:")
    print(cm)
    
    return model, history


def predict_image(model, image_path, transform, device, class_names):
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
    
    parser = argparse.ArgumentParser(description='ConvNeXt V2 商品分类模型训练')
    parser.add_argument('--data_dir', type=str, default='..',
                        help='数据集目录 (指向项目根目录，包含image5文件夹)')
    parser.add_argument('--output_dir', type=str, default='../convnext_models',
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
                        help='训练设备')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device
    )
