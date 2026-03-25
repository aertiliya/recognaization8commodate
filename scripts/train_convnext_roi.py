"""
第三步：训练 ConvNeXt V2（修复了训练集和验证集读取重复的Bug）
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

CLASSES = [
    "hks_large", "hks_small", "hn_can", "jlb_can",
    "kkkl_can", "wlj_can", "xb_wt", "xb"
]

class GTROIDataset(Dataset):
    def __init__(self, roi_dir, split='train', transform=None):
        self.roi_dir = Path(roi_dir)
        self.split = split
        self.transform = transform
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}
        
        # 🚨 关键修复：严格定向到 train 或 val 子目录！
        split_dir = self.roi_dir / split
        
        if not split_dir.exists():
            print(f"⚠️ 警告: 找不到拆分目录 {split_dir}")
            return
            
        for class_name in CLASSES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            
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
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
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
        pbar.set_postfix({'loss': running_loss / len(dataloader), 'acc': 100. * correct / total})
    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
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
    return running_loss / len(dataloader), 100. * correct / total, all_preds, all_labels

def train_model(roi_dir='gt_roi_dataset', output_dir='convnext_models', model_name='convnextv2_atto', img_size=224, batch_size=32, epochs=50, lr=0.001, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    roi_dir = os.path.abspath(roi_dir)
    output_path = Path(os.path.abspath(output_dir))
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_transform, val_transform = get_transforms(img_size)
    
    # 🚨 关键修复：分别明确加载 train 和 val
    train_dataset = GTROIDataset(roi_dir, 'train', train_transform)
    val_dataset = GTROIDataset(roi_dir, 'val', val_transform)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("❌ 错误: 数据集为空，请检查裁剪步骤！")
        return None, None
    
    # 直接使用，不再做 random_split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = create_model(num_classes=len(CLASSES), model_name=model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model_state_dict': model.state_dict(), 'classes': CLASSES}, output_path / 'best_model.pth')
            print(f"💾 保存最佳模型，准确率: {best_acc:.2f}%")
            
    return model, history

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--roi_dir', type=str, default='gt_roi_dataset')
    parser.add_argument('--output_dir', type=str, default='convnext_models')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    args = parser.parse_args()
    train_model(roi_dir=args.roi_dir, output_dir=args.output_dir, epochs=args.epochs,lr=args.lr)