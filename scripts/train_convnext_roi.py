
"""
第四步：训练 ConvNeXt V2 (双卡并行加速 + 防过拟合终极版)
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

CLASSES = ["hks_large", "hks_small", "hn_can", "jlb_can", "kkkl_can", "wlj_can", "xb_wt", "xb"]

class GTROIDataset(Dataset):
    def __init__(self, roi_dir, split='train', transform=None):
        self.roi_dir = Path(roi_dir)
        self.split = split
        self.transform = transform
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}
        
        split_dir = self.roi_dir / split
        if not split_dir.exists(): return
            
        for class_name in CLASSES:
            class_dir = split_dir / class_name
            if not class_dir.exists(): continue
            class_images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            for img_path in class_images:
                self.samples.append((str(img_path), self.class_to_idx[class_name]))
        print(f"📦 加载 {split} 数据集: {len(self.samples)} 张")
    
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        return self.transform(image) if self.transform else image, label

def get_transforms(img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def train_model(roi_dir='gt_roi_dataset', output_dir='convnext_models', epochs=20, batch_size=32, lr=0.0001, device='cuda', multi_gpu=False):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_t, val_t = get_transforms()
    train_dataset = GTROIDataset(roi_dir, 'train', train_t)
    val_dataset = GTROIDataset(roi_dir, 'val', val_t)
    
    if len(train_dataset) == 0: return
    
    # 🏎️ 如果用双卡，Batch Size 相当于翻倍，我们可以适当调大 DataLoader 的效率
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = timm.create_model('convnextv2_atto', pretrained=True, num_classes=8, drop_rate=0.4)
    model = model.to(device)
    
    # ================= 核心：双卡/多卡并行支持 =================
    if multi_gpu and torch.cuda.device_count() > 1:
        print(f"🚀 启用多卡并行加速! 检测到 {torch.cuda.device_count()} 张 GPU。")
        model = nn.DataParallel(model)
    else:
        print("🏍️ 启用单卡/CPU模式训练。")
        
    criterion = nn.CrossEntropyLoss()
    
    # 阶段 1：冻结主干
    print("\n❄️ 阶段 1: 冻结主干网络，仅训练分类头...")
    # 注意：如果被 DataParallel 包裹，模型的属性访问前面要加 module.
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    for name, param in base_model.named_parameters():
        if 'head' not in name:
            param.requires_grad = False
            
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr * 5, weight_decay=0.05)
    best_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        # 阶段 2：解冻主干
        if epoch == 6:
            print("\n🔥 阶段 2: 解冻主干网络，开始全量微调...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - 5)
            
        model.train()
        train_loss, train_correct, total = 0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}'):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += outputs.max(1)[1].eq(labels).sum().item()
            total += labels.size(0)
            
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()
                val_correct += outputs.max(1)[1].eq(labels).sum().item()
                val_total += labels.size(0)
                
        if epoch >= 6: scheduler.step()
            
        t_acc = 100. * train_correct / total
        v_acc = 100. * val_correct / val_total
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Acc: {t_acc:.2f}% | Val Loss: {val_loss/len(val_loader):.4f}, Acc: {v_acc:.2f}%")
        
        if v_acc > best_acc:
            best_acc = v_acc
            # 🚨 关键：如果是多卡训练，保存时必须剥离 DataParallel 层，否则后续单卡推理会报错！
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({'model_state_dict': state_dict, 'classes': CLASSES}, output_path / 'best_model.pth')
            print(f"💾 保存最佳模型: {best_acc:.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--roi_dir', type=str, default='gt_roi_dataset')
    parser.add_argument('--output_dir', type=str, default='convnext_models')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--multi_gpu', action='store_true', help='是否使用多卡训练')
    args = parser.parse_args()
    train_model(roi_dir=args.roi_dir, output_dir=args.output_dir, epochs=args.epochs, lr=args.lr, multi_gpu=args.multi_gpu)