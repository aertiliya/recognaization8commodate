# YOLO11 商品检测 + ConvNeXt V2 分类

## 项目概述

本项目实现商品识别的两阶段流程：
1. **YOLO11 检测**: 检测图像中的商品位置
2. **ConvNeXt V2 分类**: 对检测到的商品进行细粒度分类

### 数据集信息
- **类别数量**: 8类
- **类别名称**: hks_large, hks_small, hn_can, jlb_can, kkkl_can, wlj_can, xb_wt, xb
- **训练集**: 每类400张，共3200张
- **验证集**: 从训练集划分20%

## 工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     完整工作流程                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │ 数据标注  │ -> │ YOLO检测  │ -> │ ConvNeXt │ -> │ 最终结果  │ │
│  │ X-AnyLab │    │  训练    │    │  分类训练 │    │ 检测+分类 │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练YOLO11检测模型

```bash
cd scripts
python 02_train_yolo11.py --mode train --model_size s --epochs 100
```

### 3. 训练ConvNeXt V2分类模型

```bash
python train_convnext.py --model_name convnextv2_atto --epochs 50
```

### 4. 运行推理（检测+分类）

```bash
# 处理单张图像
python inference.py --source ../image5/test/xb485.jpg --output ../results

# 处理整个目录
python inference.py --source ../image5/test --output ../results
```

## 文件结构

```
项目目录/
├── image5/
│   ├── train/              # 训练集图像和标注
│   └── test/               # 测试集
├── scripts/
│   ├── 01_convert_json_to_yolo.py  # 格式转换
│   ├── 02_train_yolo11.py          # YOLO训练
│   ├── 03_auto_label.py            # 辅助标注
│   ├── train_convnext.py           # ConvNeXt训练
│   ├── inference.py                # 推理脚本
│   ├── data.yaml                   # 数据集配置
│   └── run_pipeline.py             # 一键运行
├── yolo_dataset/           # YOLO格式数据集
├── convnext_models/        # ConvNeXt模型输出
│   ├── best_model.pth      # 最佳模型
│   ├── final_model.pth     # 最终模型
│   └── training_history.json
└── results/                # 推理结果
```

## ConvNeXt V2 模型选择

| 模型 | 参数量 | 速度 | 精度 | 适用场景 |
|------|--------|------|------|----------|
| convnextv2_atto | 3.7M | 最快 | 较低 | 快速原型 |
| convnextv2_femto | 5.2M | 快 | 中等 | 平衡选择 |
| convnextv2_pico | 9.1M | 快 | 较高 | 推荐使用 |
| convnextv2_nano | 15.6M | 中等 | 高 | 追求精度 |
| convnextv2_tiny | 28.6M | 较慢 | 更高 | 高精度需求 |

## 训练参数

### YOLO11 训练参数
```bash
python 02_train_yolo11.py \
    --mode train \
    --model_size s \
    --epochs 100 \
    --batch 16 \
    --img_size 640
```

### ConvNeXt V2 训练参数
```bash
python train_convnext.py \
    --model_name convnextv2_atto \
    --img_size 224 \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.001
```

## 推理使用

### 单张图像推理
```bash
python inference.py \
    --yolo_model last.pt \
    --convnext_model ../convnext_models/best_model.pth \
    --source ../image5/test/xb485.jpg \
    --output ../results \
    --conf 0.25 \
    --iou 0.45
```

### 批量推理
```bash
python inference.py \
    --source ../image5/test \
    --output ../results
```

### 输出格式
推理结果包含：
- JSON标注文件（X-AnyLabeling格式）
- 可视化结果图像（带标注框和类别）
- 汇总结果文件（all_results.json）

## 性能优化建议

### 训练优化
1. 使用混合精度训练（自动启用）
2. 增大batch_size（根据GPU内存调整）
3. 使用更大的模型（convnextv2_pico或nano）
4. 增加训练轮数

### 推理优化
1. 使用ONNX格式导出模型
2. 批量处理图像
3. 使用TensorRT加速

## 常见问题

### Q: ConvNeXt训练时显存不足？
A: 减小batch_size或使用更小的模型（convnextv2_atto）

### Q: 如何选择合适的模型大小？
A: 
- 快速验证: convnextv2_atto
- 平衡选择: convnextv2_pico
- 追求精度: convnextv2_nano

### Q: 推理速度慢怎么办？
A: 
1. 使用更小的模型
2. 减小输入图像大小
3. 使用GPU加速
4. 考虑模型量化

## 模型评估

训练完成后会自动生成：
- 分类报告（precision, recall, f1-score）
- 混淆矩阵
- 训练曲线（loss和accuracy）

## 注意事项

1. 确保数据集路径正确
2. 训练前检查GPU可用性
3. 根据硬件调整batch_size
4. 保存训练好的模型用于推理