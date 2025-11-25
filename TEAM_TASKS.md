# 植物病害识别分类 - 双人协作项目计划

## 🎯 项目目标与策略
- **核心目标**: 完成 PlantDoc 数据集的图像分类任务。
- **策略**: **“保稳冲优”**。优先确保 CNN (ResNet) + 传统方法 (SVM) 的完整实现与对比，确保拿到基础分和加分项。
- **分工模式**: 前后端分离式。一人主攻深度学习，一人主攻数据处理、传统方法与分析。

## 👥 角色定义
- **队友 A (深度学习攻坚)**: 负责核心深度学习模型的构建、训练与调优。
- **队友 B (广度探索与分析)**: 负责数据处理、传统机器学习方法实现、结果可视化与深度分析。

## 📂 项目结构与文件职责 (结构初始化)

```text
plant-disease-recognition/
├── data/                  # [共同] 存放数据集
│   ├── raw/               # 原始 PlantDoc 数据
│   └── processed/         # 预处理后的数据
├── notebooks/             # [共同] Jupyter Notebooks 用于实验和可视化
│   ├── 01_data_split.ipynb        # [共同] 数据集划分
│   ├── 02_eda_visualization.ipynb # [队友 B] 数据探索性分析
│   └── 03_training_experiments.ipynb # [队友 A] 训练过程记录
├── src/                   # 源代码
│   ├── dataset.py         # [队友 A] 数据加载器 (Dataset/DataLoader)
│   ├── models.py          # [队友 A] 深度学习模型定义 (CNN, ResNet等)
│   ├── train.py           # [队友 A] 深度学习训练脚本
│   ├── transforms.py      # [队友 B] 数据增强与预处理策略
│   ├── traditional_ml.py  # [队友 B] 传统机器学习流程 (特征提取+SVM)
│   ├── evaluate.py        # [队友 B] 评估工具 (混淆矩阵绘制, 指标计算)
│   └── utils.py           # [共同] 通用工具函数
├── results/               # [共同] 存放模型权重、日志、混淆矩阵图
└── requirements.txt       # [共同] 依赖库
```

## 📅 详细分工表

| 阶段 | 队友 A (深度学习攻坚) | 队友 B (广度探索与分析) | 协作关键点 |
| :--- | :--- | :--- | :--- |
| **1. 准备** | **下载数据集**：将 PlantDoc 数据集解压到 `data/raw`。 | **环境配置**：确保两人都安装了 `torch`, `torchvision`, `opencv-python`, `scikit-learn`。 | **必须统一数据划分**：运行 `notebooks/01_data_split.ipynb` (需新建)，生成 train/val/test 的文件列表，确保两人用的数据是一样的。 |
| **2. 开发** | **完善 `src/dataset.py`**：实现 `PlantDocDataset` 类，确保能正确读取图片和标签。 | **完善 `src/transforms.py`**：实现数据增强（旋转、裁剪等），这是 A 和 B 都要用的模块。 | A 的 Dataset 需要调用 B 写的 transforms，这里需要沟通好接口。 |
| **3. 核心** | **完善 `src/models.py` & `src/train.py`**：<br>1. 实现 ResNet18/50 的微调代码。<br>2. 写好训练循环，保存验证集准确率最高的模型。 | **完善 `src/traditional_ml.py`**：<br>1. 实现 HOG/SIFT 特征提取。<br>2. 使用 SVM 进行训练和预测。<br>3. 记录准确率，作为 Baseline。 | B 的传统方法结果将作为 A 的深度学习结果的对比基准。 |
| **4. 实验** | **消融实验**：<br>1. 改变 Batch Size (16 vs 32)。<br>2. 改变学习率。<br>3. 对比 ResNet18 vs ResNet50。 | **分析与可视化 (`src/evaluate.py`)**：<br>1. 绘制 A 训练过程的 Loss 曲线。<br>2. 绘制 A 和 B 模型的混淆矩阵。<br>3. **失败案例分析**：找出 A 模型分错的图片，分析原因。 | B 需要拿 A 训练好的模型权重文件 (`.pth`) 来跑评估脚本。 |
| **5. 报告** | **撰写部分**：<br>1. 深度学习模型架构图。<br>2. 训练策略描述。<br>3. 深度学习实验结果表格。 | **撰写部分**：<br>1. 数据预处理流程。<br>2. 传统方法原理。<br>3. 错误分析与可视化图表。<br>4. 总结与对比。 | 最后合并成一份 PDF。 |

## ✅ 任务清单 (Checklist)

### 阶段 1: 准备与环境
- [] **[共同]** 下载 PlantDoc 数据集并放入 `data/raw`
- [] **[共同]** 安装依赖 (`pip install -r requirements.txt`)
- [] **[共同]** 创建并运行 `notebooks/01_data_split.ipynb`，生成数据划分 CSV/TXT

### 阶段 2: 基础模块开发
- [] **[队友 B]** 完成 `src/transforms.py` (数据增强)
- [] **[队友 A]** 完成 `src/dataset.py` (数据加载，依赖 transforms)
- [] **[队友 A]** 验证 DataLoader 是否能正常读取数据

### 阶段 3: 核心模型实现
- [] **[队友 A]** 完成 `src/models.py` (ResNet 定义 + EfficientNet)
- [] **[队友 A]** 完成 `src/train.py` (训练循环 + Mixup + Label Smoothing)
- [] **[队友 A]** 跑通第一个 Epoch，确保无报错
- [] **[队友 B]** 完成 `src/traditional_ml.py` (HOG特征提取)
- [] **[队友 B]** 训练 SVM 模型并记录 Baseline 准确率

### 阶段 4: 实验与优化
- [] **[队友 A]** 运行 ResNet18 完整训练 (例如 20-50 epochs)
- [] **[队友 A]** 运行 ResNet50 对比实验
- [] **[队友 A]** (可选) 调整超参数 (LR, Batch Size) 进行消融实验
- [] **[队友 A]** (额外) 实现 EfficientNet-B0 和 模型融合 (Ensemble)
- [] **[队友 B]** 完成 `src/evaluate.py` (A 已实现基础 TTA 和混淆矩阵，B 需补充 Loss 曲线)
- [] **[队友 B]** 使用 A 的最佳模型权重进行评估和可视化
- [] **[队友 B]** 挑选 3-5 张典型错误样本进行分析

### 阶段 5: 报告撰写
- [] **[队友 A]** 撰写深度学习方法与实验结果部分
- [] **[队友 B]** 撰写数据处理、传统方法、分析与总结部分
- [] **[共同]** 合并报告，检查引用规范，导出 PDF
