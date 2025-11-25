import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import os
from src.transforms import get_transforms

# 实现数据加载逻辑

class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)

def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    train_dir = os.path.join(data_dir, 'TRAIN')
    test_dir = os.path.join(data_dir, 'TEST')

    # 1. 定义数据集
    full_train_dataset = datasets.ImageFolder(root=train_dir)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=get_transforms('test'))

    classes = full_train_dataset.classes
    print(f"检测到 {len(classes)} 个类别: {classes[:5]}...")

    # 2. 划分训练集和验证集 (80% 训练, 20% 验证)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 3. 分别应用数据增强

    train_dataset = TransformSubset(train_dataset, transform=get_transforms('train'))
    val_dataset = TransformSubset(val_dataset, transform=get_transforms('val'))

    print(f"数据集加载完成: 训练集 {len(train_dataset)} 张, 验证集 {len(val_dataset)} 张, 测试集 {len(test_dataset)} 张")

    # 4. 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, classes

