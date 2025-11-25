import torch
import torch.optim as optim
import torch.nn as nn
import os
import time
from tqdm import tqdm
from src.dataset import get_dataloaders
from src.models import get_model

import argparse


import numpy as np

# Mixup 辅助函数
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train(args):
    # 0. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Config: {args}")

    # 1. 准备数据
    print("Loading data...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        args.data_dir, 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    num_classes = len(classes)
    
    # 2. 准备模型
    model = get_model(model_name=args.model, num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    # 3. 定义损失函数和优化器
    # 使用 Label Smoothing (标签平滑) 防止过拟合，提升泛化能力
    if args.label_smoothing > 0:
        try:
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
            print(f"Using Label Smoothing: {args.label_smoothing}")
        except TypeError:
            print("Warning: Your PyTorch version doesn't support label_smoothing. Using default CrossEntropyLoss.")
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 使用 SGD + Momentum
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    # 使用 CosineAnnealingLR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 4. 训练循环
    best_acc = 0.0
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.2, use_cuda=torch.cuda.is_available())
            inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs, targets_a, targets_b))

            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            pbar.set_postfix({'loss': loss.item()})

        scheduler.step()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Train Loss: {epoch_loss:.4f}')

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        if val_acc > best_acc:
            best_acc = val_acc
            # 文件名包含超参数信息，方便对比
            best_model_name = f'best_{args.model}_mixup.pth'
            best_model_path = os.path.join(save_dir, best_model_name)
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved to {best_model_path}')

    print(f'Training complete. Best val Acc: {best_acc:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plant Disease Training')
    parser.add_argument('--data_dir', type=str, default=r'd:\code\plant\plant-disease-recognition\data\raw', help='Path to dataset')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50', 'efficientnet_b0'], help='Model architecture')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing epsilon (default: 0.0)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    
    args = parser.parse_args()
    train(args)


