import torch
import torch.nn as nn
from torchvision import models

# 定义深度学习模型

def get_model(model_name='resnet18', num_classes=27, pretrained=True):
    """
    Args:
        model_name (str): 模型名称
        num_classes (int): 类别数量 (PlantDoc 为 27)
        pretrained (bool): 是否使用 ImageNet 预训练权重
    """
    print(f"正在初始化模型: {model_name}, num_classes={num_classes}, pretrained={pretrained}")
    
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        # 修改全连接层 (fc) 以匹配 num_classes
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'efficientnet_b0':
        # EfficientNet-B0: 性能优于 ResNet50，且参数更少
        model = models.efficientnet_b0(pretrained=pretrained)
        # EfficientNet 的分类头结构不同: classifier[1] 是全连接层
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

