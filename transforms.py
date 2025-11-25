from torchvision import transforms

# 负责人: 队友 B
# 说明: 队友 A 为了跑通训练流程，提供了一个基础版本。
# 任务: 队友 B 请在此基础上优化数据增强策略 (如添加 ColorJitter, Rotation 等)

# 这个文件会被 src/dataset.py 引用

def get_transforms(split='train'):
    """
    Args:
        split (str): 'train' (增强) 或 'val'/'test' (仅需 Resize/Normalize)
    Returns:
        torchvision.transforms.Compose
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),                                  # 统一尺寸
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),            # 较大变化范围
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),                          # 旋转幅度略调大
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.1),                # 增大抖动范围
            transforms.RandomGrayscale(p=0.1),                              # 有 10% 概率灰度化
            transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)), # 随机高斯模糊
            transforms.RandomAffine(
                degrees=0, translate=(0.08,0.08), scale=(0.95,1.05), shear=10
            ),                                                              # 仿射变换
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])