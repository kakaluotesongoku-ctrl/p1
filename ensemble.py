import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
import os

from src.dataset import get_dataloaders
from src.models import get_model
from src.evaluate import tta_predict

def ensemble_evaluate(models_info, data_dir, batch_size=32, use_tta=True):
    """
    Args:
        models_info: list of dict, e.g. 
                     [{'name': 'resnet50', 'path': '...', 'weight': 0.6}, 
                      {'name': 'efficientnet_b0', 'path': '...', 'weight': 0.4}]
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Ensemble Evaluation on {device}")
    
    # 1. 加载数据
    _, _, test_loader, classes = get_dataloaders(data_dir, batch_size=batch_size)
    num_classes = len(classes)
    
    # 2. 加载所有模型
    loaded_models = []
    weights = []
    
    for info in models_info:
        print(f"Loading {info['name']} from {info['path']}...")
        model = get_model(model_name=info['name'], num_classes=num_classes, pretrained=False)
        # 加载权重
        checkpoint = torch.load(info['path'], map_location=device)
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()
        
        loaded_models.append(model)
        weights.append(info['weight'])
        
    # 归一化权重
    weights = np.array(weights)
    weights = weights / weights.sum()
    print(f"Model weights: {weights}")
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Ensembling"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            batch_probs = torch.zeros(inputs.size(0), num_classes).to(device)
            
            for i, model in enumerate(loaded_models):
                if use_tta:
                    # tta_predict 返回的是 logits (未经过 softmax)
                    outputs = tta_predict(model, device, inputs)
                else:
                    outputs = model(inputs)
                
                # 转换为概率分布
                probs = F.softmax(outputs, dim=1)
                
                # 加权累加
                batch_probs += weights[i] * probs
            
            # 取最大概率对应的类别
            _, preds = torch.max(batch_probs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    # 计算准确率
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"Ensemble Test Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))

if __name__ == "__main__":
    DATA_DIR = r'data/raw'
    
    # 定义参与融合的模型
    # 建议给准确率高的模型更高的权重
    models_to_ensemble = [
        {
            'name': 'resnet50',
            'path': r'results/best_resnet50_mixup.pth',
            'weight': 0.7  # ResNet50 表现较好 (66%)
        },
        {
            'name': 'efficientnet_b0',
            'path': r'results/best_efficientnet_b0_mixup.pth',
            'weight': 0.3  # EfficientNet 表现稍弱 (62%)
        }
    ]
    
    ensemble_evaluate(models_to_ensemble, DATA_DIR, use_tta=True)
