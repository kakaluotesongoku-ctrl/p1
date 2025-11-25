import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm
from src.dataset import get_dataloaders
from src.models import get_model

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()

def plot_loss_curve(log_file_path, save_dir=None):
    """
    读取类似 log.txt 文件，绘制 Loss 和 Accuracy 曲线
    日志示例格式:
    epoch,train_loss,valid_loss,train_acc,valid_acc
    1,0.56,0.48,0.82,0.85
    ...
    """
    import csv
    epochs = []
    train_loss = []
    valid_loss = []
    train_acc = []
    valid_acc = []
    with open(log_file_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            if len(row) < 5: continue
            epochs.append(int(row[0]))
            train_loss.append(float(row[1]))
            valid_loss.append(float(row[2]))
            train_acc.append(float(row[3]))
            valid_acc.append(float(row[4]))

    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, valid_loss, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    if save_dir:
        loss_fig_path = os.path.join(save_dir, "loss_curve.png")
        plt.savefig(loss_fig_path)
        print(f"Loss curve saved to {loss_fig_path}")
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_acc, label='Train Acc')
    plt.plot(epochs, valid_acc, label='Valid Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    if save_dir:
        acc_fig_path = os.path.join(save_dir, "acc_curve.png")
        plt.savefig(acc_fig_path)
        print(f"Accuracy curve saved to {acc_fig_path}")
    plt.show()

def analyze_bad_cases(model, dataloader, save_dir="bad_cases", max_save=5):
    """
    找出模型预测错误的样本，保存下来用于报告分析
    保存：原图、真实label、预测label
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = next(model.parameters()).device
    model.eval()
    saved_count = 0
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for i in range(inputs.size(0)):
                if preds[i] != labels[i] and saved_count < max_save:
                    img_np = inputs[i].cpu().numpy().transpose(1,2,0)  # HWC
                    img_np = np.clip(img_np * [0.229, 0.224, 0.225], 0, 1)  # 避免负值
                    plt.figure()
                    plt.imshow(img_np)
                    plt.title(f"True: {labels[i].item()} Pred: {preds[i].item()}")
                    bad_path = os.path.join(save_dir, f"badcase_{saved_count}_true{labels[i].item()}_pred{preds[i].item()}.png")
                    plt.savefig(bad_path)
                    plt.close()
                    saved_count += 1
            if saved_count >= max_save:
                print(f"Analysis done. Saved {saved_count} bad cases.")
                break
    if saved_count == 0:
        print("No mistakes found in this batch!")

def tta_predict(model, device, inputs):
    outputs = [model(inputs)]
    outputs.append(model(torch.flip(inputs, [3])))  # 水平翻转
    # outputs.append(model(torch.flip(inputs, [2])))  # 可选：垂直翻转
    outputs = torch.stack(outputs).mean(0)
    return outputs

def evaluate(model_path, data_dir, model_name='resnet50', batch_size=32, use_tta=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating model: {model_path} on {device}")

    _, _, test_loader, classes = get_dataloaders(data_dir, batch_size=batch_size)
    num_classes = len(classes)

    model = get_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            if use_tta:
                outputs = tta_predict(model, device, inputs)
            else:
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))

    save_dir = os.path.dirname(model_path)
    cm_path = os.path.join(save_dir, f'confusion_matrix_{"tta" if use_tta else "normal"}.png')
    plot_confusion_matrix(y_true, y_pred, classes, save_path=cm_path)

    # 补充：分析错误样本
    print("Analyzing bad cases (saving up to 5 error samples)...")
    analyze_bad_cases(model, test_loader, save_dir=os.path.join(save_dir, "bad_cases"), max_save=5)

if __name__ == "__main__":
    MODEL_PATH = r'results/best_resnet50_mixup.pth'
    DATA_DIR = r'data/raw'
    LOG_PATH = r'results/log.txt'

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please make sure you are in the project root.")
        exit(1)

    # 离线可视化 Loss/Accuracy 曲线
    if os.path.exists(LOG_PATH):
        plot_loss_curve(LOG_PATH, save_dir=os.path.dirname(LOG_PATH))

    # 运行普通评估
    evaluate(MODEL_PATH, DATA_DIR, model_name='resnet50', use_tta=False)

    # 运行 TTA 评估
    print("\nRunning TTA Evaluation...")
    evaluate(MODEL_PATH, DATA_DIR, model_name='resnet50', use_tta=True)