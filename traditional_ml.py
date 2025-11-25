import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import joblib

# 负责人: 队友 B
# 任务: 实现传统机器学习流程 (HOG 特征提取 + 分类)

def extract_hog_features(image_path):
    """
    提取图像的 HOG 特征
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 64)) # HOG 推荐尺寸
    feature = hog(img, orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
    return feature

def load_data(data_dir):
    """
    加载数据目录下所有图片与标签
    """
    X = []
    y = []
    classes = sorted(os.listdir(data_dir))
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            if os.path.splitext(fpath)[-1].lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            feature = extract_hog_features(fpath)
            X.append(feature)
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y, classes

def train_traditional_model(data_dir, save_path='svm_model.joblib'):
    # 1. 加载数据
    X, y, classes = load_data(data_dir)
    # 2. 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    # 3. 训练 SVM 分类器
    clf = SVC(kernel='linear', probability=True, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, save_path)
    # 4. 评估
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"SVM Baseline Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=classes))
    return acc

if __name__ == "__main__":
    # 运行训练流程
    DATA_DIR = r'data/raw'
    train_traditional_model(DATA_DIR)