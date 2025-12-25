import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#机器学习基础模型
def train_random_forest(features: np.ndarray, labels: np.ndarray) -> RandomForestClassifier:
    """训练随机森林模型（文档核心机器学习算法）"""
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(features, labels)
    return model

def train_svm(features: np.ndarray, labels: np.ndarray) -> SVC:
    """训练SVM模型（小样本故障识别优化）"""
    model = SVC(kernel="rbf", probability=True, random_state=42)
    model.fit(features, labels)
    return model

# 核心调用示例
if __name__ == "__main__":
    # 模拟故障特征数据（加速度峰值、速度有效值、磁场有效值）
    features = np.array([[41.29, 6.13, 56.74], [38.52, 5.87, 54.32], [45.18, 7.02, 58.91]])
    labels = np.array([1, 0, 2])  # 0-正常，1-轴承磨损，2-转子不平衡
    rf_model = train_random_forest(features, labels)
    svm_model = train_svm(features, labels)
    print("基础机器学习模型训练完成")