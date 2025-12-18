# 构建融合机器学习模型矩阵
from sklearn.svm import SVC
from hminisom import MiniSom
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination


class MLModelMatrix:
    def __init__(self):
        # 1. SVM模型：高维特征分类（故障类型识别）
        self.svm = SVC(kernel='rbf', probability=True)
        # 2. SOM模型：无监督异常模式发现
        self.som = MiniSom(x=10, y=10, input_len=3, sigma=1.0, learning_rate=0.5)
        # 3. 贝叶斯网络：融合不确定性与专家先验
        self.bn_model = BayesianNetwork([('vibration_feat', 'fault_type'),
                                         ('temperature_feat', 'fault_type'),
                                         ('magnetic_feat', 'fault_type')])
        self.bn_infer = VariableElimination(self.bn_model)

    # 模型训练：从海量数据中学习健康状态规律
    def train(self, train_data, train_labels):
        # 提取高维特征（振动频谱、温度趋势、磁场强度）
        features = self._extract_high_dim_features(train_data)
        # 分别训练三大模型
        self.svm.fit(features, train_labels)
        self.som.train_random(features, num_iteration=100)
        self._train_bayesian_network(features, train_labels)

    # 联合推理：输出初步诊断结果
    def infer(self, test_data):
        test_feat = self._extract_high_dim_features(test_data)
        svm_pred = self.svm.predict_proba(test_feat)
        som_anomaly = self.som.distance_map().max() - self.som.distance_map()
        bn_pred = self.bn_infer.query(variables=['fault_type'], evidence={
            'vibration_feat': test_feat[0][0],
            'temperature_feat': test_feat[0][1],
            'magnetic_feat': test_feat[0][2]
        })
        return {"svm": svm_pred, "som_anomaly": som_anomaly, "bayesian": bn_pred}


# 初始化模型矩阵并加载预训练参数（演示用）
model_matrix = MLModelMatrix()
model_matrix.load_pretrained_weights("pretrained_model_weights.pkl")