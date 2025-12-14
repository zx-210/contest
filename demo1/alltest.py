
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import joblib
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime


# ============= 数据模型定义 =============
class DeviceType(Enum):
    """设备类型枚举"""
    MOTOR = "电机"
    PUMP = "泵"
    FAN = "风机"
    COMPRESSOR = "压缩机"
    CENTRIFUGAL_PUMP = "离心泵"
    CENTRIFUGAL_FAN = "离心式风机"


class FaultType(Enum):
    """故障类型枚举"""
    UNBALANCE = "转子不平衡"
    MISALIGNMENT = "转子不对中"
    LOOSENESS = "松动或碰摩"
    CAVITATION = "汽蚀"
    TURBULENCE = "紊流"
    ROTOR_BAR_LOOSE = "转子条松动"
    BEARING_CAGE_WEAR = "轴承保持架磨损"
    BEARING_ROLLER_WEAR = "轴承滚动体磨损"
    BEARING_OUTER_RACE_WEAR = "轴承外圈磨损"
    BEARING_INNER_RACE_WEAR = "轴承内圈磨损"
    BEARING_LUBRICATION = "轴承润滑不良"
    OIL_WHIRL = "油膜涡动"


@dataclass
class SensorData:
    """传感器数据结构"""
    timestamp: datetime
    device_id: str
    # 振动数据 (3轴)
    vibration_x: np.ndarray  # 水平
    vibration_y: np.ndarray  # 垂直
    vibration_z: np.ndarray  # 轴向
    # 温度
    temperature: float
    # 磁场数据 (3轴)
    magnetic_x: float
    magnetic_y: float
    magnetic_z: float
    # 采样参数
    sampling_rate: int = 51200  # Z轴最高51.2kHz
    sample_points: int = 25600  # Z轴采样点数


@dataclass
class FaultDiagnosisResult:
    """故障诊断结果"""
    device_id: str
    timestamp: datetime
    fault_type: FaultType
    confidence: float  # 置信度 0-1
    severity: str  # 轻微/中等/严重
    location: str  # 故障部位
    recommendation: str  # 维修建议
    features: Dict[str, float]  # 特征值
    waveform_data: Optional[np.ndarray] = None


# ============= 特征提取模块 =============
class FeatureExtractor:
    """特征提取器 - 从原始数据提取故障特征"""

    @staticmethod
    def extract_vibration_features(vibration_data: np.ndarray,
                                   sampling_rate: float) -> Dict[str, float]:
        """
        提取振动信号特征
        包含时域、频域特征
        """
        features = {}

        # 时域特征
        features['rms'] = np.sqrt(np.mean(vibration_data ** 2))  # 有效值
        features['peak'] = np.max(np.abs(vibration_data))  # 峰值
        features['kurtosis'] = pd.Series(vibration_data).kurtosis()  # 峭度
        features['skewness'] = pd.Series(vibration_data).skew()  # 偏度
        features['crest_factor'] = features['peak'] / features['rms'] if features['rms'] > 0 else 0

        # 频域特征 (FFT变换)
        n = len(vibration_data)
        freq = np.fft.rfftfreq(n, d=1 / sampling_rate)
        fft_vals = np.abs(np.fft.rfft(vibration_data))

        if len(fft_vals) > 0:
            features['dominant_freq'] = freq[np.argmax(fft_vals)]
            features['dominant_amp'] = np.max(fft_vals)

            # 提取倍频特征 (1X, 2X, 3X...)
            if features['dominant_freq'] > 0:
                for i in range(1, 4):
                    target_freq = features['dominant_freq'] * i
                    idx = np.argmin(np.abs(freq - target_freq))
                    if idx < len(fft_vals):
                        features[f'harmonic_{i}X'] = fft_vals[idx]

        return features

    @staticmethod
    def extract_temperature_features(temperature: float,
                                     baseline_temp: float) -> Dict[str, float]:
        """提取温度特征"""
        return {
            'temperature': temperature,
            'temp_deviation': temperature - baseline_temp,
            'temp_rate_of_change': 0  # 需要历史数据计算
        }

    @staticmethod
    def extract_magnetic_features(magnetic_x: float,
                                  magnetic_y: float,
                                  magnetic_z: float) -> Dict[str, float]:
        """提取磁场特征"""
        magnetic_vector = np.array([magnetic_x, magnetic_y, magnetic_z])
        magnitude = np.linalg.norm(magnetic_vector)

        return {
            'magnetic_magnitude': magnitude,
            'magnetic_x': magnetic_x,
            'magnetic_y': magnetic_y,
            'magnetic_z': magnetic_z
        }


# ============= 机器学习模型模块 =============
class FaultDiagnosisModel:
    """
    故障诊断模型基类
    支持多种算法：随机森林、深度神经网络等
    """

    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.feature_names = None

    def train(self, X: pd.DataFrame, y: pd.Series):
        """训练模型"""
        if self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

            self.model.fit(X_train, y_train)
            self.feature_names = X.columns.tolist()

            # 评估模型
            accuracy = self.model.score(X_test, y_test)
            logging.info(f"模型训练完成，测试集准确率: {accuracy:.4f}")

        elif self.model_type == "neural_network":
            # 深度学习模型实现
            import tensorflow as tf

            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(len(y.unique()), activation='softmax')
            ])

            self.model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            self.model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

    def predict(self, features: Dict) -> Tuple[FaultType, float]:
        """预测故障类型"""
        if self.model is None:
            raise ValueError("模型未训练")

        # 将特征字典转换为模型输入格式
        feature_vector = [features.get(name, 0) for name in self.feature_names]
        feature_array = np.array(feature_vector).reshape(1, -1)

        if self.model_type == "random_forest":
            prediction = self.model.predict(feature_array)[0]
            proba = self.model.predict_proba(feature_array)[0]
            confidence = np.max(proba)

        elif self.model_type == "neural_network":
            prediction_proba = self.model.predict(feature_array)[0]
            prediction = np.argmax(prediction_proba)
            confidence = prediction_proba[prediction]

        # 将预测结果映射到故障类型
        fault_type = self._map_prediction_to_fault(prediction)
        return fault_type, confidence

    def _map_prediction_to_fault(self, prediction_idx: int) -> FaultType:
        """将预测索引映射到具体的故障类型"""
        # 这里需要根据实际训练数据的标签映射
        fault_mapping = {
            0: FaultType.UNBALANCE,
            1: FaultType.MISALIGNMENT,
            2: FaultType.LOOSENESS,
            3: FaultType.BEARING_INNER_RACE_WEAR,
            # ... 添加更多映射
        }
        return fault_mapping.get(prediction_idx, FaultType.UNBALANCE)

    def save_model(self, filepath: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("没有可保存的模型")
        joblib.dump(self.model, filepath)

    def load_model(self, filepath: str):
        """加载模型"""
        self.model = joblib.load(filepath)


# ============= 专家知识系统模块 =============
class ExpertKnowledgeSystem:
    """
    专家知识系统 - 结合领域知识进行故障判断
    对应文档中的"专家知识图谱技术"
    """

    def __init__(self):
        self.knowledge_base = self._initialize_knowledge_base()

    def _initialize_knowledge_base(self) -> Dict:
        """初始化专家知识库"""
        return {
            "不平衡特征": {
                "description": "转子质量分布不均匀",
                "indicators": ["1X频率突出", "振动相位稳定", "轴向振动小"],
                "severity_thresholds": {
                    "轻微": {"1X振幅": 2.0, "相位稳定性": 0.8},
                    "严重": {"1X振幅": 5.0, "相位稳定性": 0.9}
                }
            },
            "不对中特征": {
                "description": "转子轴线不重合",
                "indicators": ["2X频率突出", "轴向振动大", "反相振动"],
                "severity_thresholds": {
                    "轻微": {"2X/1X比率": 0.3, "轴向振动": 1.5},
                    "严重": {"2X/1X比率": 0.5, "轴向振动": 3.0}
                }
            },
            "轴承故障": {
                "description": "滚动轴承损伤",
                "indicators": ["高频共振", "包络谱特征频率", "冲击脉冲"],
                "fault_frequencies": {
                    "内圈": "BPFI",
                    "外圈": "BPFO",
                    "滚动体": "BSF",
                    "保持架": "FTF"
                }
            }
        }

    def analyze_with_expert_rules(self, features: Dict,
                                  device_type: DeviceType) -> List[Dict]:
        """应用专家规则进行分析"""
        diagnoses = []

        # 检查不平衡特征
        if features.get('dominant_freq', 0) > 0:
            harmonic_1x = features.get('harmonic_1X', 0)
            harmonic_2x = features.get('harmonic_2X', 0)

            # 规则1: 如果1X频率占主导且2X/1X比率低，可能是不平衡
            if harmonic_1x > harmonic_2x * 3:
                imbalance_score = self._calculate_imbalance_score(features)
                diagnoses.append({
                    "fault_type": FaultType.UNBALANCE,
                    "score": imbalance_score,
                    "evidence": f"1X振幅: {harmonic_1x:.4f}, 2X/1X比率: {harmonic_2x / harmonic_1x:.4f}"
                })

            # 规则2: 如果2X频率显著，可能是不对中
            if harmonic_2x > harmonic_1x * 0.5:
                misalignment_score = self._calculate_misalignment_score(features)
                diagnoses.append({
                    "fault_type": FaultType.MISALIGNMENT,
                    "score": misalignment_score,
                    "evidence": f"2X/1X比率: {harmonic_2x / harmonic_1x:.4f}"
                })

        # 检查轴承故障特征 (通过包络谱分析)
        if features.get('kurtosis', 0) > 3.5:  # 高峭度表示冲击
            bearing_score = self._calculate_bearing_score(features)
            if bearing_score > 0.7:
                # 进一步识别具体轴承故障类型
                bearing_type = self._identify_bearing_fault_type(features)
                diagnoses.append({
                    "fault_type": bearing_type,
                    "score": bearing_score,
                    "evidence": f"峭度值: {features.get('kurtosis', 0):.4f}, 峰值因子: {features.get('crest_factor', 0):.4f}"
                })

        return diagnoses

    def _calculate_imbalance_score(self, features: Dict) -> float:
        """计算不平衡故障得分"""
        harmonic_1x = features.get('harmonic_1X', 0)
        baseline = 0.1  # 基线值，需要根据历史数据调整
        score = min(harmonic_1x / (baseline * 5), 1.0)
        return score

    def _calculate_misalignment_score(self, features: Dict) -> float:
        """计算不对中故障得分"""
        harmonic_2x = features.get('harmonic_2X', 0)
        harmonic_1x = features.get('harmonic_1X', 1)
        ratio = harmonic_2x / harmonic_1x
        score = min(ratio / 0.5, 1.0)  # 如果比率达到0.5，得分为1
        return score

    def _calculate_bearing_score(self, features: Dict) -> float:
        """计算轴承故障得分"""
        kurtosis = features.get('kurtosis', 3)
        crest_factor = features.get('crest_factor', 1)

        # 基于峭度和峰值因子的综合评分
        kurtosis_score = min((kurtosis - 3) / 4, 1.0)  # 正常峭度为3
        crest_score = min((crest_factor - 1) / 5, 1.0)  # 正常峰值因子约1-3

        return 0.6 * kurtosis_score + 0.4 * crest_score

    def _identify_bearing_fault_type(self, features: Dict) -> FaultType:
        """识别具体轴承故障类型"""
        # 这里需要更复杂的包络谱分析
        # 简化的逻辑：根据特征频率比率判断
        envelope_features = features.get('envelope_spectrum', {})

        if 'inner_race_freq' in envelope_features:
            return FaultType.BEARING_INNER_RACE_WEAR
        elif 'outer_race_freq' in envelope_features:
            return FaultType.BEARING_OUTER_RACE_WEAR
        elif 'rolling_element_freq' in envelope_features:
            return FaultType.BEARING_ROLLER_WEAR
        else:
            return FaultType.BEARING_LUBRICATION


# ============= 主诊断引擎 =============
class AIMFaultDiagnosisEngine:
    """
    AiM智能故障诊断引擎 - 主控制器
    结合机器学习模型和专家系统
    """

    def __init__(self, device_id: str, device_type: DeviceType):
        self.device_id = device_id
        self.device_type = device_type
        self.ml_model = FaultDiagnosisModel(model_type="random_forest")
        self.expert_system = ExpertKnowledgeSystem()
        self.feature_extractor = FeatureExtractor()
        self.history_data = []  # 历史数据存储
        self.model_trained = False

        # 加载设备特定模型 (实现"一机一模型")
        self._load_device_specific_model()

    def _load_device_specific_model(self):
        """加载设备特定模型"""
        model_path = f"models/{self.device_id}_model.pkl"
        try:
            self.ml_model.load_model(model_path)
            self.model_trained = True
            logging.info(f"已加载设备 {self.device_id} 的专属模型")
        except FileNotFoundError:
            logging.warning(f"设备 {self.device_id} 的专属模型不存在，将使用通用模型")
            self._load_general_model()

    def _load_general_model(self):
        """加载通用模型"""
        # 根据设备类型加载不同的通用模型
        general_model_path = f"models/general_{self.device_type.value}_model.pkl"
        try:
            self.ml_model.load_model(general_model_path)
            self.model_trained = True
        except FileNotFoundError:
            logging.warning("通用模型也不存在，需要进行训练")

    def process_sensor_data(self, sensor_data: SensorData) -> FaultDiagnosisResult:
        """
        处理传感器数据，进行故障诊断
        这是主入口函数
        """
        # 1. 特征提取
        features = self._extract_all_features(sensor_data)

        # 2. 机器学习模型预测
        ml_fault_type, ml_confidence = self.ml_model.predict(features)

        # 3. 专家系统分析
        expert_diagnoses = self.expert_system.analyze_with_expert_rules(
            features, self.device_type
        )

        # 4. 结果融合 (结合ML和专家系统)
        final_diagnosis = self._fuse_diagnoses(
            ml_fault_type, ml_confidence, expert_diagnoses
        )

        # 5. 生成诊断结果
        result = FaultDiagnosisResult(
            device_id=self.device_id,
            timestamp=sensor_data.timestamp,
            fault_type=final_diagnosis["fault_type"],
            confidence=final_diagnosis["confidence"],
            severity=self._determine_severity(features, final_diagnosis["fault_type"]),
            location=self._locate_fault(final_diagnosis["fault_type"]),
            recommendation=self._generate_recommendation(final_diagnosis),
            features=features
        )

        # 6. 存储到历史数据库
        self._store_to_history(result)

        return result

    def _extract_all_features(self, sensor_data: SensorData) -> Dict:
        """提取所有特征"""
        features = {}

        # 振动特征 (三轴分别提取)
        vibration_features_x = self.feature_extractor.extract_vibration_features(
            sensor_data.vibration_x, sensor_data.sampling_rate
        )
        vibration_features_y = self.feature_extractor.extract_vibration_features(
            sensor_data.vibration_y, sensor_data.sampling_rate
        )
        vibration_features_z = self.feature_extractor.extract_vibration_features(
            sensor_data.vibration_z, sensor_data.sampling_rate
        )

        # 合并振动特征，添加轴标识
        for key, value in vibration_features_x.items():
            features[f"vib_x_{key}"] = value
        for key, value in vibration_features_y.items():
            features[f"vib_y_{key}"] = value
        for key, value in vibration_features_z.items():
            features[f"vib_z_{key}"] = value

        # 温度特征
        temp_features = self.feature_extractor.extract_temperature_features(
            sensor_data.temperature, baseline_temp=25.0  # 基线温度需要从历史数据获取
        )
        features.update(temp_features)

        # 磁场特征
        magnetic_features = self.feature_extractor.extract_magnetic_features(
            sensor_data.magnetic_x, sensor_data.magnetic_y, sensor_data.magnetic_z
        )
        features.update(magnetic_features)

        return features

    def _fuse_diagnoses(self, ml_fault_type: FaultType,
                        ml_confidence: float,
                        expert_diagnoses: List[Dict]) -> Dict:
        """融合机器学习和专家系统的诊断结果"""
        if not expert_diagnoses:
            return {
                "fault_type": ml_fault_type,
                "confidence": ml_confidence,
                "source": "ml_only"
            }

        # 找出专家系统中置信度最高的诊断
        best_expert = max(expert_diagnoses, key=lambda x: x["score"])

        # 如果机器学习置信度很高，优先使用ML结果
        if ml_confidence > 0.8:
            return {
                "fault_type": ml_fault_type,
                "confidence": ml_confidence,
                "source": "ml_primary"
            }

        # 如果专家系统得分很高，且与ML结果一致
        if best_expert["score"] > 0.7 and best_expert["fault_type"] == ml_fault_type:
            combined_confidence = (ml_confidence + best_expert["score"]) / 2
            return {
                "fault_type": ml_fault_type,
                "confidence": combined_confidence,
                "source": "combined"
            }

        # 如果不一致，选择置信度更高的
        if best_expert["score"] > ml_confidence:
            return {
                "fault_type": best_expert["fault_type"],
                "confidence": best_expert["score"],
                "source": "expert_primary"
            }
        else:
            return {
                "fault_type": ml_fault_type,
                "confidence": ml_confidence,
                "source": "ml_primary"
            }

    def _determine_severity(self, features: Dict, fault_type: FaultType) -> str:
        """确定故障严重程度"""
        # 根据特征值和故障类型判断
        if fault_type == FaultType.UNBALANCE:
            vibration_level = features.get('vib_z_rms', 0)
            if vibration_level < 2.0:
                return "轻微"
            elif vibration_level < 4.0:
                return "中等"
            else:
                return "严重"

        elif fault_type in [FaultType.BEARING_INNER_RACE_WEAR,
                            FaultType.BEARING_OUTER_RACE_WEAR]:
            kurtosis = features.get('vib_z_kurtosis', 3)
            if kurtosis < 4:
                return "早期"
            elif kurtosis < 6:
                return "中等"
            else:
                return "严重"

        return "中等"

    def _locate_fault(self, fault_type: FaultType) -> str:
        """定位故障部位"""
        location_map = {
            FaultType.UNBALANCE: "转子",
            FaultType.MISALIGNMENT: "联轴器",
            FaultType.LOOSENESS: "轴承座或基础",
            FaultType.BEARING_INNER_RACE_WEAR: "轴承内圈",
            FaultType.BEARING_OUTER_RACE_WEAR: "轴承外圈",
            FaultType.BEARING_ROLLER_WEAR: "轴承滚动体",
            FaultType.BEARING_CAGE_WEAR: "轴承保持架",
            FaultType.BEARING_LUBRICATION: "轴承润滑系统",
            FaultType.ROTOR_BAR_LOOSE: "电机转子条",
            FaultType.OIL_WHIRL: "滑动轴承油膜"
        }
        return location_map.get(fault_type, "未知部位")

    def _generate_recommendation(self, diagnosis: Dict) -> str:
        """生成维修建议"""
        fault_type = diagnosis["fault_type"]
        severity = diagnosis.get("severity", "中等")

        recommendations = {
            FaultType.UNBALANCE: {
                "轻微": "加强监测，下次检修时检查平衡",
                "中等": "安排计划停机，进行动平衡校正",
                "严重": "立即停机，进行动平衡校正"
            },
            FaultType.MISALIGNMENT: {
                "轻微": "调整联轴器对中，加强监测",
                "中等": "停机进行激光对中校正",
                "严重": "立即停机，检查基础并进行精确对中"
            },
            FaultType.BEARING_INNER_RACE_WEAR: {
                "早期": "加强润滑，安排下次检修更换",
                "中等": "计划停机更换轴承",
                "严重": "立即停机更换轴承，检查轴颈"
            }
        }

        fault_rec = recommendations.get(fault_type, {})
        return fault_rec.get(severity, "请联系专家进行现场诊断")

    def _store_to_history(self, result: FaultDiagnosisResult):
        """存储诊断结果到历史数据库"""
        self.history_data.append(result)

        # 保持最近1000条记录
        if len(self.history_data) > 1000:
            self.history_data = self.history_data[-1000:]

    def train_model(self, training_data: pd.DataFrame, labels: pd.Series):
        """训练设备专属模型"""
        logging.info(f"开始训练设备 {self.device_id} 的专属模型")
        self.ml_model.train(training_data, labels)

        # 保存模型
        model_path = f"models/{self.device_id}_model.pkl"
        self.ml_model.save_model(model_path)
        self.model_trained = True
        logging.info(f"模型已保存到 {model_path}")


# ============= 实时监测服务 =============
class RealTimeMonitoringService:
    """
    实时监测服务 - 7*24小时设备看护
    对应文档中的"远程诊断功能"
    """

    def __init__(self, vpn_enabled: bool = False):
        self.vpn_enabled = vpn_enabled
        self.diagnosis_engines = {}  # device_id -> AIMFaultDiagnosisEngine
        self.alert_threshold = 0.7  # 报警阈值
        self.expert_team_available = vpn_enabled

    def register_device(self, device_id: str, device_type: DeviceType):
        """注册设备到监测系统"""
        engine = AIMFaultDiagnosisEngine(device_id, device_type)
        self.diagnosis_engines[device_id] = engine
        logging.info(f"设备 {device_id} 已注册到监测系统")

    def process_realtime_data(self, sensor_data: SensorData):
        """处理实时数据流"""
        device_id = sensor_data.device_id

        if device_id not in self.diagnosis_engines:
            logging.warning(f"设备 {device_id} 未注册，自动注册")
            # 这里需要根据设备ID获取设备类型，简化处理
            device_type = DeviceType.MOTOR
            self.register_device(device_id, device_type)

        engine = self.diagnosis_engines[device_id]

        # 进行故障诊断
        result = engine.process_sensor_data(sensor_data)

        # 检查是否需要报警
        if result.confidence > self.alert_threshold:
            self._trigger_alert(result)

            # 如果开通了VPN，可以请求专家支持
            if self.expert_team_available:
                self._request_expert_support(result)

        return result

    def _trigger_alert(self, result: FaultDiagnosisResult):
        """触发报警"""
        alert_message = (
            f"🚨 设备报警！\n"
            f"设备ID: {result.device_id}\n"
            f"故障类型: {result.fault_type.value}\n"
            f"严重程度: {result.severity}\n"
            f"置信度: {result.confidence:.2%}\n"
            f"故障部位: {result.location}\n"
            f"建议: {result.recommendation}\n"
            f"时间: {result.timestamp}"
        )

        # 发送报警（多种方式）
        self._send_alert_email(alert_message)
        self._send_alert_sms(alert_message)
        self._push_to_web_dashboard(result)

        logging.warning(alert_message)

    def _send_alert_email(self, message: str):
        """发送邮件报警"""
        # 实现邮件发送逻辑
        pass

    def _send_alert_sms(self, message: str):
        """发送短信报警"""
        # 实现短信发送逻辑
        pass

    def _push_to_web_dashboard(self, result: FaultDiagnosisResult):
        """推送到Web仪表板"""
        # 实现WebSocket推送
        pass

    def _request_expert_support(self, result: FaultDiagnosisResult):
        """请求专家支持"""
        if not self.vpn_enabled:
            return

        # 通过VPN连接远程专家系统
        expert_report = self._connect_to_expert_center(result)

        # 更新诊断结果
        logging.info(f"专家诊断结果: {expert_report}")

        # 可以发送更详细的诊断报告给现场人员

    def _connect_to_expert_center(self, result: FaultDiagnosisResult) -> Dict:
        """连接远程专家中心"""
        # 模拟专家中心响应
        return {
            "expert_confirm": True,
            "additional_findings": "建议检查基础螺栓紧固情况",
            "priority": "高",
            "estimated_remaining_life": "30天" if result.severity == "严重" else "90天"
        }


# ============= API接口 =============
class AIMDiagnosisAPI:
    """
    RESTful API接口 - 供其他系统调用
    对应文档中的"标准API接入接出"
    """

    def __init__(self):
        self.monitoring_service = RealTimeMonitoringService(vpn_enabled=False)

    def data_ingestion_endpoint(self, data: Dict):
        """数据接入端点"""
        # 解析数据
        sensor_data = self._parse_sensor_data(data)

        # 处理数据
        result = self.monitoring_service.process_realtime_data(sensor_data)

        # 返回结果
        return self._format_response(result)

    def diagnosis_endpoint(self, device_id: str, start_time: str, end_time: str):
        """历史诊断查询端点"""
        # 查询历史数据
        engine = self.monitoring_service.diagnosis_engines.get(device_id)
        if not engine:
            return {"error": "设备不存在"}

        # 筛选时间范围内的诊断结果
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(end_time)

        history = [
            r for r in engine.history_data
            if start_dt <= r.timestamp <= end_dt
        ]

        return {
            "device_id": device_id,
            "period": f"{start_time} 到 {end_time}",
            "diagnosis_count": len(history),
            "results": [
                {
                    "time": r.timestamp.isoformat(),
                    "fault_type": r.fault_type.value,
                    "severity": r.severity,
                    "confidence": r.confidence
                }
                for r in history
            ]
        }

    def _parse_sensor_data(self, data: Dict) -> SensorData:
        """解析传感器数据"""
        return SensorData(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            device_id=data["device_id"],
            vibration_x=np.array(data["vibration"]["x"]),
            vibration_y=np.array(data["vibration"]["y"]),
            vibration_z=np.array(data["vibration"]["z"]),
            temperature=data["temperature"],
            magnetic_x=data["magnetic"]["x"],
            magnetic_y=data["magnetic"]["y"],
            magnetic_z=data["magnetic"]["z"]
        )

    def _format_response(self, result: FaultDiagnosisResult) -> Dict:
        """格式化响应"""
        return {
            "diagnosis_id": str(hash(result)),
            "device_id": result.device_id,
            "timestamp": result.timestamp.isoformat(),
            "status": "alert" if result.confidence > 0.7 else "normal",
            "fault_type": result.fault_type.value,
            "severity": result.severity,
            "confidence": result.confidence,
            "location": result.location,
            "recommendation": result.recommendation,
            "features_summary": {
                k: round(v, 4)
                for k, v in result.features.items()
                if not isinstance(v, (list, np.ndarray))
            }
        }


# ============= 使用示例 =============
def main():
    """主函数 - 演示系统使用"""

    # 1. 初始化系统
    logging.basicConfig(level=logging.INFO)

    # 2. 创建监测服务 (如果开通了VPN)
    monitoring_service = RealTimeMonitoringService(vpn_enabled=True)

    # 3. 注册设备
    monitoring_service.register_device("Motor_001", DeviceType.MOTOR)
    monitoring_service.register_device("Pump_002", DeviceType.CENTRIFUGAL_PUMP)

    # 4. 模拟传感器数据
    def generate_mock_sensor_data(device_id: str, has_fault: bool = False):
        """生成模拟传感器数据"""
        n_samples = 25600
        time = np.arange(n_samples) / 51200

        if has_fault:
            # 模拟不平衡故障
            vibration_z = 0.5 * np.sin(2 * np.pi * 25 * time) + 0.1 * np.random.randn(n_samples)
        else:
            # 正常数据
            vibration_z = 0.1 * np.sin(2 * np.pi * 25 * time) + 0.01 * np.random.randn(n_samples)

        return SensorData(
            timestamp=datetime.now(),
            device_id=device_id,
            vibration_x=0.01 * np.random.randn(n_samples),
            vibration_y=0.01 * np.random.randn(n_samples),
            vibration_z=vibration_z,
            temperature=35.5,
            magnetic_x=10.2,
            magnetic_y=15.3,
            magnetic_z=20.1
        )

    # 5. 模拟数据流处理
    print("开始模拟实时监测...")

    # 正常数据
    normal_data = generate_mock_sensor_data("Motor_001", has_fault=False)
    result_normal = monitoring_service.process_realtime_data(normal_data)
    print(f"正常状态诊断: {result_normal.fault_type.value}, 置信度: {result_normal.confidence:.2%}")

    # 故障数据
    fault_data = generate_mock_sensor_data("Motor_001", has_fault=True)
    result_fault = monitoring_service.process_realtime_data(fault_data)
    print(f"故障状态诊断: {result_fault.fault_type.value}, 置信度: {result_fault.confidence:.2%}")

    # 6. API使用示例
    api = AIMDiagnosisAPI()

    # 模拟API调用
    mock_api_data = {
        "timestamp": datetime.now().isoformat(),
        "device_id": "Motor_001",
        "vibration": {
            "x": [0.1] * 1000,
            "y": [0.1] * 1000,
            "z": [0.5] * 1000
        },
        "temperature": 38.2,
        "magnetic": {
            "x": 10.5,
            "y": 15.2,
            "z": 20.3
        }
    }

    api_response = api.data_ingestion_endpoint(mock_api_data)
    print(f"API响应: {api_response['status']}")


if __name__ == "__main__":
    main()