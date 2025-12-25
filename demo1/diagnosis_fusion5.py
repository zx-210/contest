import numpy as np
#诊断结果融合与输出
def fuse_results(ml_pred_proba: np.ndarray, expert_verify: bool, class_names: list) -> tuple:
    """融合机器学习与专家规则结果（最终诊断）"""
    ml_pred_idx = ml_pred_proba.argmax(axis=1)[0]
    confidence = ml_pred_proba[0][ml_pred_idx]
    # 专家验证通过则提升置信度，未通过则降低
    if expert_verify:
        confidence = min(confidence * 1.1, 1.0)
    else:
        confidence = max(confidence * 0.7, 0.0)
    # 判定故障等级（文档故障程度划分）
    fault_level = "轻微" if confidence < 0.7 else "明显" if confidence < 0.9 else "严重"
    return class_names[ml_pred_idx], confidence, fault_level

def generate_diagnosis_report(fault_type: str, confidence: float, fault_level: str, feature_dict: dict) -> str:
    """生成诊断报告（文档运维输出要求）"""
    report = f"""
    【设备故障诊断报告】
    故障类型：{fault_type}
    故障等级：{fault_level}
    诊断置信度：{confidence:.2f}
    关键特征：
      - 速度有效值：{feature_dict['velocity_rms']}mm/s
      - Z轴加速度峰值：{feature_dict['accel_peak_z']}m/s²
      - 磁场有效值：{feature_dict['mag_rms']}uT
    处理建议：{get_maintenance_suggestion(fault_type)}
    """
    return report

def get_maintenance_suggestion(fault_type: str) -> str:
    """故障维修建议（文档专家经验输出）"""
    suggestions = {
        "转子不平衡": "排查转子零件缺失、弯曲，检查底座松动",
        "转子不对中": "检查设备基础水平、联轴器对中情况",
        "轴承磨损": "检查轴承运行状态，查看异响及润滑情况",
        "正常": "设备运行正常，持续监测"
    }
    return suggestions.get(fault_type, "建议尽快停机检修")

# 核心调用示例
if __name__ == "__main__":
    # 模拟输入
    ml_pred_proba = np.array([[0.1, 0.75, 0.15]])  # 0-正常，1-轴承磨损，2-转子不平衡
    expert_verify_result = True
    class_names = ["正常", "轴承磨损", "转子不平衡"]
    feature_dict = {"velocity_rms": 6.13, "accel_peak_z": 41.29, "mag_rms": 56.74}
    # 融合诊断
    fault_type, confidence, fault_level = fuse_results(ml_pred_proba, expert_verify_result, class_names)
    # 生成报告
    report = generate_diagnosis_report(fault_type, confidence, fault_level, feature_dict)
    print(report)