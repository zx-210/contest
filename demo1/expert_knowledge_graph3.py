import numpy as np
#专家知识图谱模型
FAULT_RULES = {
    "转子不平衡": {"feature": "velocity_rms", "threshold": 6.0, "freq_feature": "1倍频能量占比>50%"},
    "转子不对中": {"feature": "velocity_rms", "threshold": 10.0, "freq_feature": "2倍频能量占比>80%"},
    "轴承磨损": {"feature": "accel_peak_z", "threshold": 30.0, "freq_feature": "包络谱1-4倍频能量大"}
}

def expert_verify(fault_pred: str, feature_dict: dict) -> tuple:
    """用专家规则验证机器学习结果（双核驱动核心）"""
    if fault_pred not in FAULT_RULES:
        return False, "无匹配专家规则"
    rule = FAULT_RULES[fault_pred]
    # 数值特征验证
    if feature_dict[rule["feature"]] < rule["threshold"]:
        return False, f"{rule['feature']}未达阈值"
    # 频谱特征验证（模拟）
    print(f"频谱特征验证：{rule['freq_feature']}")
    return True, "专家规则验证通过"

# 核心调用示例
if __name__ == "__main__":
    # 模拟特征字典
    feature_dict = {"velocity_rms": 10.87, "accel_peak_z": 28.5, "mag_rms": 57.3}
    ml_pred = "转子不对中"  # 机器学习预测结果
    verify_result, msg = expert_verify(ml_pred, feature_dict)
    print(f"专家验证结果：{'通过' if verify_result else '未通过'}，原因：{msg}")