import math
# 构建故障知识图谱与逻辑校验系统
class FaultKnowledgeGraph:
    def __init__(self):
        # 存储结构化专家知识：故障类型→特征参数→判定规则
        self.fault_rules = {
            "转子不平衡": {
                "vibration_spectrum": lambda x: x["1x_freq_amplitude"] / x["2x_freq_amplitude"] > 2.5,
                "phase_char": lambda x: x["phase_diff"]},
            "轴承内圈故障": {
                "feature_freq": lambda x: abs(x["calculated_freq"] - self._calc_inner_ring_freq(x)),
                "vibration_rms": lambda x: x["rms_value"] > 0.8
            },
            # 更多故障类型的专家规则...
        }

    # 计算轴承内圈故障特征频率（专家公式）
    def _calc_inner_ring_freq(self, params):
        # 公式：fi = 0.5 * n * (D - d * cosα) / D * fr
        return 0.5 * params["ball_num"] * (
                    params["pitch_dia"] - params["ball_dia"] * math.cos(params["contact_angle"])) / params[
            "pitch_dia"] * params["rotor_freq"]

    # 逻辑校验：验证机器学习结果的物理合理性
    def verify_ml_result(self, ml_pred, device_params, real_data):
        top_fault = ml_pred["top1_fault"]
        if top_fault not in self.fault_rules:
            return False, "无对应专家规则"
        # 逐项校验该故障的所有专家规则
        rules = self.fault_rules[top_fault]
        for rule_name, rule_func in rules.items():
            if not rule_func(self._extract_rule_params(real_data, device_params)):
                return False, f"规则[{rule_name}]不满足"
        return True, "物理机理校验通过"


# 初始化专家知识图谱
knowledge_graph = FaultKnowledgeGraph()