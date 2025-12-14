def load_baseline_models():
    baseline_models = {
        "旋转类动设备": ["转子不平衡模型", "转子不对中模型", "松动或碰摩模型"],
        "离心泵": ["汽蚀模型"],
        "离心式风机": ["紊流模型"],
        "电气设备": ["转子条松动模型"],
        "滚动轴承": ["轴承保持架磨损模型", "轴承滚动体磨损模型",  # 这里加上了逗号
                    "轴承外圈磨损模型", "轴承内圈磨损模型", "轴承润滑不良模型"],
        "滑动轴承": ["油膜涡动模型"]
    }
    return baseline_models
def match_device_model(device_type, fault_features):
    models = load_baseline_models()
    target_models = models.get(device_type, [])
    return match_device_model(target_models, fault_features)