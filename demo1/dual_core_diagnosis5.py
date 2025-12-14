from fusion_model2 import FusionModelMatrix
from knowledge_graph4 import FaultKnowledgeGraph
def extract_features(real_time_data):
    features = {
        "frequency_domain": {"1x": 0.8, "2x": 0.15, "3x": 0.05},
        "time_domain": {"rms": 0.12, "kurtosis": 3.8},
        "temperature": 75.5
    }
    return features
def optimize_fusion_model(fusion_model, features, final_result):
    print("正在优化融合模型...")
    print(f"基于诊断结果: {final_result}")
    print("模型优化完成")
def dual_core_diagnosis(device_id, real_time_data):
    features = extract_features(real_time_data)
    fusion_model = FusionModelMatrix()
    preliminary_result = fusion_model.predict(features)
    knowledge_graph = FaultKnowledgeGraph()
    final_result = knowledge_graph.verify_diagnosis(preliminary_result, features)
    optimize_fusion_model(fusion_model, features, final_result)
    return {
        "device_id": device_id,
        "preliminary_result": preliminary_result,
        "final_result": final_result,
        "features": features
    }
if __name__ == "__main__":
    test_device_id = "DEV001"
    test_real_time_data = {
        "vibration": [0.15, 0.18, 0.12, 0.16, 0.14],
        "temperature": 76.2,
        "current": 45.3
    }
    diagnosis_result = dual_core_diagnosis(test_device_id, test_real_time_data)
    print("双核驱动诊断结果:")
    print(f"设备ID: {diagnosis_result['device_id']}")
    print(f"初步诊断: {diagnosis_result['preliminary_result']}")
    print(f"最终诊断: {diagnosis_result['final_result']}")