import numpy as np
#一机一模型自适应优化
def adapt_model(device_id: str, base_model, new_features: np.ndarray, new_labels: np.ndarray) -> dict:
    """适配单设备模型（文档“一机一模型”核心）"""
    # 增量训练更新模型（模拟自适应学习）
    base_model.fit(new_features, new_labels)
    return {
        "device_id": device_id,
        "adapted_model": base_model,
        "update_time": np.datetime64("now"),
        "adapt_times": 1
    }

def calc_device_baseline(features_history: np.ndarray) -> dict:
    """计算设备健康基线（个性化基准）"""
    return {
        "velocity_rms_mean": np.mean(features_history[:, 1]),
        "velocity_rms_std": np.std(features_history[:, 1]),
        "accel_peak_z_mean": np.mean(features_history[:, 0])
    }

# 核心调用示例
if __name__ == "__main__":
    device_id = "JD202504250116"
    # 模拟历史健康数据
    history_features = np.array([[32.1, 4.2, 55.3], [31.8, 4.5, 54.9], [32.5, 4.3, 55.1]])
    # 模拟新采集数据（轻微故障）
    new_features = np.array([[38.5, 5.8, 56.7]])
    new_labels = np.array([1])
    # 基于随机森林的自适应优化
    from base_ml_models import train_random_forest
    base_model = train_random_forest(history_features, np.zeros(3))
    adapted_model = adapt_model(device_id, base_model, new_features, new_labels)
    baseline = calc_device_baseline(history_features)
    print(f"设备{device_id}自适应模型更新完成，健康基线：{baseline}")