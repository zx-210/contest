from DataCollection import SensorData, fetch_raw_data
import statistics
from typing import List


def calculate_moving_average(recent_data: List[SensorData], current: SensorData) -> float:
    """移动平均插值算法（完全对应原calculateMovingAverage方法）"""
    # 筛选最近5个同类传感器数据
    values = [
                 data.value for data in recent_data
                 if data.sensorType == current.sensorType
             ][-5:]  # 取最近5个

    if len(values) < 1:
        return current.value

    # 排序后取中间3个值（对应原sorted + limit(3)）
    values_sorted = sorted(values)
    middle_values = values_sorted[1:-1] if len(values_sorted) >= 3 else values_sorted

    return statistics.mean(middle_values) if middle_values else current.value


def clean_data(raw_data: List[SensorData]) -> List[SensorData]:
    """核心清洗算法（完全对应原cleanData方法）"""
    # 第一阶段：基础清洗 - 过滤负值
    stage1 = [data for data in raw_data if data.value > 0]

    # 第二阶段：传感器特定清洗（保持原阈值配置）
    sensor_ranges = {
        "Temperature": 40.0,  # 最大合理温度
        "Vibration": 6.0  # 最大合理震动值
    }
    stage2 = []

    for data in stage1:
        max_range = sensor_ranges.get(data.sensorType, float('inf'))

        # 异常值检测与评分（完全保留原逻辑）
        if data.value > max_range:
            data.outlierScore += 5
            continue  # 跳过极端异常值
        elif data.value > max_range * 0.9:
            data.outlierScore += 2  # 可疑值标记

        # 数据插值处理
        if data.outlierScore > 0:
            data.value = calculate_moving_average(stage2, data)

        stage2.append(data)

    # 第三阶段：时间序列去重（保持原去重逻辑）
    seen = set()
    cleaned_data = []
    for data in stage2:
        key = f"{data.timestamp}:{data.sensorType}"
        if key not in seen:
            seen.add(key)
            cleaned_data.append(data)

    return cleaned_data


if __name__ == "__main__":
    # 测试数据清洗
    raw_data = fetch_raw_data()
    cleaned_data = clean_data(raw_data)
    print(f"清洗后剩余 {len(cleaned_data)} 条数据")