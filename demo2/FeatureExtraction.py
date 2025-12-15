from DataCollection import SensorData, fetch_raw_data
from DataCleaning import clean_data
from collections import defaultdict
from typing import List, Dict


def generate_cleaning_report(cleaned_data: List[SensorData]) -> None:
    """生成数据清洗报告（完全对应原generateCleaningReport方法）"""
    # 按传感器类型统计特征
    stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {
        "count": 0,
        "sum": 0.0,
        "max": float('-inf'),
        "min": float('inf')
    })

    # 统计基础特征
    for data in cleaned_data:
        sensor_type = data.sensorType
        stats[sensor_type]["count"] += 1
        stats[sensor_type]["sum"] += data.value
        stats[sensor_type]["max"] = max(stats[sensor_type]["max"], data.value)
        stats[sensor_type]["min"] = min(stats[sensor_type]["min"], data.value)

    # 生成报告（保持原输出格式）
    print("\n=== 数据清洗报告 ===")
    for sensor, stat in stats.items():
        avg = stat["sum"] / stat["count"] if stat["count"] > 0 else 0
        print(
            f"{sensor}: 平均值={avg:.2f}, 最大值={stat['max']:.2f}, "
            f"最小值={stat['min']:.2f}, 数据量={int(stat['count'])}"
        )

    # 统计异常值
    outlier_count = sum(1 for data in cleaned_data if data.outlierScore > 0)
    total_count = len(cleaned_data)
    outlier_percent = (outlier_count / total_count * 100) if total_count > 0 else 0
    print(
        f"\n检测到异常值记录: {outlier_count} 条 "
        f"(占总数据的 {outlier_percent:.2f}%)"
    )


if __name__ == "__main__":
    # 测试特征提取/报告生成
    raw_data = fetch_raw_data()
    cleaned_data = clean_data(raw_data)
    generate_cleaning_report(cleaned_data)