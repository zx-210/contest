import mysql.connector
from mysql.connector import Error
from dataclasses import dataclass
from datetime import datetime


# 定义传感器数据类（对应原Java的SensorData）
@dataclass
class SensorData:
    id: int
    sensorType: str
    value: float
    timestamp: datetime
    isCritical: bool
    outlierScore: int = 0


# 数据库配置（保持原配置）
DB_CONFIG = {
    "host": "localhost",
    "database": "sensor_db",
    "user": "admin",
    "password": "secure_password123"
}


def fetch_raw_data() -> list[SensorData]:
    """从数据库获取原始传感器数据（对应原fetchRawData方法）"""
    raw_data = []
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM sensor_data")
            records = cursor.fetchall()

            for record in records:
                sensor_data = SensorData(
                    id=record["id"],
                    sensorType=record["sensor_type"],
                    value=record["value"],
                    timestamp=record["timestamp"],
                    isCritical=record["is_critical"]
                )
                raw_data.append(sensor_data)

    except Error as e:
        print(f"Database error: {e}")
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
    return raw_data


if __name__ == "__main__":
    # 测试数据采集
    data = fetch_raw_data()
    print(f"采集到 {len(data)} 条原始数据")