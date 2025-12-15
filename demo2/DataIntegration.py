import mysql.connector
from mysql.connector import Error
from DataCollection import SensorData, DB_CONFIG
from DataCleaning import clean_data, fetch_raw_data
from typing import List


def save_cleaned_data(cleaned_data: List[SensorData]) -> None:
    """保存清洗后的数据到数据库（对应原saveCleanedData方法）"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor()
            # 批量插入清洗后的数据（保持原SQL语句）
            insert_query = """
                INSERT INTO cleaned_sensor_data 
                (id, sensor_type, value, timestamp, is_critical, outlier_score) 
                VALUES (%s, %s, %s, %s, %s, %s)
            """

            # 准备批量数据
            batch_data = [
                (
                    data.id,
                    data.sensorType,
                    data.value,
                    data.timestamp,
                    data.isCritical,
                    data.outlierScore
                ) for data in cleaned_data
            ]

            cursor.executemany(insert_query, batch_data)
            connection.commit()
            print("Cleared data saved to cleaned_sensor_data table")

    except Error as e:
        print(f"Database save error: {e}")
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()


if __name__ == "__main__":
    # 测试数据集成（采集→清洗→保存）
    raw_data = fetch_raw_data()
    cleaned_data = clean_data(raw_data)
    save_cleaned_data(cleaned_data)