import numpy as np
import time
from datetime import datetime

# 核心参数（匹配300Hz采集/0.25s唤醒/异常检测核心规格）
CORE_CONFIG = {
    "z_sample_rate": 51200,    # 基础采样率（保障300Hz有效采集）
    "wakeup_interval": 0.25    # 0.25s高频唤醒（匹配硬件触发逻辑）
}

# 核心数据载体（仅保留故障诊断必需参量）
class CoreMonitor:
    def __init__(self):
        self.accel_z = []        # Z轴振动数据（故障分析核心数据源）
        self.velocity_rms = 0.0  # 速度有效值（异常检测核心指标）
        self.is_fault = False    # 异常检测结果标记

# 1. 高频采集（匹配300Hz采集速率）
def collect_core_data(monitor: CoreMonitor):
    monitor.accel_z = np.random.uniform(-50, 50, CORE_CONFIG["z_sample_points"]).tolist()

# 2. 内嵌智能异常检测算法核心逻辑
def calc_core_feature(monitor: CoreMonitor):
    accel_z = np.array(monitor.accel_z)
    velocity = np.cumsum(accel_z) / CORE_CONFIG["z_sample_rate"]
    monitor.velocity_rms = round(np.sqrt(np.mean(np.square(velocity))) * 1000, 2)
def diagnose_fault(monitor: CoreMonitor):
    monitor.is_fault = monitor.velocity_rms > CORE_CONFIG["velocity_threshold"]

# 3. 模拟ZigBee网关上传（核心数据上传）
def upload_core_data(monitor: CoreMonitor):
    print(f"【{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}】")
    print(f"速度有效值：{monitor.velocity_rms}mm/s | 故障状态：{'故障' if monitor.is_fault else '正常'}")
    print("-"*50)

# 主流程（0.25s高频唤醒闭环）
def core_main():
    monitor = CoreMonitor()
    print("关键设备监测核心系统启动...\n")
    while True:
        collect_core_data(monitor)   # 高频采集
        calc_core_feature(monitor)   # 特征计算
        diagnose_fault(monitor)      # 内嵌异常检测
        upload_core_data(monitor)    # ZigBee网关上传
        time.sleep(CORE_CONFIG["wakeup_interval"])  # 0.25s高频唤醒

if __name__ == "__main__":
    core_main()