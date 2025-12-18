# 初始化智能磁振温一体传感器采集网络
class SensorNetwork:
    def __init__(self, device_ids):
        self.device_ids = device_ids  # 支持多型号设备适配
        self.sensors = {"vibration": None, "temperature": None, "magnetic": None}

    # 7x24小时实时采集关键生命体征
    def real_time_collect(self, sample_rate=1000):
        data = {}
        for dev_id in self.device_ids:
            # 模拟多设备数据采集，包含抗干扰滤波
            vibration_data = self._filter_interference(self._collect_vibration(dev_id, sample_rate))
            temperature_data = self._collect_temperature(dev_id)
            magnetic_data = self._collect_magnetic(dev_id)
            data[dev_id] = {
                "vibration": vibration_data,
                "temperature": temperature_data,
                "magnetic": magnetic_data,
                "timestamp": self._get_current_time()
            }
        return data



# 初始化目标设备采集网络（演示用3台不同型号设备）
sensor_net = SensorNetwork(device_ids=["DEV-001", "DEV-002", "DEV-003"])