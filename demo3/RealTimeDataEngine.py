import random
from datetime import datetime
from threading import Lock
from EquipmentStatus import EquipmentStatus


class RealTimeDataEngine:
    def __init__(self):
        self.equipment_status_map = {}
        self.lock = Lock()
        self.random = random.Random()
        self.equipment_names = ["离心泵-001", "压缩机-002", "涡轮机-003", "传送带-004"]
        self.status_types = ["正常", "警告", "故障"]
        
        # 初始化设备状态
        for i in range(1, 5):
            equipment_id = f"设备-{i:03d}"
            self.equipment_status_map[equipment_id] = self.generate_status(equipment_id)

    def generate_status(self, equipment_id: str) -> EquipmentStatus:
        temperature = 20 + self.random.random() * 25
        vibration = 0.5 + self.random.random() * 5
        health_score = 50 + self.random.random() * 50
        status = self.status_types[self.random.randint(0, len(self.status_types) - 1)]
        
        # 获取设备名称索引
        try:
            index = int(equipment_id.split("-")[1]) % len(self.equipment_names)
            equipment_name = self.equipment_names[index]
        except (IndexError, ValueError):
            equipment_name = "未知设备"
        
        return EquipmentStatus(
            equipment_id,
            equipment_name,
            temperature,
            vibration,
            status,
            health_score,
            datetime.now()
        )

    def get_status(self, equipment_id: str) -> EquipmentStatus:
        with self.lock:
            if equipment_id not in self.equipment_status_map:
                self.equipment_status_map[equipment_id] = self.generate_status(equipment_id)
            return self.equipment_status_map[equipment_id]

    def update_all_status(self):
        with self.lock:
            for equipment_id in list(self.equipment_status_map.keys()):
                self.equipment_status_map[equipment_id] = self.generate_status(equipment_id)