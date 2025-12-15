from datetime import datetime

class EquipmentStatus:
    def __init__(self, equipment_id: str, equipment_name: str, temperature: float, 
                 vibration: float, status: str, health_score: float, timestamp: datetime):
        self.equipment_id = equipment_id
        self.equipment_name = equipment_name
        self.temperature = temperature
        self.vibration = vibration
        self.status = status
        self.health_score = health_score
        self.timestamp = timestamp