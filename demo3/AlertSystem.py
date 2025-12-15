from threading import Lock
from EquipmentStatus import EquipmentStatus


class AlertSystem:
    def __init__(self):
        self.alert_map = {}
        self.lock = Lock()
        
    def check_and_alert(self, status: EquipmentStatus):
        with self.lock:
            if status.status == "故障":
                self.alert_map[status.equipment_id] = status.timestamp
                self.trigger_alert(status)
            elif status.status == "警告":
                self.alert_map[status.equipment_id] = status.timestamp
            elif status.equipment_id in self.alert_map:
                # 如果设备恢复正常，从告警列表中移除
                del self.alert_map[status.equipment_id]
        
    def trigger_alert(self, status: EquipmentStatus):
        # 实现告警通知逻辑（短信/邮件/声光报警）
        print(f"【告警】设备 {status.equipment_name} (ID: {status.equipment_id}) 发生故障！")
        print(f"  温度: {status.temperature:.2f}°C")
        print(f"  震动: {status.vibration:.2f} mm/s")
        print(f"  时间: {status.timestamp}")
