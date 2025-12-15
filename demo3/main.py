from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt5.QtCore import QTimer, Qt
from RealTimeDataEngine import RealTimeDataEngine
from VisualizationEngine import VisualizationEngine
from AlertSystem import AlertSystem
import sys


class IndustrialMonitoringSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 初始化数据引擎
        self.data_engine = RealTimeDataEngine()
        
        # 初始化告警系统
        self.alert_system = AlertSystem()
        
        # 设置窗口属性
        self.setWindowTitle("工业设备监控大屏系统")
        self.setGeometry(100, 100, 1200, 700)
        
        # 创建主中央部件
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        # 初始化可视化引擎
        self.visualization_engine = VisualizationEngine(central_widget)
        
        # 设置定时刷新
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_dashboard)
        self.timer.start(2000)  # 每2秒更新一次
        
    def update_dashboard(self):
        # 更新所有设备状态
        self.data_engine.update_all_status()
        
        # 更新可视化界面
        self.visualization_engine.update_visualizations(self.data_engine.equipment_status_map)
        
        # 检查并触发告警
        for status in self.data_engine.equipment_status_map.values():
            self.alert_system.check_and_alert(status)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f0f0f0;
        }
        QWidget {
            font-family: Arial, sans-serif;
        }
    """)
    
    # 创建主窗口
    window = IndustrialMonitoringSystem()
    window.show()
    
    # 运行应用
    sys.exit(app.exec_())
