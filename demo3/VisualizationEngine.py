from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtChart import QChart, QChartView, QBarSet, QBarSeries, QCategoryAxis, QValueAxis, QPieSeries, QPieSlice
from PyQt5.QtChart import QLineSeries, QSplineSeries
from PyQt5.QtGui import QColor, QFont, QPainter
from PyQt5.QtCore import Qt
from Gauge import Gauge


class VisualizationEngine:
    def __init__(self, parent):
        # 创建主布局
        main_layout = QHBoxLayout(parent)
        
        # 创建左侧垂直布局
        left_layout = QVBoxLayout()
        
        # 创建右侧垂直布局
        right_layout = QVBoxLayout()
        
        # 创建健康评分图表
        self.health_chart = self.create_health_chart()
        left_layout.addWidget(self.health_chart)
        
        # 创建状态分布图表
        self.status_chart = self.create_status_chart()
        left_layout.addWidget(self.status_chart)
        
        # 创建趋势图表
        self.trend_chart = self.create_trend_chart()
        right_layout.addWidget(self.trend_chart)
        
        # 创建健康仪表盘
        self.health_gauge = self.create_health_gauge()
        right_layout.addWidget(self.health_gauge)
        
        # 将左右布局添加到主布局
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 1)
        
    def create_health_chart(self):
        # 创建柱状图
        chart = QChart()
        chart.setTitle("设备健康评分")
        chart.setTitleFont(QFont("Arial", 14, QFont.Bold))
        chart.setAnimationOptions(QChart.SeriesAnimations)
        
        # 创建X轴
        x_axis = QCategoryAxis()
        x_axis.setTitleText("设备名称")
        x_axis.setTitleFont(QFont("Arial", 10, QFont.Bold))
        
        # 创建Y轴
        y_axis = QValueAxis()
        y_axis.setTitleText("健康评分")
        y_axis.setTitleFont(QFont("Arial", 10, QFont.Bold))
        y_axis.setRange(0, 100)
        y_axis.setTickCount(11)
        
        # 添加轴
        chart.addAxis(x_axis, Qt.AlignBottom)
        chart.addAxis(y_axis, Qt.AlignLeft)
        
        # 创建图表视图
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        chart_view.setStyleSheet("background-color: #ecf0f1; border-radius: 5px; padding: 10px;")
        
        return chart_view
        
    def create_status_chart(self):
        # 创建饼图
        chart = QChart()
        chart.setTitle("设备状态分布")
        chart.setTitleFont(QFont("Arial", 14, QFont.Bold))
        chart.setAnimationOptions(QChart.SeriesAnimations)
        
        # 创建图表视图
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        chart_view.setStyleSheet("background-color: #ecf0f1; border-radius: 5px; padding: 10px;")
        
        return chart_view
        
    def create_trend_chart(self):
        # 创建折线图
        chart = QChart()
        chart.setTitle("温度/震动趋势")
        chart.setTitleFont(QFont("Arial", 14, QFont.Bold))
        chart.setAnimationOptions(QChart.SeriesAnimations)
        
        # 创建X轴
        x_axis = QValueAxis()
        x_axis.setTitleText("时间")
        x_axis.setTitleFont(QFont("Arial", 10, QFont.Bold))
        x_axis.setRange(0, 100)
        x_axis.setTickCount(11)
        
        # 创建Y轴
        y_axis = QValueAxis()
        y_axis.setTitleText("数值")
        y_axis.setTitleFont(QFont("Arial", 10, QFont.Bold))
        y_axis.setRange(0, 100)
        y_axis.setTickCount(6)
        
        # 添加轴
        chart.addAxis(x_axis, Qt.AlignBottom)
        chart.addAxis(y_axis, Qt.AlignLeft)
        
        # 创建图表视图
        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.Antialiasing)
        chart_view.setStyleSheet("background-color: #ecf0f1; border-radius: 5px; padding: 10px;")
        
        return chart_view
        
    def create_health_gauge(self):
        # 创建健康仪表盘
        gauge = Gauge()
        gauge.set_title("系统健康指数")
        gauge.set_min_value(0)
        gauge.set_max_value(100)
        gauge.setStyleSheet("background-color: #2c3e50; border-radius: 5px; padding: 10px;")
        
        return gauge
        
    def update_visualizations(self, status_map):
        # 更新健康评分图表
        self.update_health_chart(status_map)
        
        # 更新状态分布图表
        self.update_status_chart(status_map)
        
        # 更新趋势图表
        self.update_trend_chart(status_map)
        
        # 更新仪表盘
        self.update_health_gauge(status_map)
        
    def update_health_chart(self, status_map):
        chart = self.health_chart.chart()
        
        # 清除现有系列
        chart.removeAllSeries()
        
        # 创建新的柱状系列
        series = QBarSeries()
        bar_set = QBarSet("健康评分")
        
        # 创建新的分类轴
        new_x_axis = QCategoryAxis()
        new_x_axis.setTitleText("设备名称")
        
        # 添加数据
        index = 0
        for equipment_id, status in status_map.items():
            bar_set.append(status.health_score)
            new_x_axis.append(status.equipment_name, index + 1.0)  # 使用索引作为分类的结束值
            index += 1
        
        # 移除旧的X轴
        old_x_axis = chart.axes(Qt.Horizontal)[0]
        chart.removeAxis(old_x_axis)
        
        # 添加新的X轴
        chart.addAxis(new_x_axis, Qt.AlignBottom)
        
        # 获取或创建Y轴
        y_axes = chart.axes(Qt.Vertical)
        if len(y_axes) > 0:
            y_axis = y_axes[0]
        else:
            y_axis = QValueAxis()
            y_axis.setTitleText("健康评分")
            y_axis.setRange(0, 100)
            chart.addAxis(y_axis, Qt.AlignLeft)
        
        # 添加系列到图表
        series.append(bar_set)
        chart.addSeries(series)
        
        # 连接系列和轴
        series.attachAxis(new_x_axis)
        series.attachAxis(y_axis)
        
    def update_status_chart(self, status_map):
        chart = self.status_chart.chart()
        
        # 清除现有系列
        chart.removeAllSeries()
        
        # 创建饼图系列
        series = QPieSeries()
        
        # 统计状态分布
        status_counts = {}
        for status in status_map.values():
            if status.status in status_counts:
                status_counts[status.status] += 1
            else:
                status_counts[status.status] = 1
        
        # 设置颜色映射
        color_map = {
            "正常": QColor(0, 255, 0),
            "警告": QColor(255, 255, 0),
            "故障": QColor(255, 0, 0)
        }
        
        # 添加数据到饼图
        for status, count in status_counts.items():
            slice = series.append(status, count)
            slice.setLabel(f"{status} ({count})")  # 设置标签
            slice.setBrush(color_map.get(status, QColor(128, 128, 128)))
        
        # 添加系列到图表
        chart.addSeries(series)
        
    def update_trend_chart(self, status_map):
        chart = self.trend_chart.chart()
        
        # 清除现有系列
        chart.removeAllSeries()
        
        # 创建温度和震动系列
        temp_series = QLineSeries()
        temp_series.setName("温度")
        temp_series.setColor(QColor(255, 0, 0))
        
        vibration_series = QLineSeries()
        vibration_series.setName("震动")
        vibration_series.setColor(QColor(0, 0, 255))
        
        # 添加数据
        for i, (equipment_id, status) in enumerate(status_map.items()):
            temp_series.append(i * 10, status.temperature)
            vibration_series.append(i * 10, status.vibration)
        
        # 添加系列到图表
        chart.addSeries(temp_series)
        chart.addSeries(vibration_series)
        
        # 附加轴
        temp_series.attachAxis(chart.axes(Qt.Horizontal)[0])
        temp_series.attachAxis(chart.axes(Qt.Vertical)[0])
        vibration_series.attachAxis(chart.axes(Qt.Horizontal)[0])
        vibration_series.attachAxis(chart.axes(Qt.Vertical)[0])
        
        # 更新X轴范围
        x_axis = chart.axes(Qt.Horizontal)[0]
        x_axis.setRange(0, len(status_map) * 10)
        
    def update_health_gauge(self, status_map):
        # 计算平均健康评分
        if status_map:
            avg_health_score = sum(status.health_score for status in status_map.values()) / len(status_map)
            self.health_gauge.set_value(avg_health_score)
