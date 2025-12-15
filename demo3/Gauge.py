from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont
from PyQt5.QtCore import Qt, QRectF


class Gauge(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0.0
        self.min_value = 0.0
        self.max_value = 100.0
        self.title = "系统健康指数"
        self.setMinimumSize(200, 200)
        
    def set_value(self, value):
        if value < self.min_value:
            self.value = self.min_value
        elif value > self.max_value:
            self.value = self.max_value
        else:
            self.value = value
        self.update()
        
    def set_min_value(self, min_value):
        self.min_value = min_value
        if self.value < self.min_value:
            self.value = self.min_value
        self.update()
        
    def set_max_value(self, max_value):
        self.max_value = max_value
        if self.value > self.max_value:
            self.value = self.max_value
        self.update()
        
    def set_title(self, title):
        self.title = title
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        center_x = width // 2
        center_y = height // 2
        radius = min(center_x, center_y) - 20
        
        # 绘制背景
        background_brush = QBrush(QColor(44, 62, 80))
        painter.setBrush(background_brush)
        painter.drawRect(0, 0, width, height)
        
        # 绘制刻度盘
        dial_rect = QRectF(center_x - radius, center_y - radius, radius * 2, radius * 2)
        painter.setPen(QPen(QColor(52, 152, 219), 5))
        painter.drawArc(dial_rect, 45 * 16, 270 * 16)
        
        # 绘制刻度线
        painter.save()
        painter.translate(center_x, center_y)
        
        for i in range(13):  # 13条刻度线
            angle = 45 + i * (270 / 12)
            painter.rotate(angle)
            
            if i % 3 == 0:  # 主要刻度线
                painter.setPen(QPen(QColor(236, 240, 241), 3))
                painter.drawLine(0, -radius + 10, 0, -radius + 25)
            else:  # 次要刻度线
                painter.setPen(QPen(QColor(236, 240, 241), 2))
                painter.drawLine(0, -radius + 10, 0, -radius + 20)
            
            painter.rotate(-angle)
        
        painter.restore()
        
        # 绘制指针
        value_angle = 45 + (self.value / (self.max_value - self.min_value)) * 270
        pointer_length = radius - 30
        
        painter.save()
        painter.translate(center_x, center_y)
        painter.rotate(value_angle)
        
        pointer_pen = QPen(QColor(231, 76, 60), 5)
        painter.setPen(pointer_pen)
        painter.drawLine(0, 0, 0, -pointer_length)
        
        # 绘制指针中心
        center_brush = QBrush(QColor(231, 76, 60))
        painter.setBrush(center_brush)
        painter.drawEllipse(-5, -5, 10, 10)
        
        painter.restore()
        
        # 绘制标题
        title_font = QFont("Arial", 12, QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(236, 240, 241))
        painter.drawText(QRectF(0, height - 30, width, 20), Qt.AlignCenter, self.title)
        
        # 绘制数值
        value_font = QFont("Arial", 14, QFont.Bold)
        painter.setFont(value_font)
        painter.setPen(QColor(236, 240, 241))
        value_text = f"{self.value:.1f}%"
        painter.drawText(QRectF(0, center_y - 10, width, 20), Qt.AlignCenter, value_text)