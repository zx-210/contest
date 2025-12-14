  // 告警管理系统
    static class AlertSystem {
        private final ConcurrentHashMap<String, LocalDateTime> alertMap = new ConcurrentHashMap<>();
        
        public void checkAndAlert(EquipmentStatus status) {
            if ("故障".equals(status.status)) {
                alertMap.put(status.equipmentId, status.timestamp);
                triggerAlert(status);
            } else if ("警告".equals(status.status)) {
                alertMap.put(status.equipmentId, status.timestamp);
            }
        }
        
        private void triggerAlert(EquipmentStatus status) {
            // 实现告警通知逻辑（短信/邮件/声光报警）
        }
    }

    // 主应用类
    private final RealTimeDataEngine dataEngine = new RealTimeDataEngine();
    private final VisualizationEngine visualizationEngine;
    private final AlertSystem alertSystem = new AlertSystem();
    
    @Override
    public void start(Stage primaryStage) {
        primaryStage.setTitle("工业设备监控大屏系统");
        
        // 创建主界面
        StackPane root = new StackPane();
        Scene scene = new Scene(root, 1200, 700);
        
        // 初始化可视化引擎
        visualizationEngine = new VisualizationEngine(root);
        
        // 设置样式
        scene.getStylesheets().add("""
            .chart {
                -fx-background-color: #ecf0f1;
                -fx-background-radius: 5;
                -fx-padding: 10;
            }
            .gauge {
                -fx-background-color: #2c3e50;
            }
            """);
        
        primaryStage.setScene(scene);
        primaryStage.show();
        
        // 设置定时刷新
        AnimationTimer timer = new AnimationTimer() {
            private long lastUpdate = 0;
            
            @Override
            public void handle(long now) {
                if (now - lastUpdate >= 2_000_000_000) { // 每2秒更新一次
                    lastUpdate = now;
                    updateDashboard();
                }
            }
        };
        timer.start();
    }
    
    private void updateDashboard() {
        // 更新所有设备状态
        dataEngine.updateAllStatus();
        
        // 更新可视化界面
        visualizationEngine.updateVisualizations(dataEngine.equipmentStatusMap);
        
        // 检查并触发告警
        dataEngine.equipmentStatusMap.values().forEach(alertSystem::checkAndAlert);
    }

    public static void main(String[] args) {
        launch(args);
    }
}

// 自定义仪表组件
class Gauge extends Pane {
    private double value;
    private double minValue;
    private double maxValue;
    
    // 仪表组件实现
    // 包含绘制刻度、指针、标签等功能
}