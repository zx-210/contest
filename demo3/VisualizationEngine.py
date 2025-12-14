 // 可视化引擎
    static class VisualizationEngine {
        private final BarChart<String, Number> healthChart;
        private final PieChart statusChart;
        private final LineChart<Number, Number> trendChart;
        private final Gauge healthGauge;

        public VisualizationEngine(StackPane root) {
            // 初始化仪表盘组件
            healthChart = createHealthChart();
            statusChart = createStatusChart();
            trendChart = createTrendChart();
            healthGauge = createHealthGauge();
            
            // 布局仪表盘
            root.getChildren().addAll(healthChart, statusChart, trendChart, healthGauge);
        }

        private BarChart<String, Number> createHealthChart() {
            CategoryAxis xAxis = new CategoryAxis();
            NumberAxis yAxis = new NumberAxis(0, 100, 10);
            BarChart<String, Number> chart = new BarChart<>(xAxis, yAxis);
            chart.setTitle("设备健康评分");
            return chart;
        }

        private PieChart createStatusChart() {
            PieChart chart = new PieChart();
            chart.setTitle("设备状态分布");
            return chart;
        }

        private LineChart<Number, Number> createTrendChart() {
            NumberAxis xAxis = new NumberAxis(0, 100, 10);
            NumberAxis yAxis = new NumberAxis(0, 100, 20);
            LineChart<Number, Number> chart = new LineChart<>(xAxis, yAxis);
            chart.setTitle("温度/震动趋势");
            return chart;
        }

        private Gauge createHealthGauge() {
            Gauge gauge = new Gauge();
            gauge.setTitle("系统健康指数");
            gauge.setMinValue(0);
            gauge.setMaxValue(100);
            return gauge;
        }

        public void updateVisualizations(ConcurrentHashMap<String, EquipmentStatus> statusMap) {
            // 更新健康评分图表
            updateHealthChart(statusMap);
            
            // 更新状态分布图表
            updateStatusChart(statusMap);
            
            // 更新趋势图表
            updateTrendChart(statusMap);
            
            // 更新仪表盘
            updateHealthGauge(statusMap);
        }

        private void updateHealthChart(ConcurrentHashMap<String, EquipmentStatus> statusMap) {
            // 实现健康评分图表更新逻辑
        }

        private void updateStatusChart(ConcurrentHashMap<String, EquipmentStatus> statusMap) {
            // 实现状态分布图表更新逻辑
        }

        private void updateTrendChart(ConcurrentHashMap<String, EquipmentStatus> statusMap) {
            // 实现趋势图表更新逻辑
        }

        private void updateHealthGauge(ConcurrentHashMap<String, EquipmentStatus> statusMap) {
            // 实现仪表盘更新逻辑
        }
    }
