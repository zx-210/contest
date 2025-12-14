import java.sql. *;
import java.util. *;
import java.util.stream.Collectors;

public class AdvancedSensorDataCleaner {

// 数据库配置（实际应用中应从配置文件读取）
private static final String DB_URL = "jdbc:mysql://localhost:3306/sensor_db";
private static final String DB_USER = "admin";
private static final String DB_PASSWORD = "secure_password123";

static class SensorData {
int id;
String sensorType;
double value;
Timestamp timestamp;
boolean isCritical;
int outlierScore; // 异常值评分

SensorData(int id, String sensorType, double value, Timestamp timestamp, boolean isCritical) {
this.id = id;
this.sensorType = sensorType;
this.value = value;
this.timestamp = timestamp;
this.isCritical = isCritical;
this.outlierScore = 0;
}
}

// 主清洗流程
public static void main(String[]  args) {
    List < SensorData > rawData = fetchRawData();
List < SensorData > cleanedData = cleanData(rawData);
saveCleanedData(cleanedData);
generateCleaningReport(cleanedData);
}

// 数据库数据获取
private static
List < SensorData > fetchRawData()
{
    List < SensorData > data = new
ArrayList <> ();
try (Connection conn = DriverManager.getConnection(DB_URL, DB_USER, DB_PASSWORD);
Statement stmt = conn.createStatement();
ResultSet rs = stmt.executeQuery("SELECT * FROM sensor_data")) {

while (rs.next()) {
data.add(new SensorData(
rs.getInt("id"),
rs.getString("sensor_type"),
rs.getDouble("value"),
rs.getTimestamp("timestamp"),
rs.getBoolean("is_critical")
));
}
} catch(SQLException
e) {
System.err.println("Database error: " + e.getMessage());
}
return data;
}

// 核心清洗算法
private
static
List < SensorData > cleanData(List < SensorData > rawData)
{
// 第一阶段：基础清洗
List < SensorData > stage1 = rawData.stream()
.filter(data -> data.value > 0) // 过滤负值
.collect(Collectors.toList());

// 第二阶段：传感器特定清洗
Map < String, Double > sensorRanges = Map.of(
    "Temperature", 40.0, // 最大合理温度
"Vibration", 6.0 // 最大合理震动值
);

List < SensorData > stage2 = new
ArrayList <> ();
for (SensorData data: stage1) {
    double maxRange = sensorRanges.getOrDefault(data.sensorType, Double.MAX_VALUE);

// 异常值检测与评分
if (data.value > maxRange) {
data.outlierScore += 5;
continue; // 跳过极端异常值
} else if (data.value > maxRange * 0.9) {
data.outlierScore += 2; // 可疑值标记
}

// 数据插值处理（简单移动平均）
if (data.outlierScore > 0) {
data.value = calculateMovingAverage(stage2, data);
}

stage2.add(data);
}

// 第三阶段：时间序列去重
Set < String > seen = new
HashSet <> ();
return stage2.stream()
.filter(data -> seen.add(data.timestamp + ":" + data.sensorType))
.collect(Collectors.toList());
}

// 移动平均插值算法
private
static
double
calculateMovingAverage(List < SensorData > recentData, SensorData
current) {
List < Double > values = recentData.stream()
.filter(data -> data.sensorType.equals(current.sensorType))
.limit(5) // 取最近5个同类数据
.mapToDouble(data -> data.value)
.sorted()
.limit(3) // 取中间3个值
.toArray();

return Arrays.stream(values).average().orElse(current.value);
}

// 数据保存
private
static
void
saveCleanedData(List < SensorData > cleanedData)
{
try (Connection conn = DriverManager.getConnection(DB_URL, DB_USER, DB_PASSWORD)) {
try (PreparedStatement stmt = conn.prepareStatement(
"INSERT INTO cleaned_sensor_data " +
"(id, sensor_type, value, timestamp, is_critical, outlier_score) " +
"VALUES (?, ?, ?, ?, ?, ?)")) {

for (SensorData data: cleanedData) {
    stmt.setInt(1, data.id);
stmt.setString(2, data.sensorType);
stmt.setDouble(3, data.value);
stmt.setTimestamp(4, data.timestamp);
stmt.setBoolean(5, data.isCritical);
stmt.setInt(6, data.outlierScore);
stmt.addBatch();
}
stmt.executeBatch();
}
System.out.println("Cleared data saved to cleaned_sensor_data table");
} catch(SQLException
e) {
System.err.println("Database save error: " + e.getMessage());
}
}

// 生成清洗报告
private
static
void
generateCleaningReport(List < SensorData > cleanedData)
{
    Map < String, DoubleSummaryStatistics > stats = cleanedData.stream()
.collect(Collectors.groupingBy(
    SensorData::getSensorType,
Collectors.summarizingDouble(SensorData::getValue)
));

System.out.println("\n=== 数据清洗报告 ===");
stats.forEach((sensor, stat) ->
System.out.printf("%s: 平均值=%.2f, 最大值=%.2f, 最小值=%.2f, 数据量=%d\n",
                  sensor,
                  stat.getAverage(),
                  stat.getMax(),
                  stat.getMin(),
                  stat.getCount())
);

long
outlierCount = cleanedData.stream()
.filter(data -> data.outlierScore > 0)
.count();
System.out.printf("\n检测到异常值记录: %d 条 (占总数据的 %.2f%%)",
outlierCount,
(double)
outlierCount / cleanedData.size() * 100);
}
}