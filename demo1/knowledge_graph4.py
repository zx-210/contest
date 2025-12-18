import random
import numpy as np
from sklearn.svm import SVC

# 双核驱动核心诊断流程
def fault_diagnosis(device_id):
    # 1. 模拟采集振动/温度/磁场数据（数据层）
    vib_rms = random.uniform(0.2, 1.0)  # 振动有效值
    temp = 45 + random.uniform(-2, 5)    # 温度
    mag = 100 + random.uniform(-5, 8)    # 磁场

    # 2. 数据核：SVM模型推理故障
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(np.random.rand(30,3), ["转子不平衡", "正常", "轴承内圈故障"]*10)
    pred = svm.predict_proba([[vib_rms, temp, mag]])[0]
    fault = svm.classes_[np.argmax(pred)]
    conf = round(np.max(pred), 2)

    # 3. 知识核：专家规则校验
    verify = True if (fault=="正常" or vib_rms>0.5) else False
    msg = "校验通过" if verify else "物理规则不匹配"

    # 输出结果
    return {"设备ID":device_id, "诊断结果":fault if verify else "待确认", "置信度":conf, "校验":msg}

# 演示运行
print(fault_diagnosis("DEV-001"))