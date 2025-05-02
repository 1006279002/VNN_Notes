### **1. YOLO 核心思想**
YOLO 是一种**单阶段目标检测算法**（2016年首次提出），其核心特点是：
- **实时性**：将目标检测视为单一回归问题，直接预测边界框和类别概率（比 Fast R-CNN 更快）。
- **全局推理**：对整张图像进行**一次性预测**，避免区域提议的冗余计算。

不断使用卷积层和池化层，最后使用全连接层来获得每个网格中的具体数据。

---

### **2. 关键函数与流程**
#### **(1) 网格划分（Grid Cell）**
- 将输入图像划分为 $S \times S$ 的网格（如 $7 \times 7$）。
- 每个网格负责预测 $B$ 个边界框（Bounding Box）和对应的置信度。

#### **(2) 边界框预测函数**
每个边界框包含 5 个参数：
$$
\text{box} = (x, y, w, h, \text{confidence})
$$
- $(x, y)$：边界框中心相对于当前网格的偏移量（范围 $[0,1]$）。
- $(w, h)$：边界框的宽高相对于整张图像的占比。
- $\text{confidence}$：预测框包含目标且位置准确的置信度（$= P(\text{object}) \times \text{IOU}$）。

#### **(3) 类别概率预测**
每个网格预测 $C$ 个类别的条件概率：
$$
P(\text{class}_i | \text{object}), \quad i \in \{1, 2, ..., C\}
$$
最终类别置信度：
$$
\text{class\_score} = P(\text{class}_i | \text{object}) \times \text{confidence}
$$

#### **(4) 损失函数（YOLOv1）**
$$
L = \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^B \mathbb{1}_{ij}^{\text{obj}} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right] \\
+ \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^B \mathbb{1}_{ij}^{\text{obj}} \left[ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right] \\
+ \sum_{i=0}^{S^2} \sum_{j=0}^B \mathbb{1}_{ij}^{\text{obj}} (C_i - \hat{C}_i)^2 \\
+ \lambda_{\text{noobj}} \sum_{i=0}^{S^2} \sum_{j=0}^B \mathbb{1}_{ij}^{\text{noobj}} (C_i - \hat{C}_i)^2 \\
+ \sum_{i=0}^{S^2} \mathbb{1}_{i}^{\text{obj}} \sum_{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2
$$
其中：
- $\mathbb{1}_{ij}^{\text{obj}}$ 表示第 $i$ 个网格的第 $j$ 个边界框是否负责检测目标。
- $\lambda_{\text{coord}}$ 和 $\lambda_{\text{noobj}}$ 是平衡权重（通常设为 5 和 0.5）。

---

### **3. YOLO 版本对比**
| 版本     | 关键改进                     | 速度 (FPS) | mAP (VOC) |
| ------ | ------------------------ | -------- | --------- |
| YOLOv1 | 初始版本，单阶段检测               | 45       | 63.4%     |
| YOLOv2 | 引入锚框（Anchor Boxes）、批量归一化 | 67       | 78.6%     |
| YOLOv3 | 多尺度预测、Darknet-53 主干网络    | 30       | 82.3%     |
| YOLOv4 | CSPNet、PANet、Mish 激活函数   | 62       | 84.5%     |
| YOLOv5 | 简化实现、PyTorch 框架优化        | 140      | 85.2%     |

具体演化流程可以看这篇[文章](https://zhuanlan.zhihu.com/p/13491328897)
