### **MobileNet 简介**  
**MobileNet** 是由 Google 团队提出的一系列轻量级卷积神经网络，专为 **移动端和嵌入式设备** 设计，在保持较高精度的同时大幅降低计算量和参数量。其核心思想是使用 **深度可分离卷积（Depthwise Separable Convolution）** 替代传统卷积，显著减少计算成本。

---

### **核心特点**
#### 1. **深度可分离卷积（Depthwise Separable Convolution）**
   - **传统卷积**：计算成本高，参数量为 $K \times K \times C_{in} \times C_{out}$（$K$ 为卷积核大小）。  
   - **深度可分离卷积**：分为两步(每步后面都做[[../Normalization/批次标准化|批次标准化]]和[[../基本原理/常规激活函数#^c8ed49|ReLU6激活函数]])：  
     - **Depthwise 卷积**：每个**输入通道单独卷积**，无法增加通道数，参数量 $K \times K \times C_{in}$。  
     - **Pointwise 卷积**：$1 \times 1$ 卷积融合通道，**混合通道之间的信息**，便于降维升维，参数量 $1 \times 1 \times C_{in} \times C_{out}$。  
   - **计算量对比**：  
     $$
     \text{传统卷积计算量} : \text{深度可分离卷积计算量} \approx \frac{1}{C_{out}} + \frac{1}{K^2}
     $$
     例如，$K=3$、$C_{out}=256$ 时，计算量减少约 **8~9 倍**。
	先通过**深度卷积**得到关联性不强的每个通道特征图，然后再利用逐点卷积将**几个特征图再关联起来**

#### 2. **宽度乘子（Width Multiplier）与分辨率乘子（Resolution Multiplier）**
   - **Width Multiplier（$\alpha$）**：按比例减少通道数（默认 $\alpha=1$），进一步压缩模型（如 MobileNet-0.75）。  可以在不同的平台按照不同的需求进行实际调整，满足不同的需求。
   - **Resolution Multiplier（$\beta$）**：降低输入图像分辨率（如 $224 \times 224 \to 192 \times 192$），减少计算量。

---

### **MobileNet 系列演进**
| 版本              | 核心改进                                                                    | 典型应用场景        |
| --------------- | ----------------------------------------------------------------------- | ------------- |
| **MobileNetV1** | 首次引入深度可分离卷积                                                             | 移动端实时分类/检测    |
| **MobileNetV2** | 添加**倒残差结构**（Inverted Residuals，与ResNet残差结构功能相反）和线性瓶颈（Linear Bottleneck） | 低算力设备（如树莓派）   |
| **MobileNetV3** | 结合 NAS（神经架构搜索）和 h-swish 激活函数，优化计算效率                                     | 手机端 AI（如人脸解锁） |
| **MobileNetV4** | 统一卷积与注意力机制，支持超轻量级设计（2024 年最新版本）                                         | 边缘计算（如无人机视觉）  |

---

### **性能对比（ImageNet 分类任务）**
| 模型            | 参数量 (M) | 计算量 (MFLOPs) | Top-1 准确率 |
|-----------------|------------|-----------------|--------------|
| MobileNetV1     | 4.2        | 569             | 70.6%        |
| MobileNetV2     | 3.4        | 300             | 72.0%        |
| MobileNetV3-Small | 2.9       | 66              | 67.4%        |
| MobileNetV4      | 2.1       | 42              | 75.8%        |

---

### **代码示例（PyTorch）**
```python
import torch
from torchvision.models import mobilenet_v3_small

model = mobilenet_v3_small(pretrained=True)
input = torch.randn(1, 3, 224, 224)
output = model(input)  # 输出分类结果
```

---

### **应用场景**
1. **移动端实时检测**：与 YOLO 结合（如 YOLO-MobileNet 轻量化版本）。  
2. **嵌入式设备**：树莓派、Jetson Nano 等低功耗场景。  
3. **端侧 AI**：手机拍照增强、AR 滤镜等。  
