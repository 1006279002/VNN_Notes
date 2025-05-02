### **MobileNetV3 中的 Squeeze-and-Excitation (SE) 模块**

MobileNetV3 通过引入 **Squeeze-and-Excitation (SE) 模块** 进一步优化了通道注意力机制，显著提升了模型性能，尤其是在资源受限的移动端设备上。SE 模块能够自适应地重新校准通道特征响应，增强重要特征并抑制无关特征。

---

## **1. SE 模块的核心思想**
SE 模块包含两个主要操作：
1. **Squeeze（压缩）**：全局**平均**池化（GAP）压缩空间维度，生成通道描述符。
2. **Excitation（激励）**：通过全连接层学习通道间的非线性关系，生成通道权重。

**数学表达**：
$$\mathbf{z} = \mathbf{F}_{sq}(\mathbf{u}) = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} \mathbf{u}(i,j)$$
$$\mathbf{s} = \mathbf{F}_{ex}(\mathbf{z}, \mathbf{W}) = \sigma(\mathbf{W}_2 \delta(\mathbf{W}_1 \mathbf{z}))$$
$$\mathbf{\tilde{u}} = \mathbf{s} \odot \mathbf{u}$$
其中：
- $\mathbf{u}$是输入特征图（$C \times H \times W$）。
- $\mathbf{z}$ 是压缩后的通道描述符（$C \times 1 \times 1$）。
- $\mathbf{s}$ 是激励后的通道权重（$C \times 1 \times 1$）。
- $\odot$表示逐通道乘法（Channel-wise Scaling）。

---

## **2. SE 模块在 MobileNetV3 中的应用**
MobileNetV3 结合了 **SE 模块** 和 [[../基本原理/常规激活函数#^c8ed49|h-swish激活函数]]，优化了计算效率：
- **SE 模块的位置**：通常插入在 **倒残差块（Inverted Residual Block）** 的深度卷积（Depthwise Conv）之后。
- **计算量优化**：
  - 使用 **瓶颈结构（Bottleneck）** 减少全连接层的计算量。
  - 采用 **ReLU → h-swish** 组合提升非线性表达能力。

**MobileNetV3-SE 结构示例**：
```python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        reduced_channels = channels // reduction_ratio
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Squeeze (GAP)
            nn.Conv2d(channels, reduced_channels, 1),  # FC1 (降维)
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1),  # FC2 (升维)
            nn.Hardsigmoid()  # 替代Sigmoid，更高效
        )

    def forward(self, x):
        return x * self.se(x)  # 通道加权
```

---

## **3. SE 模块的优势**
| **特点**          | **传统卷积** | **SE 增强卷积** |
|-------------------|-------------|-----------------|
| **计算量**        | 高          | 仅增加少量计算（GAP + 2 FC） |
| **参数量**        | 固定        | 可调节（通过 `reduction_ratio`） |
| **特征选择能力**  | 无          | 自适应增强重要通道 |
| **部署友好性**    | 一般        | 支持硬件加速（如 NPU） |
