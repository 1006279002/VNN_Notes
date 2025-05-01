Layer Normalization（层归一化）是深度学习中用于稳定神经网络训练的重要技术，其核心运作原理如下：

### 1. 计算方式
对**单个样本**的某一层所有神经元输出进行标准化：
$$
\mu = \frac{1}{H}\sum_{i=1}^{H}x_i \quad \sigma^2 = \frac{1}{H}\sum_{i=1}^{H}(x_i-\mu)^2
$$
$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \quad \text{(标准化)}
$$
$$
y_i = \gamma \hat{x}_i + \beta \quad \text{(缩放平移)}
$$
其中：
- $H$：该层的神经元数量
- $\gamma,\beta$：可学习的缩放和偏移参数
- $\epsilon$：防止除零的小常数（如1e-5）

### 2. 与Batch Norm对比
| 特性                | Layer Norm               | Batch Norm              |
|---------------------|--------------------------|-------------------------|
| 归一化维度           | 沿特征维度               | 沿批量维度              |
| 小批量稳定性         | 不受批量大小影响         | 需要足够大的批量        |
| RNN/Transformer适用性| 适用                     | 不适用                  |

### 3. 在Transformer中的应用
- 位置：出现在[自注意力机制](../Transformer/自注意力机制.md)层和FFN层之后
- 作用：缓解梯度消失/爆炸，允许更深的网络结构
- 典型配置：
  ```python
  # PyTorch实现
  self.norm = nn.LayerNorm(hidden_size)
  ```

### 4. 优势
- 对序列长度变化鲁棒（适合NLP任务）
- 训练和推理时行为一致
- 在[Transformer](../Transformer/Sequence-to-sequence)架构中与残差连接协同工作：
  $$
  \text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))
  $$
