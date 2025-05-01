**梯度下降（Gradient Descent）** 是机器学习中最核心的优化算法，用于通过迭代调整参数来最小化损失函数。以下是其完整说明：

---

### 1. **数学原理**
- **核心思想**：沿损失函数梯度（一阶导数）的**反方向**更新参数  
  $$
  \theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta J(\theta)
  $$
  - $\theta$：模型参数  
  - $\eta$：学习率（控制步长）  
  - $\nabla_\theta J(\theta)$：损失函数对参数的梯度

- **梯度计算示例**（线性回归）：  
  设损失函数 $J(\theta) = \frac{1}{2m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2$  
  则梯度：  
  $$
  \frac{\partial J}{\partial \theta_j} = \frac{1}{m}\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}
  $$

---

### 2. **算法变种对比**
| 类型              | 计算方式                     | 适用场景     | 内存需求 |
| --------------- | ------------------------ | -------- | ---- |
| **批量梯度下降**      | 使用全部数据计算梯度               | 凸优化问题    | 高    |
| **随机梯度下降(SGD)** | 每次随机选1个样本计算梯度            | 大规模数据    | 低    |
| **小批量梯度下降**     | 折中方案（常用batch_size=32/64） | 深度学习主流选择 | 中    |

---

### 3. **关键参数**
- **学习率($\eta$)**：  
  - 过大：震荡甚至发散  
  - 过小：收敛缓慢  
  - 动态调整方法：`Adam`、`Adagrad`等优化器

- **收敛条件**：  
  - 梯度范数小于阈值 $\|\nabla J(\theta)\| < \epsilon$  
  - 或损失函数变化量小于阈值

---

### 4. **可视化理解**
```python
# 梯度下降的Python伪代码
def gradient_descent(X, y, lr=0.01, epochs=100):
    theta = initialize_parameters()
    for _ in range(epochs):
        grad = compute_gradient(X, y, theta)  # 计算梯度
        theta -= lr * grad                    # 参数更新
    return theta
```

---

### 5. **注意事项**
- **局部最优**：非凸函数可能陷入局部最小值（但深度学习参数空间多为鞍点）
- **梯度消失/爆炸**：深层网络中需配合[Batch Normalization](../Normalization/批次标准化.md)或[ResNet](https://blog.csdn.net/weixin_44023658/article/details/105843701)使用
- **学习率调度**：可采用余弦退火等动态策略
