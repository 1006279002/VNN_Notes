### autograd 自动求导
深度学习权重的更新依赖**梯度的计算**。在PyTorch中，只需要搭建前向计算图，利用`torch.autograd`进行自动求导就可以得到所有张量的梯度

##### torch.autograd.backward()
```python
torch.autograd.backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False, grad_variables=None)
# 自动求解梯度
```
- tensors: 用于求导的张量，如 loss
- retain_graph: 保存计算图。PyTorch 采用[动态图机制](静态计算图和动态图机制)，默认每次反向传播之后都会释放计算图。这里设置为 True 可以不释放计算图。
- create_graph: 创建导数计算图，用于高阶求导
- grad_tensors: 多梯度权重。当有多个 loss 混合需要计算梯度时，设置每个 loss 的权重。

###### retain_graph参数
因为默认会释放计算图，所以如果不进行设置就直接运行下面的示例代码的话，第二次求导就会出现报错
```python
w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)
# y=(x+w)*(w+1)
a = torch.add(w, x)
b = torch.add(w, 1)
y = torch.mul(a, b)

# 第一次执行梯度求导
y.backward()
print(w.grad)
# 第二次执行梯度求导，throw error
y.backward()
```

所以只需要保存计算图，设置`retain_graph=True`就可以正常运行了
```python
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)
    # y=(x+w)*(w+1)
    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)

    # 第一次求导，设置 retain_graph=True，保留计算图
    y.backward(retain_graph=True)
    print(w.grad)
    # 第二次求导成功，没有throw error
    y.backward()
```

###### grad_tensors参数
直接先上代码
```python
w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

a = torch.add(w, x)
b = torch.add(w, 1)

y0 = torch.mul(a, b)    # y0 = (x+w) * (w+1)
y1 = torch.add(a, b)    # y1 = (x+w) + (w+1)    dy1/dw = 2

# 把两个 loss 拼接都到一起
loss = torch.cat([y0, y1], dim=0)       # [y0, y1]
# 设置两个 loss 的权重: y0 的权重是 1，y1 的权重是 2
grad_tensors = torch.tensor([1., 2.])

loss.backward(gradient=grad_tensors)    # gradient 传入 torch.autograd.backward()中的grad_tensors
# 最终的 w 的导数由两部分组成。∂y0/∂w * 1 + ∂y1/∂w * 2
print(w.grad)
```

结果如下
```text
tensor([9.])
```


这个loss由两部分组成，分别是$y_0和y_1$，通过计算得到$\frac{\partial y_0}{\partial w}=5,\frac{\partial y_1}{\partial w}=2$，通过`grad_tensors`参数设置两个不同的loss的**权重**分别为1和2，所以最终对$w$的梯度表示为$\frac{\partial y_0}{\partial w}\times 1 + \frac{\partial y_1}{\partial w} \times 2=9$

##### torch.autograd.grad()
```python
torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False)
# 求取梯度
```
- outputs: 用于求导的张量，如 loss
- inputs: 需要梯度的张量
- create_graph: 创建导数计算图，用于高阶求导
- retain_graph:保存计算图
- grad_outputs: 多梯度权重计算

这个函数的返回值是一个**元组(tuple)** 需要取出第0个元素才是真正的梯度

下面是一个**求二阶导**的代码示例
```python
x = torch.tensor([3.], requires_grad=True)
y = torch.pow(x, 2)     # y = x**2
# 如果需要求 2 阶导，需要设置 create_graph=True，让一阶导数 grad_1 也拥有计算图
grad_1 = torch.autograd.grad(y, x, create_graph=True)   # grad_1 = dy/dx = 2x = 2 * 3 = 6
print(grad_1)
# 这里求 2 阶导
grad_2 = torch.autograd.grad(grad_1[0], x)              # grad_2 = d(dy/dx)/dx = d(2x)/dx = 2
print(grad_2)
```

结果如下
```text
(tensor([6.], grad_fn=<MulBackward0>),)
(tensor([2.]),)
```

---
需要注意几个问题
###### 每次反向传播计算要清零梯度
反向传播求导时，计算的梯度**不会自动清零**，所以需要手动添加清零函数，否则梯度会不断叠加，如下面代码示例
```python
w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)
# 进行 4 次反向传播求导，每次最后都没有清零
for i in range(4):
    a = torch.add(w, x)
    b = torch.add(w, 1)
    y = torch.mul(a, b)
    y.backward()
    print(w.grad)
    # 每次都把梯度清零  
	# w.grad.zero_()
```

输出如下
```text
tensor([5.])
tensor([10.])
tensor([15.])
tensor([20.])
```

###### 依赖叶子节点的结点，其`requires_grad`默认为True
代码示例
```python
w = torch.tensor([1.], requires_grad=True)  
x = torch.tensor([2.], requires_grad=True)  
# y = (x + w) * (w + 1)  
a = torch.add(w, x)  
b = torch.add(w, 1)  
y = torch.mul(a, b)  
  
print(a.requires_grad, b.requires_grad, y.requires_grad)
```

输出如下
```text
True True True
```

###### 叶子节点不可执行inplace操作
以加法来说，inplace 操作有`a += x`，`a.add_(x)`，改变后的值和原来的值内存地址是同一个。非 inplace 操作有`a = a + x`，`a.add(x)`，改变后的值和原来的值内存地址不是同一个。

代码示例
```python
print("非 inplace 操作")
a = torch.ones((1, ))
print(id(a), a)
# 非 inplace 操作，内存地址不一样
a = a + torch.ones((1, ))
print(id(a), a)

print("inplace 操作")
a = torch.ones((1, ))
print(id(a), a)
# inplace 操作，内存地址一样
a += torch.ones((1, ))
print(id(a), a)
```

输出如下
```text
非 inplace 操作
2248804761680 tensor([1.])
2249752115184 tensor([2.])
inplace 操作
2248804761680 tensor([1.])
2248804761680 tensor([2.])
```

如果在反向传播之前inplace改变了叶子的值，那么backward()会发生报错
```python
w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)
# y = (x + w) * (w + 1)
a = torch.add(w, x)
b = torch.add(w, 1)
y = torch.mul(a, b)
# 在反向传播之前 inplace 改变了 w 的值，再执行 backward() 会报错
w.add_(1)
y.backward()
```

```text
RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
```

### 逻辑回归
逻辑回归是线性的**二分类模型**，模型表达式$y=f(z)=\frac{1}{1+e^{-z}}$，其中$z=WX+b$，称之为[[../基本原理/常规激活函数#^83857e|sigmoid函数]]，也叫logistic函数

分类原则如下$$class=\begin{cases}0&\text{0.5>y}\\1&\text{0.5$\le$y }\end{cases}$$

这种回归的方式是可以**更好描述置信度**，符合**概率取值**，由对数几率回归变化而来$$ln\frac{y}{1-y}=WX+b$$其中$y$表示的是一个类别的概率，$1-y$代表另一个类别的概率

### PyTorch对逻辑回归的实现

^64ea81

PyTorch构建一个模型需要5个大步骤
* 数据: 包括**数据读取**，**数据清洗**，**数据划分**和**数据预处理**
* 模型: 包括**构建模型模块**，**组织复杂网络**，**初始化网络参数**，**定义网络层**
* 损失函数: 包括**创建损失函数**，**设置损失函数超参数**，根据不同任务**选择合适的损失函数**
* 优化器: 包括根据梯度使用某种优化器**更新参数**，**管理模型参数**，管理多个参数组**实现不同学习率**，**调整学习率**
* 迭代训练: 组织上面四个模块**反复训练**，包括观察训练效果，例如**绘制Loss/Accuracy曲线**或者利用**TensorBoard进行可视化分析**

代码示例
```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(10)

# ============================ step 1/5 生成数据 ============================
sample_nums = 100
mean_value = 1.7
bias = 1
n_data = torch.ones(sample_nums, 2)
# 使用正态分布随机生成样本，均值为张量，方差为标量
x0 = torch.normal(mean_value * n_data, 1) + bias      # 类别0 数据 shape=(100, 2)
# 生成对应标签
y0 = torch.zeros(sample_nums)                         # 类别0 标签 shape=(100, 1)
# 使用正态分布随机生成样本，均值为张量，方差为标量
x1 = torch.normal(-mean_value * n_data, 1) + bias     # 类别1 数据 shape=(100, 2)
# 生成对应标签
y1 = torch.ones(sample_nums)                          # 类别1 标签 shape=(100, 1)
train_x = torch.cat((x0, x1), 0)
train_y = torch.cat((y0, y1), 0)

# ============================ step 2/5 选择模型 ============================
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.features = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        x = self.sigmoid(x)
        return x

lr_net = LR()   # 实例化逻辑回归模型

# ============================ step 3/5 选择损失函数 ============================
loss_fn = nn.BCELoss()

# ============================ step 4/5 选择优化器   ============================
lr = 0.01  # 学习率
optimizer = torch.optim.SGD(lr_net.parameters(), lr=lr, momentum=0.9)

# ============================ step 5/5 模型训练 ============================
for iteration in range(1000):

    # 前向传播
    y_pred = lr_net(train_x)
    # 计算 loss
    loss = loss_fn(y_pred.squeeze(), train_y)
    # 反向传播
    loss.backward()
    # 更新参数
    optimizer.step()
    # 清空梯度
    optimizer.zero_grad()
    # 绘图
    if iteration % 20 == 0:
        mask = y_pred.ge(0.5).float().squeeze()  # 以0.5为阈值进行分类
        correct = (mask == train_y).sum()  # 计算正确预测的样本个数
        acc = correct.item() / train_y.size(0)  # 计算分类准确率

        plt.scatter(x0.data.numpy()[:, 0], x0.data.numpy()[:, 1], c='r', label='class 0')
        plt.scatter(x1.data.numpy()[:, 0], x1.data.numpy()[:, 1], c='b', label='class 1')

        w0, w1 = lr_net.features.weight[0]
        w0, w1 = float(w0.item()), float(w1.item())
        plot_b = float(lr_net.features.bias[0].item())
        plot_x = np.arange(-6, 6, 0.1)
        plot_y = (-w0 * plot_x - plot_b) / w1

        plt.xlim(-5, 7)
        plt.ylim(-7, 7)
        plt.plot(plot_x, plot_y)

        plt.text(-5, 5, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.title("Iteration: {}\nw0:{:.2f} w1:{:.2f} b: {:.2f} accuracy:{:.2%}".format(iteration, w0, w1, plot_b, acc))
        plt.legend()
        # plt.savefig(str(iteration / 20)+".png")
        plt.show()
        plt.pause(0.5)
        # 如果准确率大于 99%，则停止训练
        if acc > 0.99:
            break
```

结果如下图(引用博客)
![gif2](../data/gif2.gif)
