主要介绍了如何使用Hook函数提取网络中的特征图进行可视化，和CAM(Class Activation Map 类激活图)

### Hook函数概念
Hook 函数是在不改变主体的情况下，实现额外功能。由于 PyTorch 是基于**动态图**实现的，因此在一次迭代运算结束后，一些中间变量如非叶子节点的梯度和特征图，会被**释放**掉。在这种情况下想要提取和记录这些中间变量，就需要使用 Hook 函数。

#### torch.Tensor.register_hook(hook)
注册一个反向传播 **hook 函数**，仅输入一个参数，为张量的梯度

```python
# hook函数定义如下
def hook_name(grad)
```
其中`grad`是张量的梯度

示例代码如下
```python
w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)
a = torch.add(w, x)
b = torch.add(w, 1)
y = torch.mul(a, b)

# 保存梯度的 list
a_grad = list()

# 定义 hook 函数，把梯度添加到 list 中
def grad_hook(grad):
 a_grad.append(grad)

# 一个张量注册 hook 函数
handle = a.register_hook(grad_hook)

y.backward()

# 查看梯度
print("gradient:", w.grad, x.grad, a.grad, b.grad, y.grad)
# 查看在 hook 函数里 list 记录的梯度
print("a_grad[0]: ", a_grad[0])
handle.remove()
```

输出如下
```text
gradient: tensor([5.]) tensor([2.]) None None None
a_grad[0]:  tensor([2.])
```

`hook`函数可以修改梯度的值，无需返回也可以作为新的梯度赋值给原来的梯度，代码示例如下
```python
w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)
a = torch.add(w, x)
b = torch.add(w, 1)
y = torch.mul(a, b)

a_grad = list()

def grad_hook(grad):
    grad *= 2
    return grad*3

handle = w.register_hook(grad_hook)

y.backward()

# 查看梯度
print("w.grad: ", w.grad)
handle.remove()
```

输出结果如下
```text
w.grad:  tensor([30.])
```

#### torch.nn.Module.register_forward_hook(hook)
注册 module 的前向传播`hook`函数，可用于获取中间的 feature map

```python
# hook函数定义
def hook(module, input, output)
```
- `module`：当前网络层
- `input`：当前网络层输入数据
- `output`：当前网络层输出数据

下面的代码是记录中间卷积层输入和输出的feature map
```python
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 2, 3)
            self.pool1 = nn.MaxPool2d(2, 2)

        def forward(self, x):
            x = self.conv1(x)
            x = self.pool1(x)
            return x

    def forward_hook(module, data_input, data_output):
        fmap_block.append(data_output)
        input_block.append(data_input)

    # 初始化网络
    net = Net()
    net.conv1.weight[0].detach().fill_(1)
    net.conv1.weight[1].detach().fill_(2)
    net.conv1.bias.data.detach().zero_()

    # 注册hook
    fmap_block = list()
    input_block = list()
    net.conv1.register_forward_hook(forward_hook)

    # inference
    fake_img = torch.ones((1, 1, 4, 4))   # batch size * channel * H * W
    output = net(fake_img)


    # 观察
    print("output shape: {}\noutput value: {}\n".format(output.shape, output))
    print("feature maps shape: {}\noutput value: {}\n".format(fmap_block[0].shape, fmap_block[0]))
    print("input shape: {}\ninput value: {}".format(input_block[0][0].shape, input_block[0]))
```

输出如下
```text
output shape: torch.Size([1, 2, 1, 1])
output value: tensor([[[[ 9.]],
         [[18.]]]], grad_fn=<MaxPool2DWithIndicesBackward>)
feature maps shape: torch.Size([1, 2, 2, 2])
output value: tensor([[[[ 9.,  9.],
          [ 9.,  9.]],
         [[18., 18.],
          [18., 18.]]]], grad_fn=<ThnnConv2DBackward>)
input shape: torch.Size([1, 1, 4, 4])
input value: (tensor([[[[1., 1., 1., 1.],
          [1., 1., 1., 1.],
          [1., 1., 1., 1.],
          [1., 1., 1., 1.]]]]),)
```

#### torch.Tensor.register_forward_pre_hook()
注册module的**向前传播前**的`hook`函数。可用于获取输入数据
```python
# hook函数定义
def hook(module,input)
```
- `module`：当前网络层
- `input`：当前网络层输入数据

#### torch.Tensor.register_backward_hook()
注册 module 的反向传播的`hook`函数，可用于获取梯度
```python
# hook函数定义
def hook(module, input, output)
```
- `module`：当前网络层
- `input`：当前网络层输入的梯度数据
- `output`：当前网络层输出的梯度数据

代码示例
```python
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 2, 3)
            self.pool1 = nn.MaxPool2d(2, 2)

        def forward(self, x):
            x = self.conv1(x)
            x = self.pool1(x)
            return x

    def forward_hook(module, data_input, data_output):
        fmap_block.append(data_output)
        input_block.append(data_input)

    def forward_pre_hook(module, data_input):
        print("forward_pre_hook input:{}".format(data_input))

    def backward_hook(module, grad_input, grad_output):
        print("backward hook input:{}".format(grad_input))
        print("backward hook output:{}".format(grad_output))

    # 初始化网络
    net = Net()
    net.conv1.weight[0].detach().fill_(1)
    net.conv1.weight[1].detach().fill_(2)
    net.conv1.bias.data.detach().zero_()

    # 注册hook
    fmap_block = list()
    input_block = list()
    net.conv1.register_forward_hook(forward_hook)
    net.conv1.register_forward_pre_hook(forward_pre_hook)
    net.conv1.register_backward_hook(backward_hook)

    # inference
    fake_img = torch.ones((1, 1, 4, 4))   # batch size * channel * H * W
    output = net(fake_img)

    loss_fnc = nn.L1Loss()
    target = torch.randn_like(output)
    loss = loss_fnc(target, output)
    loss.backward()
```

输出如下
```text
forward_pre_hook input:(tensor([[[[1., 1., 1., 1.],
          [1., 1., 1., 1.],
          [1., 1., 1., 1.],
          [1., 1., 1., 1.]]]]),)
backward hook input:(None, tensor([[[[0.5000, 0.5000, 0.5000],
          [0.5000, 0.5000, 0.5000],
          [0.5000, 0.5000, 0.5000]]],
        [[[0.5000, 0.5000, 0.5000],
          [0.5000, 0.5000, 0.5000],
          [0.5000, 0.5000, 0.5000]]]]), tensor([0.5000, 0.5000]))
backward hook output:(tensor([[[[0.5000, 0.0000],
          [0.0000, 0.0000]],
         [[0.5000, 0.0000],
          [0.0000, 0.0000]]]]),)
```

### `hook`函数实现机制
`hook`函数实现的原理是在`module`的`__call()__`函数进行拦截，`__call()__`函数可以分为 4 个部分：
- 第 1 部分是实现 `_forward_pre_hooks`
- 第 2 部分是实现 forward 前向传播
- 第 3 部分是实现 `_forward_hooks`
- 第 4 部分是实现 `_backward_hooks`

由于卷积层也是一个`module`，因此可以记录`_forward_hooks`。

### hook函数提取网络的特征图
通过`hook`函数获取 AlexNet 每个卷积层的所有卷积核参数，以形状作为 key，value 对应该层多个卷积核的 list。然后取出每层的第一个卷积核，形状是 \[1, in_channle, h, w]，转换为 \[in_channle, 1, h, w]

具体代码看实例

### CAM算法
CAM是类激活图，作用是中间层的**特征可视化**，对深度网络有更好的解释性

全局平均池化GAP，就是将最后的特征图的每一个通道进行一个平均池化，这样最后的向量长度就和**通道数一致**，再做全连接分类。这时候就可以获得各种不同通道对应的权重，利用权重计算通道，就可以获得类激活图了。

特征图通过hook提取，权重通过网络的参数列表获得，定义$c_x$是通道， 那么类激活图就是$$(w_1,w_2,...,w_N)(c_1,c_2,...,c_N)^T$$通过将类激活图resize，和原图进行相乘就可以提取出图中重要的部分




