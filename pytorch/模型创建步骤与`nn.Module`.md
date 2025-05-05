### 网络模型的创建步骤
两个要素
* 构建子模块
* 拼接子模块

借用之前的[[DataLoader与DataSet#^93820c|Lenet]]示例，下面是其源码
```python
class LeNet(nn.Module):
 # 子模块创建
    def __init__(self, classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)
 # 子模块拼接
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
```
继承`nn.Module`，自定义的模型必须实现`__init__()`和`forward()`方法，前者就是**创建子模块**，后者就是**连接子模块**

当训练时调用`outputs = net(inputs)`的时候，会进入`module.py`的`__call__()`函数中，最终会调用模型自己的`forward()`函数
```python
    def __call__(self, *input, **kwargs):
        for hook in self._forward_pre_hooks.values():
            result = hook(self, input)
            if result is not None:
                if not isinstance(result, tuple):
                    result = (result,)
                input = result
        if torch._C._get_tracing_state():
            result = self._slow_forward(*input, **kwargs)
        else:
            result = self.forward(*input, **kwargs)
        ...
        ...
        ...
```

### nn.Module
其含八个属性，全都是`OrderDict`，每个模型的`__init__()`都会调用父类的`__init()__`，如下所示
```python
# 旧版本代码，现版本存在一定差异，但是理解是一样的
   def __init__(self):
        """
        Initializes internal Module state, shared by both nn.Module and ScriptModule.
        """
        torch._C._log_api_usage_once("python.nn_module")

        self.training = True
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._backward_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._forward_pre_hooks = OrderedDict()
        self._state_dict_hooks = OrderedDict()
        self._load_state_dict_pre_hooks = OrderedDict()
        self._modules = OrderedDict()
```
- `_parameters` 属性：存储管理 nn.Parameter 类型的参数
- `_modules` 属性：存储管理 nn.Module 类型的参数
- `_buffers` 属性：存储管理缓冲属性，如 BN 层中的 running_mean
- 5 个 `***_hooks` 属性：存储管理钩子函数

已知`nn.Conv2d`和`nn.Linear`都是继承自`nn.Module`，所以一个module中是包含多个子module的，而这时候`_module`属性就会包含这几个子module

在`nn.Module`中，通过`__setattr__()`函数拦截所有的**类属性赋值操作**，通过这个函数来判断具体是存放给`_oarameter`还是`_modules`，具体代码如下
```python
    def __setattr__(self, name, value):
        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError("cannot assign '{}' as parameter '{}' "
                                "(torch.nn.Parameter or None expected)"
                                .format(torch.typename(value), name))
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError("cannot assign '{}' as child module '{}' "
                                    "(torch.nn.Module or None expected)"
                                    .format(torch.typename(value), name))
                modules[name] = value
            ...
            ...
            ...
```

#### 总结
- 一个 module 里可包含多个**子 module**。比如 LeNet 是一个 Module，里面包括多个卷积层、池化层、全连接层等子 module
- 一个 module 相当于一个运算，必须实现 `forward()` 函数
- 每个 module 都有 **8 个字典管理自己的属性**

### 模型容器
除了上面的内容还存在**三种模型容器**
- nn.Sequential：按照顺序包装多个网络层
- nn.ModuleList：像 python 的 list 一样包装多个网络层，可以迭代
- nn.ModuleDict：像 python 的 dict 一样包装多个网络层，通过 (key, value) 的方式为每个网络层指定名称。

#### nn.Sequential
传统神经网络存在**特征提取**和**分类器**两个部分，而这两个部分就都可以分别使用`nn.Sequential`来包装，对于我们的Lenet示例来说就可以产生如下代码
```python
class LeNetSequential(nn.Module):
    def __init__(self, classes):
        super(LeNet2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x
```

在初始化时，`nn.Sequential`会调用`__init__()`方法，将每一个子 module 添加到 自身的`_modules`属性中。传入的参数可以是一个 list，或者一个 OrderDict。如果是一个 OrderDict，那么则使用 OrderDict 里的 key，否则使用数字作为 key

```python
    def __init__(self, *args):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
```

按照示例，初始化后存在两个子`module`:`features`和`classifier`，其中`features`中的子module将网络层以**序号**作为key，向前传播时依序号调用其中的所有的module

一旦网络层增多，难以利用**序号**查找特定网络层的时候，可以使用OrderDict来处理，具体代码如下
```python
class LeNetSequentialOrderDict(nn.Module):
    def __init__(self, classes):
        super(LeNetSequentialOrderDict, self).__init__()

        self.features = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(3, 6, 5),
            'relu1': nn.ReLU(inplace=True),
            'pool1': nn.MaxPool2d(kernel_size=2, stride=2),

            'conv2': nn.Conv2d(6, 16, 5),
            'relu2': nn.ReLU(inplace=True),
            'pool2': nn.MaxPool2d(kernel_size=2, stride=2),
        }))

        self.classifier = nn.Sequential(OrderedDict({
            'fc1': nn.Linear(16*5*5, 120),
            'relu3': nn.ReLU(),

            'fc2': nn.Linear(120, 84),
            'relu4': nn.ReLU(inplace=True),

            'fc3': nn.Linear(84, classes),
        }))
        ...
        ...
        ...
```

这样就可以按照自定义标签来初始化`Sequential`，方便后续查找

##### 总结
`nn.Sequetial`是`nn.Module`的容器，用于按顺序包装一组网络层，有以下两个特性。
- 顺序性：各网络层之间严格按照顺序构建，我们在构建网络时，一定要注意前后网络层之间输入和输出数据之间的**形状是否匹配**
- 自带`forward()`函数：在`nn.Sequetial`的`forward()`函数里通过 for 循环依次读取每个网络层，执行前向传播运算。这使得我们我们构建的**模型更加简洁**

#### nn.ModuleList
`nn.ModuleList`是`nn.Module`的容器，包装一组网络层，以迭代的方式调用网络层

核心方法
- `append()`：在 ModuleList 后面添加网络层
- `extend()`：拼接两个 ModuleList
- `insert()`：在 ModuleList 的指定位置中插入网络层

可以如同下面这样**简单迭代**创建网络层，虽然在`forward()`中要手动调用每一个网络层
```python
class ModuleList(nn.Module):
    def __init__(self):
        super(ModuleList, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(20)])

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
        return x


net = ModuleList()

print(net)

fake_data = torch.ones((10, 10))

output = net(fake_data)

print(output)
```

#### nn.ModuleDict
`nn.ModuleDict`是`nn.Module`的容器，包装一组网络层

核心方法
- `clear()`：清空 ModuleDict
- `items()`：返回可迭代的键值对 `(key, value)`
- `keys()`：返回字典的所有 key
- `values()`：返回字典的所有 value
- `pop()`：返回一对键值，并从字典中删除

下面的代码示例创建了两个dict:`self.choices`和`self.activations`，向前传播时通过传入对应的key来执行对应的网络层
```python
class ModuleDict(nn.Module):
    def __init__(self):
        super(ModuleDict, self).__init__()
        self.choices = nn.ModuleDict({
            'conv': nn.Conv2d(10, 10, 3),
            'pool': nn.MaxPool2d(3)
        })

        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'prelu': nn.PReLU()
        })

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x


net = ModuleDict()

fake_img = torch.randn((4, 10, 32, 32))

output = net(fake_img, 'conv', 'relu')
# output = net(fake_img, 'conv', 'prelu')
print(output)
```

#### 容器总结
- `nn.Sequential`：顺序性，各网络层之间严格按照顺序执行，常用于 block 构建，在前向传播时的代码调用变得简洁
- `nn.ModuleList`：迭代行，常用于**大量重复网络构建**，通过 for 循环实现重复构建
- `nn.ModuleDict`：索引性，常用于可选择的网络层

### PyTorch中的AlexNet
AlexNet 是 Hinton 和他的学生等人在 2012 年提出的卷积神经网络

特点如下
- 采用 ReLU 替换饱和激活 函数，减轻梯度消失
- 采用 LRN (Local Response Normalization) 对数据进行局部归一化，减轻梯度消失
- 采用 Dropout 提高网络的鲁棒性，增加泛化能力
- 使用 Data Augmentation，包括 TenCrop 和一些色彩修改

下面是torchvision.models中的AlexNet代码，采用了`nn.Sequential`封装了网络层
```python
class AlexNet(nn.Module):  
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:  
        super().__init__()  
        _log_api_usage_once(self)  
        self.features = nn.Sequential(  
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(kernel_size=3, stride=2),  
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(kernel_size=3, stride=2),  
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(kernel_size=3, stride=2),  
        )  
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))  
        self.classifier = nn.Sequential(  
            nn.Dropout(p=dropout),  
            nn.Linear(256 * 6 * 6, 4096),  
            nn.ReLU(inplace=True),  
            nn.Dropout(p=dropout),  
            nn.Linear(4096, 4096),  
            nn.ReLU(inplace=True),  
            nn.Linear(4096, num_classes),  
        )  
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        x = self.features(x)  
        x = self.avgpool(x)  
        x = torch.flatten(x, 1)  
        x = self.classifier(x)  
        return x
```


