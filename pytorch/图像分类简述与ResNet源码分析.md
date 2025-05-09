### 模型概览
在`torchvision.model`中，有很多封装好的模型

可以分类 3 类：
- 经典网络
	- alexnet
	- vgg
	- resnet
	- inception
	- densenet
	- googlenet
- 轻量化网络
	- squeezenet
	- mobilenet
	- shufflenetv2
- 自动神经结构搜索方法的网络
	- mnasnet

现在还加入了transformer的模型

### ResNet使用
以`ResNet 18`为例

先加载训练好的模型参数
```python
resnet18 = models.resnet18()

# 修改全连接层的输出
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 2)

# 加载模型参数
checkpoint = torch.load(m_path)
resnet18.load_state_dict(checkpoint['model_state_dict'])
```

需要将模型加载到GPU上，并且转换为`eval()`模式

在interface时，主要流程如下
- 代码要放在`with torch.no_grad():`下。`torch.no_grad()`会关闭反向传播，可以减少内存、加快速度。  
- 根据路径读取图片，把图片转换为 tensor，然后使用`unsqueeze_(0)`方法把形状扩大为 $B\times C\times H\times W$，再把 tensor 放到 GPU 上 。  
- 模型的输出数据`outputs`的形状是 $1\times2$，表示 `batch_size` 为 1，分类数量为 2。`torch.max(outputs,0)`是返回`outputs`中**每一列**最大的元素和索引，`torch.max(outputs,1)`是返回`outputs`中**每一行**最大的元素和索引。  这里使用`_, pred_int = torch.max(outputs.data, 1)`返回最大元素的索引，然后根据索引获得 label：`pred_str = classes[int(pred_int)]`。

```python
    with torch.no_grad():
        for idx, img_name in enumerate(img_names):

            path_img = os.path.join(img_dir, img_name)

            # step 1/4 : path --> img
            img_rgb = Image.open(path_img).convert('RGB')

            # step 2/4 : img --> tensor
            img_tensor = img_transform(img_rgb, inference_transform)
            img_tensor.unsqueeze_(0)
            img_tensor = img_tensor.to(device)

            # step 3/4 : tensor --> vector
            outputs = resnet18(img_tensor)

            # step 4/4 : get label
            _, pred_int = torch.max(outputs.data, 1)
            pred_str = classes[int(pred_int)]
```

interface阶段的注意事项：
- 确保 model 处于 eval 状态，而非 trainning 状态
- 设置 `torch.no_grad()`，减少内存消耗，加快运算速度
- 数据预处理需要保持一致，比如 RGB 或者 rBGR

### 残差连接
以ResNet为例
![残差结构](../Excalidraw/残差结构)

$F(x)$路径被称为**残差路径**，用于拟合残差，$x$路径被称为shortcut，要求$F(x)$和$x$尺寸相同，实现element-wise计算

shortcut存在两种类型，取决于残差路径是否改变了`feature map`数量和尺寸
- 一种是将输入$x$原封不动地输出。
- 另一种则需要经过 $1\times1$ 卷积来**升维**或者**降采样**，主要作用是将输出与 $F(x)$ 路径的输出保持`shape`一致，对网络性能的提升并不明显。

### 网络结构
ResNet存在许多变种，数据处理流程大致如下
- 输入的图片形状是 $3\times224\times224$。
- 图片经过 `conv1` 层，输出图片大小为 $64\times112\times112$。
- 图片经过 `max pool` 层，输出图片大小为 $64\times56\times56$。
- 图片经过 `conv2` 层，输出图片大小为 $64\times56\times56$。**（注意，图片经过这个 `layer`, 大小是不变的）**
- 图片经过 `conv3` 层，输出图片大小为 $128\times28\times28$。
- 图片经过 `conv4` 层，输出图片大小为 $256\times14\times14$。
- 图片经过 `conv5` 层，输出图片大小为 $512\times7\times7$。
- 图片经过 `avg pool` 层，输出大小为 $512\times1\times1$。
- 图片经过 `fc` 层，输出维度为 $num\_classes$，表示每个分类的 `logits`。

下面，我们称每个 `conv` 层为一个 `layer`（第一个 `conv` 层就是一个**卷积层**，因此第一个 `conv` 层除外）。

其中 `ResNet 18`、`ResNet 34` 的每个 `layer` 由多个 `BasicBlock` 组成，只是每个 `layer` 里堆叠的 `BasicBlock` 数量不一样。

而 `ResNet 50`、`ResNet 101`、`ResNet 152` 的每个 `layer` 由多个 `Bottleneck` 组成，只是每个 `layer` 里堆叠的 `Bottleneck` 数量不一样

### 源码分析
看一看各个`ResNet`的源码
#### 构造函数
相较过去的旧代码，新版pytorch将**预训练**给删除，以weights来作为代替

##### ResNet 18
构造函数如下所示
```python
def resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:  
    
    weights = ResNet18_Weights.verify(weights)  
  
    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, **kwargs)
```

其中的`[2, 2, 2, 2]` 表示有 4 个 `layer`，每个 layer 中有 2 个 `BasicBlock`，每个`BasicBlock`中又有两个卷积层，加上一开始的卷积层和最后的全连接层，总共就会有$1+4\times4+1=18$个层，后面以此类推

##### ResNet 34
构造函数如下所示
```python
def resnet34(*, weights: Optional[ResNet34_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:  
    
    weights = ResNet34_Weights.verify(weights)  
  
    return _resnet(BasicBlock, [3, 4, 6, 3], weights, progress, **kwargs)
```

##### ResNet 50
构造函数如下所示
```python
def resnet50(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:  
    
    weights = ResNet50_Weights.verify(weights)  
  
    return _resnet(Bottleneck, [3, 4, 6, 3], weights, progress, **kwargs)
```
注意它采用的是`Bottleneck`而不是先前的`BasicBlock`

##### `_resnet()`
上面所有的构造函数最终都调用了这个函数，主要功能是创建模型，然后加载训练好的参数，代码如下
```python
def _resnet(  
    block: Type[Union[BasicBlock, Bottleneck]],  
    layers: List[int],  
    weights: Optional[WeightsEnum],  
    progress: bool,  
    **kwargs: Any,  
) -> ResNet:  
    if weights is not None:  
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))  
  
    model = ResNet(block, layers, **kwargs)  
  
    if weights is not None:  
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))  
  
    return model
```

### ResNet()
#### 构造函数
函数代码如下
```python
def __init__(  
    self,  
    block: Type[Union[BasicBlock, Bottleneck]],  
    layers: List[int],  
    num_classes: int = 1000,  
    zero_init_residual: bool = False,  
    groups: int = 1,  
    width_per_group: int = 64,  
    replace_stride_with_dilation: Optional[List[bool]] = None,  
    norm_layer: Optional[Callable[..., nn.Module]] = None,  
) -> None:  
    super().__init__()  
    _log_api_usage_once(self)  
    if norm_layer is None:  
        norm_layer = nn.BatchNorm2d  
    self._norm_layer = norm_layer  
  
    self.inplanes = 64  
    self.dilation = 1  
    if replace_stride_with_dilation is None:  
        # each element in the tuple indicates if we should replace  
        # the 2x2 stride with a dilated convolution instead        
        replace_stride_with_dilation = [False, False, False]  
    if len(replace_stride_with_dilation) != 3:  
        raise ValueError(  
            "replace_stride_with_dilation should be None "  
            f"or a 3-element tuple, got {replace_stride_with_dilation}"  
        )  
    self.groups = groups  
    self.base_width = width_per_group  
    self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)  
    self.bn1 = norm_layer(self.inplanes)  
    self.relu = nn.ReLU(inplace=True)  
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  
    self.layer1 = self._make_layer(block, 64, layers[0])  
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])  
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])  
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])  
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  
    self.fc = nn.Linear(512 * block.expansion, num_classes)  
  
    for m in self.modules():  
        if isinstance(m, nn.Conv2d):  
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")  
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):  
            nn.init.constant_(m.weight, 1)  
            nn.init.constant_(m.bias, 0)  
  
    # Zero-initialize the last BN in each residual branch,  
    # so that the residual branch starts with zeros, and each residual block behaves like an identity.    # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677    if zero_init_residual:  
        for m in self.modules():  
            if isinstance(m, Bottleneck) and m.bn3.weight is not None:  
                nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]  
            elif isinstance(m, BasicBlock) and m.bn2.weight is not None:  
                nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
```


构造函数的重要参数如下：
- `block`：每个 `layer` 里面使用的 `block`，可以是 `BasicBlock`,`Bottleneck`。
- `num_classes`：分类数量，用于构建最后的全连接层。
- `layers`：一个 list，表示每个 `layer` 中 `block` 的数量。

构造函数的主要流程如下：
- 判断是否传入 `norm_layer`，没有传入，则使用 `BatchNorm2d`。  
- 判断是否传入孔洞卷积参数 `replace_stride_with_dilation`，如果不指定，则赋值为 `[False, False, False]`，表示不使用孔洞卷积。  
- 读取分组卷积的参数 `groups`，`width_per_group`。  
- 然后真正开始构造网络。  
	- `conv1` 层的结构是 `Conv2d -> norm_layer -> ReLU -> MaxPool2d`。
	-  `layer1`层，这个 `layer` 的参数没有指定 `stride`，默认 `stride=1`，因此这个 `layer` 不会改变图片大小  
	-  `layer2`层（注意这个 `layer` 和后面的`layer`指定 `stride=2`，会降采样，详情看下面 `_make_layer` 的[[#^181453|讲解]]）  
	- `layer3`层  
	- `layer4`层  
	- 接着是 `AdaptiveAvgPool2d` 层和 `fc` 层。  
- 最后是网络参数的初始：  
	- 卷积层采用 `kaiming_normal_()` 初始化方法。
	- `bn` 层和 `GroupNorm` 层初始化为 `weight=1`，`bias=0`。
	- 其中每个 `BasicBlock` 和 `Bottleneck` 的最后一层 `bn` 的 `weight=0`，可以提升准确率 0.2~0.3%。

#### `forward()`
结构十分简单，因为每一层基本都包装完整，只要向前调用即可，代码如下
```python
def _forward_impl(self, x: Tensor) -> Tensor:  
    # See note [TorchScript super()]  
    x = self.conv1(x)  
    x = self.bn1(x)  
    x = self.relu(x)  
    x = self.maxpool(x)  
  
    x = self.layer1(x)  
    x = self.layer2(x)  
    x = self.layer3(x)  
    x = self.layer4(x)  
  
    x = self.avgpool(x)  
    x = torch.flatten(x, 1)  
    x = self.fc(x)  
  
    return x
```

#### `_make_layer()`
具体代码如下 ^181453
```python
def _make_layer(  
    self,  
    block: Type[Union[BasicBlock, Bottleneck]],  
    planes: int,  
    blocks: int,  
    stride: int = 1,  
    dilate: bool = False,  
) -> nn.Sequential:  
    norm_layer = self._norm_layer  
    downsample = None  
    previous_dilation = self.dilation  
    if dilate:  
        self.dilation *= stride  
        stride = 1  
    if stride != 1 or self.inplanes != planes * block.expansion:  
        downsample = nn.Sequential(  
            conv1x1(self.inplanes, planes * block.expansion, stride),  
            norm_layer(planes * block.expansion),  
        )  
  
    layers = []  
    layers.append(  
        block(  
            self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer  
        )  
    )  
    self.inplanes = planes * block.expansion  
    for _ in range(1, blocks):  
        layers.append(  
            block(  
                self.inplanes,  
                planes,  
                groups=self.groups,  
                base_width=self.base_width,  
                dilation=self.dilation,  
                norm_layer=norm_layer,  
            )  
        )  
  
    return nn.Sequential(*layers)
```
- `block`：每个 `layer` 里面使用的 `block`，可以是 `BasicBlock`，`Bottleneck`。
- `planes`：输出的通道数
- `blocks`：一个整数，表示该层 `layer` 有多少个 `block`。
- `stride`：第一个 `block` 的卷积层的 `stride`，默认为 1。注意，只有在每个 `layer` 的第一个 `block` 的第一个卷积层使用该参数。
- `dilate`：是否使用孔洞卷积。

主要流程如下：
- 判断孔洞卷积，计算 `previous_dilation` 参数。  
- 判断 `stride` 是否为 1，输入通道和输出通道是否相等。如果这两个条件都不成立，那么表明需要建立一个 1 X 1 的卷积层，来**改变通道数和改变图片大小**。具体是建立 `downsample` 层，包括 `conv1x1 -> norm_layer`。  
- 建立第一个 `block`，把 `downsample` 传给 `block` 作为降采样的层，并且 `stride` 也使用传入的 `stride`（stride=2）。**后面我们会分析 `downsample` 层在 `BasicBlock` 和 `Bottleneck` 中，具体是怎么用的**。  
- 改变通道数`self.inplanes = planes * block.expansion`。  
	- 在 `BasicBlock` 里，`expansion=1`，因此这一步**不会改变通道数**。
	- 在 `Bottleneck` 里，`expansion=4`，因此这一步**会改变通道数**。
- 图片经过第一个 `block`后，就会改变通道数和图片大小。接下来 for 循环添加剩下的 `block`。从第 2 个 `block` 起，输入和输出通道数是相等的，因此就不用传入 `downsample` 和 `stride`（那么 `block` 的 `stride` 默认使用 1，下面我们会分析 `BasicBlock` 和 `Bottleneck` 的源码）

### BasicBlock
#### 构造函数
具体代码如下
```python
expansion: int = 1

def __init__(  
    self,  
    inplanes: int,  
    planes: int,  
    stride: int = 1,  
    downsample: Optional[nn.Module] = None,  
    groups: int = 1,  
    base_width: int = 64,  
    dilation: int = 1,  
    norm_layer: Optional[Callable[..., nn.Module]] = None,  
) -> None:  
    super().__init__()  
    if norm_layer is None:  
        norm_layer = nn.BatchNorm2d  
    if groups != 1 or base_width != 64:  
        raise ValueError("BasicBlock only supports groups=1 and base_width=64")  
    if dilation > 1:  
        raise NotImplementedError("Dilation > 1 not supported in BasicBlock")  
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1  
    self.conv1 = conv3x3(inplanes, planes, stride)  
    self.bn1 = norm_layer(planes)  
    self.relu = nn.ReLU(inplace=True)  
    self.conv2 = conv3x3(planes, planes)  
    self.bn2 = norm_layer(planes)  
    self.downsample = downsample  
    self.stride = stride
```
- `inplanes`：输入通道数。 
- `planes`：输出通道数。  
- `stride`：第一个卷积层的 `stride`。  
- `downsample`：从 `layer` 中传入的 `downsample` 层。  
- `groups`：分组卷积的分组数，使用 1  
- `base_width`：每组卷积的通道数，使用 64  
- `dilation`：孔洞卷积，为 1，表示不使用 孔洞卷积

主要流程如下：
- 首先判断是否传入了 `norm_layer` 层，如果没有，则使用 `BatchNorm2d`。
- 校验参数：`groups == 1`，`base_width == 64`，`dilation == 1`。也就是说，在 `BasicBlock` 中，不使用孔洞卷积和分组卷积。
- 定义第 1 组 `conv3x3 -> norm_layer -> relu`，这里使用传入的 `stride` 和 `inplanes`。（**如果是 `layer2` ，`layer3` ，`layer4` 里的第一个 `BasicBlock`，那么 `stride=2`，这里会降采样和改变通道数**）。
- 定义第 2 组 `conv3x3 -> norm_layer`，这里不使用传入的 `stride` （默认为 1），输入通道数和输出通道数使用`planes`，也就是**不需要降采样和改变通道数**。
#### forward()
具体代码如下
```python
def forward(self, x: Tensor) -> Tensor:  
    identity = x  
  
    out = self.conv1(x)  
    out = self.bn1(out)  
    out = self.relu(out)  
  
    out = self.conv2(out)  
    out = self.bn2(out)  
  
    if self.downsample is not None:  
        identity = self.downsample(x)  
  
    out += identity  
    out = self.relu(out)  
  
    return out
```

`forward()` 方法的主要流程如下：
- `x` 赋值给 `identity`，用于后面的 `shortcut` 连接。
- `x` 经过第 1 组 `conv3x3 -> norm_layer -> relu`，如果是 `layer2` ，`layer3` ，`layer4` 里的第一个 `BasicBlock`，那么 `stride=2`，第一个卷积层会降采样。
- `x` 经过第 1 组 `conv3x3 -> norm_layer`，得到 `out`。
- 如果是 `layer2` ，`layer3` ，`layer4` 里的第一个 `BasicBlock`，那么 `downsample` 不为空，会经过 `downsample` 层，得到 `identity`。
- 最后将 `identity` 和 `out` 相加，经过 `relu` ，得到输出。

> 注意，2 个卷积层都需要经过 `relu` 层，但它们使用的是同一个 `relu` 层。
### Bottleneck
#### 构造函数
具体代码如下
```python
class Bottleneck(nn.Module):   
    expansion: int = 4  
  
    def __init__(  
        self,  
        inplanes: int,  
        planes: int,  
        stride: int = 1,  
        downsample: Optional[nn.Module] = None,  
        groups: int = 1,  
        base_width: int = 64,  
        dilation: int = 1,  
        norm_layer: Optional[Callable[..., nn.Module]] = None,  
    ) -> None:  
        super().__init__()  
        if norm_layer is None:  
            norm_layer = nn.BatchNorm2d  
        width = int(planes * (base_width / 64.0)) * groups  
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1  
        self.conv1 = conv1x1(inplanes, width)  
        self.bn1 = norm_layer(width)  
        self.conv2 = conv3x3(width, width, stride, groups, dilation)  
        self.bn2 = norm_layer(width)  
        self.conv3 = conv1x1(width, planes * self.expansion)  
        self.bn3 = norm_layer(planes * self.expansion)  
        self.relu = nn.ReLU(inplace=True)  
        self.downsample = downsample  
        self.stride = stride
```
- `inplanes`：输入通道数。
- `planes`：输出通道数。
- `stride`：第一个卷积层的 `stride`。
- `downsample`：从 `layer` 中传入的 `downsample` 层。
- `groups`：分组卷积的分组数，使用 1
- `base_width`：每组卷积的通道数，使用 64
- `dilation`：孔洞卷积，为 1，表示不使用 孔洞卷积

主要流程如下：
- 首先判断是否传入了 `norm_layer` 层，如果没有，则使用 `BatchNorm2d`。
- 计算 `width`，等于传入的 `planes`，用于中间的 $3\times3$ 卷积。
- 定义第 1 组 `conv1x1 -> norm_layer`，这里不使用传入的 `stride`，使用 `width`，作用是进行降维，减少通道数。
- 定义第 2 组 `conv3x3 -> norm_layer`，这里使用传入的 `stride`，输入通道数和输出通道数使用`width`。（**如果是 `layer2` ，`layer3` ，`layer4` 里的第一个 `Bottleneck`，那么 `stride=2`，这里会降采样**）。
- 定义第 3 组 `conv1x1 -> norm_layer`，这里不使用传入的 `stride`，使用 `planes * self.expansion`，作用是进行升维，增加通道数。

#### forward()
代码如下
```python
def forward(self, x: Tensor) -> Tensor:  
    identity = x  
  
    out = self.conv1(x)  
    out = self.bn1(out)  
    out = self.relu(out)  
  
    out = self.conv2(out)  
    out = self.bn2(out)  
    out = self.relu(out)  
  
    out = self.conv3(out)  
    out = self.bn3(out)  
  
    if self.downsample is not None:  
        identity = self.downsample(x)  
  
    out += identity  
    out = self.relu(out)  
  
    return out
```

`forward()` 方法的主要流程如下：
- `x` 赋值给 `identity`，用于后面的 `shortcut` 连接。
- `x` 经过第 1 组 `conv1x1 -> norm_layer -> relu`，作用是进行降维，减少通道数。
- `x` 经过第 2 组 `conv3x3 -> norm_layer -> relu`。如果是 `layer2` ，`layer3` ，`layer4` 里的第一个 `Bottleneck`，那么 `stride=2`，第一个卷积层会降采样。
- `x` 经过第 1 组 `conv1x1 -> norm_layer -> relu`，作用是进行降维，减少通道数。
- 如果是 `layer2` ，`layer3` ，`layer4` 里的第一个 `Bottleneck`，那么 `downsample` 不为空，会经过 `downsample` 层，得到 `identity`。
- 最后将 `identity` 和 `out` 相加，经过 `relu` ，得到输出。

> 注意，3 个卷积层都需要经过 `relu` 层，但它们使用的是同一个 `relu` 层。

### 总结
- `BasicBlock` 中有 1 个 $3\times3$ 卷积层，如果是 `layer` 的第一个 `BasicBlock`，那么第一个卷积层的 `stride=2`，作用是进行降采样。
- `Bottleneck` 中有 2 个 $1\times1$ 卷积层， 1 个 $3\times3$ 卷积层。先经过第 1 个 $1\times1$ 卷积层，进行降维，然后经过 $3\times3$ 卷积层（如果是 `layer` 的第一个 `Bottleneck`，那么 $3\times3$ 卷积层的 `stride=2`，作用是进行降采样），最后经过 $1\times1$ 卷积层，进行升维 。

