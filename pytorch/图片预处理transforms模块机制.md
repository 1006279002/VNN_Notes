在安装pytorch的同时，还安装了`torchvision`，是一个CV的工具包，重点模块如下
- `torchvision.transforms`: 里面包括常用的图像预处理方法
- `torchvision.datasets`: 里面包括常用数据集如 **mnist**、**CIFAR-10**、**Image-Net** 等
- `torchvision.models`: 里面包括常用的预训练好的模型，如 **AlexNet**、**VGG**、**ResNet**、**GoogleNet** 等

模型训练依靠**数据的数量和分布**，所以通过预处理可以**增强数据**，通过增加数据多样性提高模型的泛化能力

常用的预处理方法主要是以下几种
- 数据中心化
- 数据标准化
- 缩放
- 裁剪
- 旋转
- 翻转
- 填充
- 噪声添加
- 灰度变换
- 线性变换
- 仿射变换
- 亮度、饱和度以及对比度变换

针对[[DataLoader与DataSet#^0be56e|人民币二分实验]]来说，采用了一定的增强
```python
# 设置训练集的数据增强和转化
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),# 缩放
    transforms.RandomCrop(32, padding=4), #裁剪
    transforms.ToTensor(), # 转为张量，同时归一化
    transforms.Normalize(norm_mean, norm_std),# 标准化
])

# 设置验证集的数据增强和转化，不需要 RandomCrop
valid_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
```

如果需要多个transform操作，那就需要将所有的操作作为一个`list`存放在`transforms.Compose`中，需要注意的是`transforms.toTensor()`操作把图片转换为张量，同时进行了**归一化操作**

在`__getitem__()`函数中的`self.transform(img)`会调用`Compose`的`__call__()`函数
```python
def __call__(self, img):
 for t in self.transforms:
  img = t(img)
 return img
```
遍历调用每一个transform对图片进行处理

##### transforms.Normalize
```python
torchvision.transforms.Normalize(mean, std, inplace=False)
# 逐channel对图像进行标准化
```
$$output=\frac{input-mean}{std}$$
- mean: 各通道的均值
- std: 各通道的标准差
- inplace: 是否原地操作

其实就是调用了`F.normalize`，具体函数如下
```python
def normalize(tensor, mean, std, inplace=False):
    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    return tensor
```

首先判断是否为 tensor，如果不是 tensor 则抛出异常。然后根据`inplace`是否为 true 进行 clone，接着把 mean 和 std 都转换为 tensor (原本是 list)，最后减去均值除以方差

让数据变成均值为0，标准差为1的标准化([[../Normalization/批次标准化|批次标准化]])

如果离0均值比较远，收敛速度就会变慢(可以调整[[autograd与逻辑回归#^64ea81|逻辑回归实验]]中的`bias`大小进行验证)
