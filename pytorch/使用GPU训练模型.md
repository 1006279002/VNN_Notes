数据运算的时候，**数据和模型需要放在同一个设备上**，可以利用`to()`来切换设备/数据类型

* 从CPU到GPU
```python
device=torch.device("cuda")
tensor=tensor.to(device)
module.to(device)
```
* 从GPU到CPU
```python
device=torch.device("cpu")
tensor=tensor.to(device)
module.to(device)
```

`tensor`和`module`的`to()`区别是，`tensor.to()`执行的不是inplace操作，`module.to()`执行的是inplace操作

### `tensor.to()`和`module.to()`
先导入库
```python
import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

然后执行tensor切换操作
```python
x_cpu = torch.ones((3, 3))
print("x_cpu:\ndevice: {} is_cuda: {} id: {}".format(x_cpu.device, x_cpu.is_cuda, id(x_cpu)))

x_gpu = x_cpu.to(device)
print("x_gpu:\ndevice: {} is_cuda: {} id: {}".format(x_gpu.device, x_gpu.is_cuda, id(x_gpu)))
```

输出如下
```text
x_cpu:
device: cpu is_cuda: False id: 2376668664352
x_gpu:
device: cuda:0 is_cuda: True id: 2376668665952
```
可见地址不同，`tensor.to()`并非inplace操作

下面是module切换操作
```python
net = nn.Sequential(nn.Linear(3, 3))

print("\nid:{} is_cuda: {}".format(id(net), next(net.parameters()).is_cuda))

net.to(device)
print("\nid:{} is_cuda: {}".format(id(net), next(net.parameters()).is_cuda))
```

输出如下
```text
id:1599471644896 is_cuda: False

id:1599471644896 is_cuda: True
```

### `torch.cuda`常用方法
- `torch.cuda.device_count()`：返回当前可见可用的 GPU 数量
- `torch.cuda.get_device_name()`：获取 GPU 名称
- `torch.cuda.manual_seed()`：为当前 GPU 设置随机种子
- `torch.cuda.manual_seed_all()`：为所有可见 GPU 设置随机种子
- `torch.cuda.set_device()`：设置主 GPU 为哪一个物理 GPU，此方法不推荐使用
- `os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2", "3")`：设置可见 GPU

在 PyTorch 中，有**物理 GPU** 可以**逻辑 GPU** 之分，可以设置它们之间的对应关系

通常默认`gpu0`是主GPU

### 多GPU的分发并行
```python
torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
# 包装模型，实现分发并行机制，将数据平均分发到各GPU上
```
每个GPU实际数据量为$\frac{batch\_size}{GPU数量}$

- `module`：需要包装分发的模型
- `device_ids`：可分发的 GPU，默认分发到所有可见可用的 GPU
- `output_device`：结果输出设备

注意：使用`DataParallel`的时候，`device`要指定某个GPU为主GPU，否则会报错

这是因为，使用多 GPU 需要有一个主 GPU，来把每个 batch 的数据分发到每个 GPU，并从每个 GPU 收集计算好的结果。如果不指定主 GPU，那么数据就直接分发到每个 GPU，会造成有些数据在某个 GPU，而另一部分数据在其他 GPU，计算出错。

建议分发并行在非Windows平台

`nvidia-smi -q -d Memory`是查询所有 GPU 的内存信息，`-q`表示查询，`-d`是指定查询的内容

如果采用动态设置主GPU，一般是采用最大剩余内存的GPU

### 提高GPU利用率
`nvidia-smi`命令查看GPU的利用率
```text
Thu May  8 09:46:08 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 576.02                 Driver Version: 576.02         CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4060 ...  WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   51C    P8              6W /   42W |       0MiB /   8188MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            8384    C+G   ...ram Files\Tencent\QQNT\QQ.exe      N/A      |
|    0   N/A  N/A           20496      C   ...grams\LM Studio\LM Studio.exe      N/A      |
+-----------------------------------------------------------------------------------------+
```

使用 GPU 训练模型，需要尽量提高 GPU 的 `Memory Usage` 和 `Volatile GPU-Util` 这两个指标，可以更进一步加速你的训练过程

#### Memory Usage
这个指标是由数据量主要是由模型大小，以及数据量的大小决定的。

模型大小是由网络的参数和网络结构决定的，模型越大，训练反而越慢。

我们主要调整的是每个 batch 训练的数据量的大小，也就是 **batch_size**。

在模型**结构固定**的情况下，尽量将`batch size`设置得比较大，充分利用 GPU 的内存。

#### Volatile GPU-Util
设置比较大的 `batch size`可以提高 GPU 的内存使用率，却不一定能提高 GPU 运算单元的使用率

如果`batch size`得比较大，那么在 `Dataset`和 `DataLoader` ，CPU 处理一个 batch 的数据就会很慢，这时你会发现`Volatile GPU-Util`的值会在 `0%，20%，70%，95%，0%` 之间不断变化

`nvidia-smi`命令查看可以GPU的利用率，但不能动态刷新，如果像每隔一秒刷新GPU信息，可以使用`watch -n 1 nvidia-smi`

解决方法是：设置 `Dataloader`的两个参数：
- `num_workers`：默认只使用一个 CPU 读取和处理数据。可以设置为 4、8、16 等参数。但线程数**并不是越大越好**。因为，多核处理需要把数据分发到每个 CPU，处理完成后需要从多个 CPU 收集数据，这个过程也是需要时间的。如果设置`num_workers`过大，分发和收集数据等操作占用了太多时间，反而会降低效率。
- `pin_memory`：如果内存较大，**建议设置为 True**。
	- 设置为 True，表示把数据直接映射到 GPU 的相关内存块上，省掉了一点数据传输时间。
	- 设置为 False，表示从 CPU 传入到缓存 RAM 里面，再给传输到 GPU 上。

### GPU的相关报错
#### 1.

如果模型是在 GPU 上保存的，在无 GPU 设备上加载模型时`torch.load(path_state_dict)`,会出现下面的报错：

```text
RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False. If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.
```

可能的原因：gpu 训练的模型保存后，在无 gpu 设备上无法直接加载。解决方法是设置`map_location="cpu"`：`torch.load(path_state_dict, map_location="cpu")`

#### 2.

如果模型经过`net = nn.DataParallel(net)`包装后，那么所有网络层的名称前面都会加上`mmodule.`。保存模型后再次加载时没有使用`nn.DataParallel()`包装，就会加载失败，因为`state_dict`中参数的名称对应不上。

```text
Missing key(s) in state_dict: xxxxxxxxxx

Unexpected key(s) in state_dict:xxxxxxxxxx
```

解决方法是加载参数后，遍历 state_dict 的参数，如果名字是以`module.`开头，则去掉`module.`。代码如下：

```text
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    namekey = k[7:] if k.startswith('module.') else k
    new_state_dict[namekey] = v
```

然后再把参数加载到模型中。

