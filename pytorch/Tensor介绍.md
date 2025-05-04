### 什么是Tensor
Tensor，即**张量**，是用于衡量标量、向量、矩阵的**高维拓展**，标量就是0维张量，向量是一维张量，矩阵是二维张量，三通道图片是三维张量，以此类推。

### Tensor中的参数
在pytorch中，tensor包含下面几个属性
* data:存放被包装的tensor
* grad:data的梯度
* grad_fn:创建tensor所使用的function，自动求导的关键 ^50db12
* requires_grad:指示是否需要梯度，并非所有张量都需要计算梯度
* is_leaf:指示是否叶子节点，与[[../Computational Graph/计算图|计算图]]有关 ^e44ffe
* dtype:张量的数据类型，如`torch.FloatTensor`，`torch.cuda.FloatTensor`
* shape:张量的形状，例如`(64,3,224,224)`
* device:张量所在的设备(CPU/GPU)，GPU是加速计算的关键

其中，dtype存在9种数据类型，具体如下

| Data Type               | dtype                             | Tensor types           |
| ----------------------- | --------------------------------- | ---------------------- |
| 32-bit floating point   | `torch.float32` or `torch.float`  | `torch.*.FloatTensor`  |
| 64-bit floating point   | `torch.float64` or `torch.double` | `torch.*.DoubleTensor` |
| 16-bit floating point   | `torch.float16` or `torch.half`   | `torch.*.HalfTensor`   |
| 8-bit integer(unsigned) | `torch.uint8`                     | `torch.*.ByteTensor`   |
| 8-bit integer(signed)   | `torch.int8`                      | `torch.*.CharTensor`   |
| 16-bit integer(signed)  | `torch.int16` or `torch.short`    | `torch.*.ShortTensor`  |
| 32-bit integer(signed)  | `torch.int32` or `torch.int`      | `torch.*.IntTensor`    |
| 64-bit integer(signed)  | `torch.int64` or `torch.long`     | `torch.*.LongTensor`   |
| Boolean                 | `torch.bool`                      | `torch.*.BoolTensor`   |

### Tensor的创建方式
#### 直接创建Tensor
##### torch.tensor()
```python
torch.tensor(data,dtype=None,device=None,requires_grid=False,pin_memory=False)
```
* data:数据，可以是list，numpy
* dtype:数据类型，默认和data一致
* device:所在设备，`"cuda"`或者`"cpu"`
* requires_grid:是否需要梯度
* pin_memory:是否存于锁业内存

代码示例
```python
arr = np.ones((3, 3))
print("ndarray的数据类型：", arr.dtype)
# 创建存放在 GPU 的数据
# t = torch.tensor(arr, device='cuda')
t= torch.tensor(arr)
print(t)
```

输出如下
```text
ndarray的数据类型： float64
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
```

##### torch.from_numpy(ndarray)
或者也可以使用`torch.from_numpy(ndarray)`的方式，这样就可以将这个`tensor`和`ndarray`绑定，修改任意一个都会影响另一个

代码示例
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
t = torch.from_numpy(arr)

# 修改 array，tensor 也会被修改
# print("\n修改arr")
# arr[0, 0] = 0
# print("numpy array: ", arr)
# print("tensor : ", t)

# 修改 tensor，array 也会被修改
print("\n修改tensor")
t[0, 0] = -1
print("numpy array: ", arr)
print("tensor : ", t)
```

输出如下
```text
修改tensor
numpy array:  [[-1  2  3]
 [ 4  5  6]]
tensor :  tensor([[-1,  2,  3],
        [ 4,  5,  6]], dtype=torch.int32)
```

#### 根据数值创建Tensor
##### torch.zeros()
```python
torch.zeros(*size,out=None,dtype=None,layout=torch.strided,device=None,requires_grad=False)
```
* size:张量的形状
* out:输出的张量，如果指定了`out`，那么`torch.zeros()`返回的张量和out指向的是同一个地址
* layout:内存中的布局形式，有`strided`和`sparse_coo`等，**稀疏矩阵**使用`sparse_coo`可以减少内存占用
* device:所在设备
* requires_grad:是否需要梯度

代码示例
```python
out_t = torch.tensor([1])
# 这里指定了 out
t = torch.zeros((3, 3), out=out_t)
print(t, '\n', out_t)
# id 是取内存地址。最终 t 和 out_t 是同一个内存地址
print(id(t), id(out_t), id(t) == id(out_t))
```

输出如下
```text
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
 tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
2984903203072 2984903203072 True
```

##### torch.zeros_like
```python
torch.zeros_like(input, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format)
# 依据input形状创建全0张量
```

同理存在全1张量的创建方式`torch.ones()`和`torch.ones_like()`

##### torch.full()及torch.full_like()
```python
torch.full(size, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
# 创建自定义数值的张量
```

代码示例
```python
t = torch.full((3, 3), 1)
print(t)
```

输出如下
```text
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])
```

##### torch.arange()
```python
torch.arange(start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
# 创建等差的一维张量，区间为[start,end)
```
* start:数列起始值
* end:数列结束值，开区间
* step:数列公差，默认为1

代码示例
```python
t = torch.arange(2, 10, 2)
print(t)
```

输出如下
```text
tensor([2, 4, 6, 8])
```

##### torch.linspace()
```python
torch.linspace(start, end, steps=100, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
# 创建均分的1维张量，区间为[start,end]
```
* start:数列起始值
* end:数列结束值
* steps:数列长度(元素个数)

代码示例
```python
# t = torch.linspace(2, 10, 5)
t = torch.linspace(2, 10, 6)
print(t)
```

输出如下
```text
tensor([ 2.0000,  3.6000,  5.2000,  6.8000,  8.4000, 10.0000])
```

##### torch.logspace()
```python
torch.logspace(start, end, steps=100, base=10.0, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
# 创建对数均分的1维张量，数值区间为[start,end]，底为base
```
* start:数列起始值
* end:数列结束值
* steps:数列长度(元素个数)
* base:对数函数的底，默认为10

实际含义如下面公式所示$$(base^{start},base^{start+\frac{end-start}{steps-1}},...,base^{(start+(steps-2)*\frac{end-start}{steps-1})},base^{end})$$

代码示例
```python
# t = torch.logspace(2, 10, 5)
t = torch.logspace(2, 10, 6)
print(t)
```

输出如下
```text
tensor([1.0000e+02, 3.9811e+03, 1.5849e+05, 6.3096e+06, 2.5119e+08, 1.0000e+10])
```

##### torch.eye()
```python
torch.eye(n, m=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
# 创建单位对角矩阵，默认为方阵
```
* n:矩阵行数，通常只设置n，默认方阵
* m:矩阵列数

即生成**单位矩阵**$I$

#### 根据概率创建Tensor
##### tensor.normal()
```python
torch.normal(mean, std, *, generator=None, out=None)
# 生成正态分布
```
* mean:均值
* std:标准差

其中mean和std是tensor形式，可以是零维标量也可是高维

##### torch.randn()和torch.randn_like()
```python
torch.randn(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
# 生成标准正态分布
```
* size:张量的形状

##### torch.rand()和torch.rand_like()
```python
torch.rand(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
# 在区间[0,1)上生成均匀分布
```

##### torch.randint()和torch.randint_like()
```python
randint(low=0, high, size, *, generator=None, out=None,
dtype=None, layout=torch.strided, device=None, requires_grad=False)
# 在区间[low,high)上整数均匀分布
```

##### torch.randperm()
```python
torch.randperm(n, out=None, dtype=torch.int64, layout=torch.strided, device=None, requires_grad=False)
# 生成从0到n-1的随机排列，常用于生成索引
```

##### torch.bernoulli()
```python
torch.bernoulli(input, *, generator=None, out=None)
# 以input为概率，生成伯努利分布(0-1分布，两点分布)
```

