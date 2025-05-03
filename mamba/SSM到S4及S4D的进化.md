### SSM到S4D的升级
#### 离散数据的连续化:基于零阶保持技术的连续化与采样
SSM在**离散数据**上进行训练，仍能学习到离散数据底层的连续信息

但是如果要处理离散数据，可以通过**零阶保持技术**
* 每次收到离散信号就保留其数值，直到收到新的离散信号，这个操作本身就创建了SSM可以使用的连续信号
* 保持值的**时间**用一个新的学习参数**步长**($\Delta$)来表示，代表输入的阶段性保持
* 有了连续的输入信号就可以产生连续的输出

而针对输出的结果，依据步长$\Delta$来进行采样就可以得到**离散的输出**，并针对$A和B$进行一些处理$$\bar{A}=exp(\Delta A)$$$$\bar{B}=(\Delta A)^{-1}(exp(\Delta A)-I)\cdot \Delta B$$
这样就可以得到离散处理函数$$h_k=\bar{A}h_{k-1}+\bar{B}x_k$$$$y_k=Ch_k$$

对这个函数进行**迭代推理**可以得到一些特殊的表达关系式，例如$$\begin{aligned}y_2 &=Ch_2 \\&=C(\bar{A}h_1+\bar{B}x_2)\\&=C(\bar{A}(\bar{A}h_0+\bar{B}x_1)+\bar{B}x_2)\\&=C(\bar{A}(\bar{A}\cdot\bar{B}x_0+\bar{B}x_1)+\bar{B}x_2)\\&=C\cdot\bar{A}^2\cdot\bar{B}x_0+C\cdot\bar{A}\cdot\bar{B}x_1+C\cdot\bar{B}x_2\end{aligned}$$ ^c572f8

保存的时候，仍然保存**连续形式**下的$A$

#### 循环结构表示:方便快速推理
这个时候就可以利用RNN来进行处理了，具体处理内容还是看[[../RNN/Recurrent Structure(RNN)|Recurrent Structure(RNN)]]

#### 卷积结构表示:方便并行训练
一样是利用[[../CNN/Convolutional Layer#^13a8b6|filter]]的思想，将**一维**的输入文本卷积处理$$\bar{K}=(C\bar{B},C\bar{A}\bar{B},...,C\bar{A}^k\bar{B},...)$$$$y=x\cdot\bar{K}$$
* $x$是文本输入
* $y$是卷积结果输出
* $\bar{K}$就是卷积核Kernel，其参数$k$的大小由输入维度决定

其中要对$x$适当进行padding操作，padding的核心是**输入输出以及Kernel的维度都是一样的**，这个操作是通过[[#^c572f8|循环结构推理]]得到最终结果得出的，这样就可以将SSM直接表示为一种卷积操作。但是这样的推理速度不如RNN快。

SMMs可以**训练通过CNN，推理使用RNN**，来做到两全其美的效果

#### 长距离依赖问题的解决之道:从HiPPO
由上面的推导可以知道，隐藏状态是依赖矩阵$A$进行更新的，所以如果要保留比较长的memory，就需要对矩阵$A$进行动手脚

[HiPPO](https://proceedings.neurips.cc/paper_files/paper/2020/file/102f0bb6efb3a6128a3c750dd16729be-Paper.pdf)尝试将所有输入信号压缩为一个**系数向量**，建立了下面的一个HiPPO矩阵$$A_{nk}=-\begin{cases}(2n+1)^{1/2}(2k+1)^{1/2}&\text{if n > k}\\n+1&\text{if n = k}\\0&\text{if n < k}\end{cases}$$
可以产生一个隐藏状态来记住其历史

---
介绍完了SSM，那么就要总结一下其最大的，急需解决的问题

**矩阵不会随着输入不同而发生变化，无法针对输入进行针对性推理**

SSM规定，在test的时候，矩阵$A,B,C$不会发生任何的变化(训练的时候依然通过梯度下降学习变化)

所以，对此可以利用**选择性聚焦**来处理，即给不同的单词不同的权重，类似attention，但这样又会导致无法通过CNN的方式进行训练，因为破坏了卷积核递推公式
