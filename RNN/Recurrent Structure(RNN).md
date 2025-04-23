反复使用**同样的结构**(代码复用)

一般用$h$来表示**隐藏层**的输入输出，$y$来表示**结果向量**，$x$表示**输入向量**

## RNN类型介绍
### 常规RNN
给定一个函数$f:h',y=f(h,x)$，这个函数会被反复被执行使用
![Recurrent Structure示意图](../Excalidraw/Recurrent_Structure)
无论输入/输出如何结构(长度)，都由且仅由一个函数$f$来完成

### Deep RNN
深度RNN就是函数叠加，存在多个函数$h',y=f_1(h,x);b',c=f_2(b,y);...$
![Deep RNN示意图](../Excalidraw/Deep_RNN)

### Bi-direcitonal RNN
输入相同，输出是通过两个逆向RNN链的输出再进行一次运算得到的，如存在$h',a=f_1(h,x);b',c=f_2(b,x)$，则如下图所示
![Bi-directional_RNN示意图](../Excalidraw/Bi-directional_RNN)

### Pyramidal RNN
一种平行设计，使输入和输出的**隐藏层**互相互补，从而只需要输入向量$x$就可以得到最终的结果向量
## RNN神经元设计
### 常规设计方式
按照以下公式来设计$$\begin{cases}h'=\sigma(W^hh+W^ix+b^h)\\y=\sigma(W^oh'+b^y)\end{cases}$$
但是RNN一般都不常用了，流行的结构变成了[[Long-Short Term Memory(LSTM)]]，还存在一个[[Gative Recurrent Unit(GRU)]]

使用unidirectional RNN的时候可以将输入和实际情况进行一个delay，例如下图，这样可以更便于计算机进行学习
![示意图](../Excalidraw/unidirectional示意图)

### Stack RNN
其隐藏层可以**十分的长**(本来会增加参数数量降低程序运行效率)，通过这个RNN可以输出四个ouput，分别为**需要存储的信息**，**push**，**pop**和**nothing**，同时进行三个向量的运算，分别是
* $v_1$在头部push进存储的信息
* $v_2$在头部pop出最顶层的信息
* $v_3$不做任何变化

然后再通过如下公式计算出新的向量作为下一阶段的隐藏层$$v=v_1\times push +v_2\times pop+v_3\times nothing$$
