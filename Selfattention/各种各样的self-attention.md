因为输入的sequence长度$N$大到一定程度之后，Attention Matrix的大小会十分庞大，其运算效率会十分低，所以针对这个问题，展开研究了许多不同类型的attention

一般都是用在**影像处理**上，因为一般输入的$N$会十分巨大

### 利用人类认知跳过部分计算
#### Local Attention/Truncated Attention
只考虑**相邻向量**的attention关系，较远的向量默认设为0来进行计算，近似于CNN
#### Stride Attention
人为设计stride大小，跳过stride个数的向量，去和一定距离外的向量进行attention操作
#### Global Attention
在原sequence中添加一些**特别的token**，这些token会与所有token进行attention计算，同时**所有token**也会对**这些token**进行计算。可以直接将原数据中的某些token直接变成special token，也可以外加一些token，将这些外加的token作为special token。非special token的其他token互相不影响。

一般使用multiple head，不同head做不同的方式，参考文献[Longformer](https://arxiv.org/pdf/2004.05150)[Big Bird](https://arxiv.org/pdf/2007.14062)
### 关注重要部分
attention matrix中可能会有一些很大或很小的数据，将很小的数据**直接置为0**，在减少影响的同时也能加快计算
#### Clustering
将所有的query和key向量进行一个cluster操作，将**相近的向量**分类成同一组，近似且快速。只计算相同cluster中的向量attention，其余的attention值直接设置为0。\[[1](https://arxiv.org/pdf/2003.05997)\]\[[2](https://openreview.net/forum?id=rkgNKkHtvB)\]
#### Sinkhorn Sorting Network
通过**学习模型**来学习具体要着重计算哪些attention，其余部分直接置0。通过输入一个sequence，经过一个NN处理后得到一个大小和query-key矩阵一样的0/1矩阵，用来标记着重学习attention的位置\[[1](https://arxiv.org/pdf/2002.11296)\]
### 真的需要一个完整的attention matrix吗
重要参考文献[Linformer](https://arxiv.org/pdf/2006.04768)

一个完整的attention matrix当中可能会存在许多的**重复内容**(redundant columns)，这样会使其计算很多无用内容。所以如果提前将这些会产生重复内容的部分给拿走的话，就会极大程度地减少实际的计算量。

本身存在的$N$个key中选择$K$个具有代表性的key，同样在计算attention的时候也只选择$K$个具有代表性的value，最好不去削减query的个数

关于挑选代表性的key来处理
* 可以使用CNN的方式来挑选
* 也可以将$d\times N$的key矩阵乘上一个$N\times K$的一个矩阵，从而获得一个$d\times K$的目标矩阵，从而找出$K$个具有代表性的向量
### 矩阵计算的运算加速
[复习一下self-attention的计算过程](自注意力机制#^0d13be)

虽然上面的计算过程中是先计算Attention Matrix进行计算，但是这样的计算速度其实是不快的，如果抛开**激活函数**的过程，将原先的计算顺序$O=V(K^TQ)$转换为$O=(VK^T)Q$，这样就可以加快矩阵乘法的运算速度。

那如果把激活函数放回去呢？

以输出$b^1$作为例子举例，下面是加入softmax的计算表达式$$b^1=\sum_{i=1}^Na_{1,i}'v^i=\sum_{i=1}^N\frac{exp(q^1\cdot k^i)}{\sum_{j=1}^Nexp(q^1\cdot k^j)}v^i$$引入一个**假设的函数关系**$$exp(q\cdot k)\approx \Phi(q)\cdot\Phi(k)$$
这样就可以把上面的式子改写为(假定$q^1$的维度为$M$)$$\begin{aligned}b^1&=\sum_{i=1}^N\frac{\Phi(q^1)\cdot\Phi(k^i)}{\sum_{j=1}^N\Phi(q^1)\cdot\Phi(k^j)}v^i\\&=\frac{T\cdot\Phi(q^1)}{\sum_{j=1}^N\Phi(k^j)\cdot \Phi(q^1)} \end{aligned}$$其中矩阵$T$是一个列数为$M$，并且其每一列**向量和**是$\sum_{j=1}^N\Phi(k_i^j)v^j$，其中$i$指代的是第$i$列 ^20bc16

通过上面的公式可以得出，每一个结果之和query向量存在关系，剩下的内容都是**重复内容**

所以就可以把self-attention过程直接进行变形修改，将每一个$k$都去做一个$\Phi(k)$处理，然后将$\Phi(k)$中的每一维元素都去和$v$向量进行处理，[[#^20bc16|如上文公式所示]]。然后再计算每个向量对应的$\Phi(q)$，依次计算最终结果

那么如何将$exp(q\cdot k)$转换呢？不同文章存在不同的处理方法
* [Efficient attention](https://arxiv.org/pdf/1812.01243)
* [Linear Transformer](https://linear-transformers.com/)
* [Random Feature Attention](https://arxiv.org/pdf/2103.02143)
* [Performer](https://arxiv.org/pdf/2009.14794)
### 真的一定要用$q$和$k$来处理问题吗？
[Synthesizer](https://arxiv.org/pdf/2005.00743)表示不需要，虽然最终的输出结果也是通过attention matrix和$v$向量来计算得出，但是attention matrix却不是通过$q和k$计算得出的，而是让其变成**神经网络的参数**，通过训练产生，永远都是一样的
### 一定要用Attention吗？
有些人使用MLPs来进行处理
* [Fnet: Mixing tokens with fourier transforms](https://arxiv.org/pdf/2105.03824)
* [Pay Attention to MLPs](https://arxiv.org/pdf/2105.08050)
* [MLP-Mixer:An all-MLP Architecture for Vision](https://arxiv.org/pdf/2105.01601)

---
最后，对各种各样的进行比对和研究的论文在[这里](https://arxiv.org/pdf/2011.04006)和[这里](https://arxiv.org/pdf/2009.06732)，方便后续学习
