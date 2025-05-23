### RNN Conditional Generation
**条件生成**就是按照**特定的条件**去生成一系列具有**特定结构**的连续结构，其存在一定的现实意义。
* 语言生成
* 图片生成
* 视频生成

比方说LM，通过给定一个\<BOS\>标志来启动RNN，然后就可以生成出一个**特定结构**的连续内容。也可以用这种方式来生成**图片**(将像素变成单词来进行LM训练)

也可以利用[grid LSTM](高速网络和图格LSTM.md#^b3d7a3)来生成深度和时间相关的**图片**

可以通过Encoder和Decoder设计，即输入一句话，通过NN编码后再给NN解码，起到**翻译**的效果，是**句子到句子**的训练方式

### Attention 注意力机制
首先，先将**每个单词**都去对应一个**向量**，得到**向量组**$h$用来表示其含义，然后通过一个规定的$z^0$来对这些向量进行一个**match**操作(没有具体规定，按照实际情况)，得到的结果用向量$a_0$表示，然后通过softmax激活函数得到新的向量$\hat{a_0}$，通过这个新的向量去计算出一个向量$c^0=h\hat{a_0}$。这样就可以把$c^0$作为一个**encoder结果**给下面的NN。然后通过$c^0$在**decoder**中计算出来的$z^1$去进行下一次的操作。

说白就是**特化注意**输入中的一部分，通过这一部分来对后续训练进行特化

### Memory Network
典型例子是给机器看一些**文件**，然后再去问具体的问题，看机器能否给出具体的正确答案

先将文件按照**句子**进行划分和**向量赋值**得到向量组$X$，然后将所问的**问题**作为一个向量去进行match操作，得到具体的attention值$\alpha$后，计算输入向量$X\alpha$，然后再经DNN得到最终的answer结果 

同时也可以把句子设计为两个不同的向量组，一个用于计算attention值，一个用于计算Decoder的输入

### Neural  Turing Machine
不仅可以读Memory，还可以对Memory进行**写操作**，修改Memory内容

现在存在一组memory，然后还有一组初始的attention参数，这样就可以计算出一个向量记为$r^0$，将这个向量投入NN进行计算，得到三个新的向量$k^1,e^1,a^1$，$k^1$用于产生新的attention值，$e^1$用于减少memory，$a^1$用于增加memory

### 生成技巧
因为attention设计的时候可能会导致attention值在**同一块区域**积累，从而导致训练有问题，所以引入一个$\tau$来进行限制，使每一个attention和都接近这个$\tau$值来提高学习准确率，即最小化下面的表达式$$\sum_i(\tau-\sum_ta_t^i)^2$$

训练和测试的时候存在**mismatch**，因为生成的时候有可能进入从来都没有测试过的领域，导致所有数据全部崩盘

那么为了在测试中经可能覆盖所有的测试点，通过scheduled sampling的方式，即在**训练阶段**的选择下一阶段输入时给不同的结果一个**概率**，让它们依概率进入学习

### Beam Search

^4fe874

只保留**部分**可能分数最高的路径，部分的大小由**beam**决定。第一次投入得到多个结果，然后按照beam选择beam大小的几个参数分别多次投入输入得到多个结果，再选出beam个最大参数再投入，以此类推进行训练。

