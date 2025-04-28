简单实例就是**语音识别**，输入的声波向量和输出的文字结果的数量一般来讲不会一一对应。输入和输出都是sequence，同时输出长度由模型决定，这样就是seq2seq的实际含义

只要存在语音和中文的**资料对应**，就可以直接训练一个语音翻译模型，将听到的语音直接翻译成中文

大部分的Natural Language Processing问题都是QA(Question Answering)问题，那么其实就是把`question`和`context`作为输入，将`answer`作为输出的一个seq2seq模型\[[1](https://arxiv.org/pdf/1806.08730)\]\[[2](https://arxiv.org/pdf/1909.03329)\]

seq2seq是可以**硬解**一些问题，比方说分析一个句子的文法，可以讲句子作为输入，文法树的特定结构作为输出(e.g. {S {NP deep learning} {VP is {ADJV very powerful} } })，并且这种处理方式存在[实例](https://arxiv.org/pdf/1412.7449)

通过seq2seq硬做Multi-label Classification任务\[[1](https://arxiv.org/pdf/1909.03434)\]\[[2](https://arxiv.org/pdf/1707.05495)\]

也可以通过seq2seq来做object detection(图片处理)\[[1](https://arxiv.org/pdf/2005.12872)\]

### How to do seq2seq(Transformer)
需要一个encoder和一个decoder来处理这种架构的问题，现在主流操作是[Transformer](https://arxiv.org/pdf/1706.03762)(后文图片来源)

#### Encoder
Encoder使用self-attention来处理，在每个输入进行self-attention操作后，还要进行**residual connection**，即将输入和self-attention的结果进行**加和**，再进行layer normalization(对**一个向量本身**计算均值和标准差，然后再标准化处理)。然后将结果经过一个Fully Connected NN，再进行一次residual connection，再最后layer normalization后就是**Encoder输出**了
![pic6](../data/pic6.png)

#### Autoregressive Decoder
使用**语音识别**作为例子
