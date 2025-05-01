主要应用场景在于**图像/视频**作为输入，目标为**分类**的时候

对于一张图片，其本质为一个**三维的tensor(张量)**，包括其**长**、**宽**及**频道数(channel)**

最直觉的处理方式是将所有的像素**拉直**，直接作为一个大向量进行处理(一般是fully connected network)，但是这样带来的**参数数量**实在是太多了，会降低其运行速度，更容易产生**过拟合**

设计**Receptive Field**来初步简化，Receptive Field可以重叠，也可以输入到不同的神经元。尽管Receptive Field可以**任意设计**(长宽、重叠与否)，但是存在一种常规的设置方式
* 会看所有的channel
* kernel size一般是正方形设计(e.g. $3\times 3$)
* 每一个Receptive Field存在一组神经元(e.g. 64个神经元)
* Receptive Field通过**stride**大小移动，最好Receptive Field之间存在**重叠**
* 如果超出范围，那么就利用Padding方式进行补值(一般是zero padding)

让不同Receptive Field的神经元**共享参数**，这样就可以在**不同区域**也能检测出一样的内容，并且由于同时间输入的不同防止出结果一直不变，但是同一个Reception Field中的神经元就不要共享参数了，做无用功

CNN的bias是相对比较大的，但是它弹性小，更不容易过拟合，参数也少，计算效率高

也可以换一个理解思路，一张图片可以看成多个$3\times 3 \times channel$的filter，比方说有$n$个，这样的话可以通过多层CNN来进行处理，第一层CNN通过用filter扫描原图片进行**卷积操作**来处理这些filter，形成一个新的$n$ channel图片，第二层则是处理$3\times 3\times n$的新filter，以此类推。

因为图片中的特征在一定程度上的**缩放像素**不会丢失，所以一般还会进行**pooling池化**操作，详情见[[Pooling Layer]]

Pooling可以减少像素但保持channel数，减少运算量(如果运算力度够的话也可以不进行pooling操作)，同时也要考虑**实际应用场景**，比方说在**围棋(Alpha GO)** 的实例中，就不能使用Pooling，不然会在逻辑上存在严重问题

CNN常规流程
* 卷积层
* 池化层
* 卷积层
* 池化层
* Flatten 将矩阵转换为向量
* 全连接层
* softmax激活函数

但是CNN不能处理**旋转/缩放**等问题，这样就需要设计[[空间变换层|空间变换层]]来处理
