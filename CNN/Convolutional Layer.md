即**卷积层**，其特性如下
* **部分连接性质**，每个神经元并不会连接所有的上一层神经元，只会连接部分(人工设定，称为**Receptive Field**)
* **参数分享性质**，不同Receptive Field的神经元可以设置一样的参数(可以强制设定)

可以减少大量的参数来完成任务

Reception Field**一致**的神经元集合被称为filter或者kernel，而filter/kernel size就是指reception field中神经元的个数，而神经元的**间隔数**被称为Stride。==filter size=3的reception field其Stride=2== ^13a8b6

* 1D Single Channel
	* 直接按照**时间**分类，分割kernel去做不同的事
* 1D Multiple Channel
	* 每一个参数用多个信号去描述，例如一个**词**用一个**向量**去描述
	* 向量的不同维度就代表不同的channel
	* 仍然按照**时间**分类
* 2D Single Channel
	* 利用一个二维的Reception Field去划分kernel，而Stride一般是1
	* 一般是**灰度单图片**处理
* 2D Multiple Channel
	* 一般是**RGB彩色图片**处理
	* 举例一张$6\times 6$的彩色图片
	* 这样的话就设计$3\times 3\times 3$的Reception Field以Stride=1的方式去训练

一般在**卷积处理**的时候，防止图片越处理越小，一般通过**zero padding**(补零)的方式去补充图片，例如将一个$6\times 6$的原矩阵补成$8\times 8$的矩阵，这样在$3\times 3$的Reception Field处理之后，矩阵仍然是$6\times 6$的大小
