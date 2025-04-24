已知，对于一个神经网络来说，其表达式可以写成$$y=\sigma(W^L...\sigma(W^2\sigma(W^1x+b^1)+b^2)...+b^L)$$
向量之间的**偏微分**得到Jacobian矩阵，举例$y=f(x)$，其中$x=\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix},y=\begin{bmatrix}y_1\\y_2\end{bmatrix}$
这样就可以得到其偏微分结果为$$\frac{\partial y}{\partial x}=\begin{bmatrix}\partial y_1/\partial x_1 &\partial y_1/\partial x_2&\partial y_1/\partial x_3\\\partial y_2/\partial x_1&\partial y_2/\partial x_2&\partial y_2/\partial x_3\end{bmatrix}$$

下面对下图进行每个偏微分的计算
![神经网络计算图](../Excalidraw/神经网络计算图)
对选择问题(即$\hat{y}$只有一个值是1，别的值都是0)来说，**交叉熵**计算:$C=-log(\hat{y}^Ty)$，则可以知道$$\frac{\partial C}{\partial y}=\hat{y}^T(-y^{-1})$$

下一步是判断$\frac{\partial y}{\partial z^2}$，由于仅进行了激活函数操作(非softmax等多值影响)，所以可以知道只有对角线上存在含义，所以可以得出$$\frac{\partial y}{\partial z^2}=E\cdot \sigma'(z^2)$$

由定义可知，$$\frac{\partial z^2_i}{\partial a^1_j}=W^2_{ij}$$即$\frac{\partial z^2}{\partial a^1}=W^2$

那么$\frac{\partial z^2}{\partial W^2}$如何计算呢？将$W^2$看作一个$m\times n$的一个**向量**，那就可以得到一个$m\times (m\times n)$的二维矩阵，同时可以知道，对$\frac{\partial z^2_i}{\partial W^2_{jk}}$进行分析，如果$i\not = j$则$\frac{\partial z^2_i}{\partial W^2_{jk}}=0$；如果$i=j$则$\frac{\partial z^2_i}{\partial W^2_{jk}}=a_k^1$

以此类推，可以得出所有的偏微分数值，例如$$\frac{\partial C}{\partial W^1}=\frac{\partial C}{\partial y}\frac{\partial y}{\partial z^2}W^2\frac{\partial a^1}{\partial z^1}\frac{\partial z^1}{\partial W^1}$$就可以得出$W^1$中每一个数据对$C$的偏微分
