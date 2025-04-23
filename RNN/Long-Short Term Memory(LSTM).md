LSTM中的参数一般用$h$表示**隐藏层/输出层**，$c$表示**细胞状态层**，$x$表示**输入向量**

其中，$c$的变化速度是比较慢的，相对的$h$的变化速度是十分快的。当然，这两个都相当于隐藏层。

基本LSTM相关的公式如下$$\begin{cases}f_t=\sigma(W_f\cdot[h_{t-1},x_t]+b_f)&\text{Forget Gate}\\ i_t=\sigma(W_i\cdot[h_{t-1},x_t]+b_i)&\text{Input Gate 1}\\\tilde{C_t}=tanh(W_c\cdot[h_{t-1},x_t]+b_f)&\text{Input Gate 2}\\C_t=f_t*C_{t-1}+i_t*\tilde{C_t}&\text{Cell Gate}\\o_t=\sigma(W_o\cdot[h_{t-1},x_t]+b_o)&\text{Output Gate 1}\\h_t=o_t*tanh(C_t)&\text{Output Gate 2}\end{cases}$$
![LSTM示意图](../Excalidraw/LSTM)

