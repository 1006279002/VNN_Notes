GRU的外观和RNN基本上一致，只有一个**隐藏层**，但是其内部存在两种门
* 重制门
* 更新门

其内涵的公式包括$$\begin{cases}z_t=\sigma(W_z\cdot[h_{t-1},x_t])&\text{Reset Gate}\\r_t=\sigma(W_r\cdot[h_{t-1},x_t])&\text{Update Gate}\\\tilde{h_t}=tanh(W\cdot[r_t*h_{t-1},x_t])\\h_t=(1-z_t)*h_{t-1}+z_t*\tilde{h_t}\end{cases}$$
示意图如下
![GRU示意图](../Excalidraw/GRU)

运算量比LSTM小，使用参数更少
