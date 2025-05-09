TensorBoard是TensorFlow中强大的可视化工具，支持多种数据的可视化

PyTorch通过TensorboardX来调用，先将数据保存到event file中，然后再使用TensorBoardX读取event file展示至网页

下面是如何保存event file的代码示例
```python
    import numpy as np
    import matplotlib.pyplot as plt
    from tensorboardX import SummaryWriter
    from common_tools import set_seed
    max_epoch = 100

    writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")

    for x in range(max_epoch):

        writer.add_scalar('y=2x', x * 2, x)
        writer.add_scalar('y=pow_2_x', 2 ** x, x)

        writer.add_scalars('data/scalar_group', {"xsinx": x * np.sin(x),
                                                 "xcosx": x * np.cos(x)}, x)

    writer.close()
```



