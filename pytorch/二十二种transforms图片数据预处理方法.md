主要分几个部分介绍transforms
- 裁剪
- 旋转和翻转
- 图像变换
- transforms 方法操作
- 自定义 transforms 方法

后续介绍的所有操作都是在通过`transforms.Resize((224,224))`**缩放图片**至`(224,224)`大小后进行的，借由**原作者**编写的`transform_invert()`方法将处理后的图片重新变回图片来观察**变化效果**

原图如下所示
![pic10](../data/pic10.jpg)

经过缩放处理可以得到
![svg2](../data/svg2.svg)

### 裁剪
##### transforms.CenterCrop
```python
torchvision.transforms.CenterCrop(size)
# 从图像中心裁剪图片
```
* size是所需裁剪的图片尺寸

`transforms.CenterCrop(196)`效果如下
![svg3](../data/svg3.svg)

如果大小不够就会通过zero padding的方式填充图片，`transforms.CenterCrop(512)`效果如下
![svg4](../data/svg4.svg)

##### transforms.RandomCrop
```python
torchvision.transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')
# 从图片中随机裁剪大小为size的图片，如果有padding就先padding再裁剪
```
- size: 裁剪大小
- padding: 设置填充大小
	- 当为 a 时，上下左右均填充 a 个像素
	- 当为 (a, b) 时，左右填充 a 个像素，上下填充 b 个像素
	- 当为 (a, b, c, d) 时，左上右下分别填充 a，b，c，d 个像素
- pad_if_need: 当图片小于设置的 size，是否填充
- padding_mode:
	- constant: 像素值由 fill 设定
	- edge: 像素值由图像边缘像素设定
	- reflect: 镜像填充，最后一个像素不镜像。(\[1,2,3,4] -> \[3,2,1,2,3,4,3,2])
	- symmetric: 镜像填充，最后一个像素也镜像。(\[1,2,3,4] -> \[2,1,1,2,3,4,4,4,3])
- fill: 当 padding_mode 为 constant 时，设置填充的像素值

下面是几种不同参数导致的不同结果图片
1. `transforms.RandomCrop(224, padding=16)`![svg5](../data/svg5.svg)
2. `transforms.RandomCrop(224, padding=(16, 64))`![svg6](../data/svg6.svg)
3. `transforms.RandomCrop(224, padding=16, fill=(255, 0, 0))`![svg7](../data/svg7.svg)
4. `transforms.RandomCrop(512, pad_if_needed=True)`![svg8](../data/svg8.svg)
5. `transforms.RandomCrop(224, padding=64, padding_mode='edge')`![svg9](../data/svg9.svg)
6. `transforms.RandomCrop(224, padding=64, padding_mode='reflect')`![svg10](../data/svg10.svg)
7. `transforms.RandomCrop(1024, padding=1024, padding_mode='symmetric')`![svg11](../data/svg11.svg)

##### transforms.RandomResizedCrop
```python
torchvision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2)
# 随机大小，随机宽高比裁剪图片，根据scale的比例裁剪原图，根据raito的长宽比再裁剪，最后使用插值法将图片变换为size大小
```
- size: 裁剪的图片尺寸
- scale: 随机缩放面积比例，默认随机选取 (0.08, 1) 之间的一个数
- ratio: 随机长宽比，默认随机选取 ($\frac{3}{4}$, $\frac{4}{3}$) 之间的一个数。因为超过这个比例会有明显的失真
- interpolation: 当裁剪出来的图片小于 size 时，就要使用插值方法 resize
	- PIL.Image.NEAREST
	- PIL.Image.BILINEAR
	- PIL.Image.BICUBIC

以下是两种结果示例
* `transforms.RandomResizedCrop(size=224, scale=(0.08, 1))`(缩放比随机)![svg12](../data/svg12.svg)
* `transforms.RandomResizedCrop(size=224, scale=(0.5, 0.5))`(缩放比固定0.5)![svg13](../data/svg13.svg)

##### transforms.FiveCrop/TenCrop
```python
torchvision.transforms.FiveCrop(size)
torchvision.transforms.TenCrop(size, vertical_flip=False)
# FiveCrops是在图片的上下左右和中间截出大小为size的5张图片
# TenCrop是对这5张图片进行水平/垂直镜像获得10张图片，5张图片的区域不变
```
- size: 最后裁剪的图片尺寸
- vertical_flip: 是否垂直翻转

这两个方法返回的是tuple，所以需要将这些数据转换为tensor，经过处理之后可以获得如下图片
![svg14](../data/svg14.svg)

下面的图是通过`TenCrop`操作得到的
![svg15](../svg15.svg)

### 旋转和翻转
##### transforms.RandomHorizontalFlip/RandomVerticalFlip
根据概率，在水平或者垂直方向翻转图片

* `transforms.RandomHorizontalFlip(p=0.5)`，那么一半的图片会被水平翻转。
* `transforms.RandomHorizontalFlip(p=1)`，那么所有图片会被水平翻转。

* `transforms.RandomHorizontalFlip(p=1)`，水平翻转的效果如下![svg16](../data/svg16.svg)
* `transforms.RandomVerticalFlip(p=1)`，垂直翻转的效果如下![svg17](svg17.svg)

##### transforms.RandomRotation
```python
torchvision.transforms.RandomRotation(degrees, resample=False, expand=False, center=None, fill=None)
# 随机旋转图片
```
- degrees: 旋转角度
	- 当为 a 时，在 (-a, a) 之间随机选择旋转角度
	- 当为 (a, b) 时，在 (a, b) 之间随机选择旋转角度
- resample: 重采样方法
- expand: 是否扩大矩形框，以保持原图信息。根据中心旋转点计算扩大后的图片。**如果旋转点不是中心**，即使设置 expand = True，还是会有**部分信息丢失**。
- center: 旋转点设置，是坐标，默认中心旋转。如设置左上角为：(0, 0)

* `transforms.RandomRotation(90)`的效果如下![svg18](../data/svg18.svg)
* `transforms.RandomRotation((90), expand=True)`的效果如下![svg19](../data/svg19.svg)需要注意如果设置`expand = True`，那么一个batch中的所有图片的shape都会发生变化，会throw error，所以需要进行`resize`操作
* `transforms.RandomRotation(30, center=(0, 0))`的效果如下，设置旋转点为左上角![svg20](../data/svg20.svg)
* `transforms.RandomRotation(30, center=(0, 0), expand=True)`的效果如下，如果旋转点不为中心，那么即使expand也会丢失信息![svg21](../data/svg21.svg)

### 图像变换





