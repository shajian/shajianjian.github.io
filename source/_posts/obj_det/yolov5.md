---
title: YOLOv5 解读
date: 2023-08-09 16:48:30
tags: object detection
mathjax: true
---

源码：[ultralytics/yolov5](https://github.com/ultralytics/yolov5.git)

**# 模型结构文件**

有 5 个不同大小的模型，这 5 模型主要就是深度缩放因子和宽度缩放因子不同，影响了模型的深度（layers number）和宽度（channel size）。

以 yolov5l 为例，配置文件为

```sh
nc: 80  # number of classes
depth_multiple: 1.0  # model depth multiple
width_multiple: 1.0  # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 9
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```

各个模型文件的深度缩放因子和宽度缩放因子如下：

```sh
# yolov5x
depth_multiple: 1.33
width_multiple: 1.25
# yolov5l
depth_multiple: 1.0
width_multiple: 1.0
# yolov5m
depth_multiple: 0.67
width_multiple: 0.75
# yolov5s
depth_multiple: 0.33
width_multiple: 0.50
# yolov5n
depth_multiple: 0.33
width_multiple: 0.25
```

根据代码，我们将这两个因子简记为 gd 和 gw，

```python
gd, gw = d['depth_multiple'], d['width_multiple']
```

gd 影响的是 block 中 layer 的数量，例如 C3 block 的配置 `[-1, 3, C3, [128]]`，其中 bottleneck layer 的数量是 3，经过 gd 缩放后，变为 `int(gd * 3)` 个 bottleneck layers。如果 block 中 layer 数量为 1， 那么不使用 gd 缩放。

gw 影响 layer 的输出 channel，除最后一个输出 layer（包括 P3/P4/P5） 以外，layer 的输出 channel 需要乘以 gw，并需要保证结果是 8 的整数倍，如不是，向上增大到 8 的整数倍。

# 1. backbone

backbone 是一个 block 列表，例如第一个 block 配置

```sh
# -1：表示上一个 block 的输出 channel size。若没有上一个 block，则为 input channel size， e.g. 3
# 1：当前 block 中 layer 数量，需要乘以深度因子
# Conv：当前 block 中 layer 类型
# [...]: 分别表示：layer 输出 channel size，kernel size，stride，padding
[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
```

**# C3**

以下方配置为例说明 C3 这个 block，

```sh
# -1：上一个 block 输出 channel
# 3：C3 中 bottleneck-layer 数量基数
# [128]：本 block 输出 channel 为 128
[-1, 3, C3, [128]]
```

C3 模块由 `3` 个 conv 和 `n` 个 bottleneck 组成。

```sh
# 输入输出的 channel
c2/2             c2/2         c2/2    c2/2
        +-------+    +-------+
---+--->| conv -+--->| conv -+---+--->
   |    +-------+    +-------+   ^
   |                             |
   +-----------------------------+      (bottleneck)

# 输入输出的 channel
c1               c2/2                     c2/2            c2
        +-------+    +----------------+
   +--->| conv -+--->| bottleneck xn -+---+
   |    +-------+    +----------------+   |
---+                                      |      +-------+
   |    +-------+                         +----->|       |
   +--->| conv -+------------------------------->| conv -+--->
        +-------+                                +-------+     (C3)
```

**# SPPF**

Spatial Pyramid Pooling，以下方配置为例说明， 

```sh
# -1：上一个 block 输出 channel
# 1： 当前 block 的 SPPF layer 数量
# [1024, 5]：本 block 输出 channel 为 1024，池化 kernel size 为 5
[-1, 1, SPPF, [1024, 5]]
```

SPPF 将输入经过一个 conv，输出记为 `x`，然后 `x` 连续经过 3 个 maxpool（stride=1，所以输出 spatial size 不变），`x` 与 3 个 maxpool 的输出沿 channel 维度 concat，最后经第二个 conv 输出，

```sh
# 输入输出 channel
c1          c2/2        c2/2      c2/2       c2/2   c2*2        c2
    +-------+    +-----+    +-----+    +-----+        +-------+
--->| conv -+-+->| mp -+-+->+ mp -+-+->| mp -+------->|       |
    +-------+ |  +-----+ |  +-----+ |  +-----+        | conv -+--->
              |          |          +---------------->|       |
              |          +--------------------------->|       |
              +-------------------------------------->|       |
                                                      +-------+     (SPPF)
```

**# 下采样率**

backbone 中只有 5 个 Conv 中的 stride=2 进行了下采样，所以总的下采样率为 $2 ^ 5 = 32$ 。

# 2. head

head 子网络中各 block/layer 与 backbone 中类似处理。

**# upsample**

```sh
# -1：上一个 block 的输出 channel
# 1： 当前 block 只有一个 upsample layer
# [...]：nn.Upsample 类初始化参数，size 为 None，上采样率为 2， 使用 nearest 插值
[-1, 1, nn.Upsample, [None, 2, 'nearest']]
```

**# Concat**

```sh
# [-1, 6] 指定上一个 block 输出与第 `6` 个 block 输出，两者 concat
# 1：当前 block 只有一个 concat layer
# [1]：沿着 dim=1 进行 concat
[[-1, 6], 1, Concat, [1]]
```

整个模型结构如下图所示，

![](/images/obj_det/yolov5_1.jpg)

<center>图 1. yolov5 结构图</center>

图 1 中，`conv-2` 中的 `2` 表示下采样率为 2， `x32` 表示下采样率为 `32` ，最终送入 `Detect` 的特征相对于原始输入 size，下采样率为 `32`，最左一列对应 backbone 结构。


**# Detect**

配置

```sh
# [17, 20, 23]：指定三个 layer 的输出特征作为 Detect 的输入，如图 1 中三个彩色线数据流
[[17, 20, 23], 1, Detect, [nc, anchors]]
```

根据图 1，可知三个 layer 的输出特征 spatial size 分别为 $(H _ 0/8, W _ 0 / 8)$、$(H _ 0/16, W _ 0 / 16)$、$(H _ 0/32, W _ 0 / 32)$，其中 $(H _ 0, W _ 0)$ 是网络输入 size 。三个特征的 channel size 分别是 `256, 512, 1024` 。

**# 锚框**

三个特征平面各自预测，由于特征平面 scale 不同，所以每个 scale 使用各自的一组 anchors，

```sh
# 各个特征平面上的 anchor 尺寸，[w1, h1, w2, h2, w3, h3]
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32
```

需要注意的是，以上 anchor 的 size 基于输入图片，在训练时，需要将 anchor size 映射到相应的特征平面上。例如模型输入 size 为 (640, 640)，那么 P3 特征平面 size 为 (80, 80)，整体 stride=8，P3 的第三种 anchor (w=33, h=23) 映射到 P3 特征平面上则为 (33.0/8, 23.0/8)（保持浮点数类型），下文在计算损失时也会讲到。

**# Detect 的输出 channel**

每个 location 处预测 3 个 anchor boxes，每个 anchor box 预测 `C+5` 的数据，分别为 `C` 个分类得分，`4` 个坐标以及 `1` 个 IOU 置信度，所以需要使用 `1x1` 卷积先将特征 channel size 转为 `3*(C+5)` ，故 Detect 输出为三个 scale 的预测值，shape 为

```sh
(3*(C+5), H0/8, W0/8)
(3*(C+5), H0/16, W0/16)
(3*(C+5), H0/32, W0/32)
```

Detect 在 train 模式下的输出为一个具有 3 个 tensor 的 list，tensor 的 shape 分别为 

```sh
# b 为 batch size，3 表示每个 location 预测 3 个不同size的 anchors，
# C 为分类数量，5 表示 xywh 和置信度，H0 W0 表示模型的输入 size。
[b, 3, H0/8, W0/8, C+5]
[b, 3, H0/16, W0/16, C+5]
[b, 3, H0/32, W0/32, C+5]
```

# 3. 预处理

**# 数据准备**

每个图像文件对应一个 label 文件，label 文件中每一行对应一个目标，格式为

```sh
# 坐标已经归一化
cls_id x_center y_center width height
```

**# 加载图像**

Yolov5 整个网络最大下采样率为 `32`，所以我们约定输入图像边长至少为 `32*2` 。项目文档中，设置图像最大边长（H,W 中最大值）为 `640`

1. opencv 读取图像文件
2. 最大边长不等于 `640`，那么根据比例 `r`，resize 图像，使得 resize 之后图像最大边长为 `640`

## 3.1 数据增强

**# mosaic**

对当前图像马赛克处理。步骤：

1. 除了当前图像，另外再在数据集中随机选择 3 个图像，然后打乱这 4 个图像顺序
2. 使用这 4 个图像合成一个新的图像，这 4 个图像依次位于新图像的左上角，右上角，左下角，右下角
3. 新图像使用 `640x640` size，每个 pixel 值为 `(114, 114, 114)`
4. 在 `[640-320, 640+320]` 两次随机采样，得到 `(yc, xc)`，作为中心点坐标
5. 按上一小节 `加载图像` 中的方法加载这 4 个图像，然后确定各个图像在新图像中的 region 

  - 左上图像，以 `(yc, xc)` 为右下角点
  - 右上图像，以 `(yc, xc)` 为左下角点
  - 左下图像，以 `(yc, xc)` 为右上角点
  - 右下图像，以 `(yc, xc)` 为左上角点

  - 以左上图像为例，左上图像通常不会完全与新图像左上角区域吻合。中心点 `(yc, xc)` 对准图像右下角，然后计算新图像与左上图像的公共重叠部分。其余 3 个图像类似处理。相关代码如下，
    ```python
    x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # 新图像 region
    x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # 左上图像 region
    img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
    ```

  - 调整 label，即各个目标坐标从原来图像转变到新图像中的坐标，显然目标 size 不变，仅仅目标中心坐标改变了。如下图所示，

    ![](/images/obj_det/yolov5_2.png)

    此图演示了左上图像中目标坐标如何转换。首先计算宽度和高度方向的填充距离，

    ```python
    padw = x1a - x1b  # x1a: 公共 region 左侧在新图像的坐标；x1b：公共 region 左侧在左上图像中的坐标
    padh = y1a - y1b
    ```
    然后转换到新图像坐标时，加上填充距离即可，

    ```python
    # w,h 为子图像 resized size。原先的 box label 为归一化的 `(xc, yc, w, h)`
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    ```
  
## 3.2 数据加载

数据集 `__getitem__` 返回：

```python
# image: (C,H,W) RGB
# label: (n, 6)，n 是图像中目标数量，[1:6] 表示 cls_id, x,y,w,h （归一化）
#       [0] 第一列用于记录目标属于 batch 中哪个 image
torch.from_numpy(img), labels_out
```

数据加载器返回的一个 batch：

```python
im, label, path, shapes = zip(*batch)
...
# (B, C, H, W)
# (N, 6)
torch.stack(im, 0), torch.cat(label, 0)
```

由于 batch 中每个 image 中目标数量不等，$[n_1, \ldots, n _ B]$，将所有目标 label 展平，得到 $(N, 6)$ 的 tensor，其中每一行表示一个目标，第一列记录了每个目标所属的 image 编号。 $N=\sum _ i n _ i$ 。

# 4. 损失

构建检测模型时，anchors 的 size 根据特征 scale 做了相应的调整，即，每个 anchor 在相应的特征平面上的 size，

```python
# 位于 DetectionModel 初始化函数中
m.anchors /= m.stride.view(-1, 1, 1)  # shape: (nl=3, na=3, 2)
# anchor size / stride
# [10,13, 16,30, 33,23] / 8
# [30,61, 62,45, 59,119] / 16
# [116,90, 156,198, 373,326] / 32
```

计算损失的类函数为 `ComputeLoss`，输入为

```sh
# pred 是三个 scale feature maps 的预测结果，每个 location 有 3 个 anchors，每个 anchor 预测 C+5 个数据
pred: [(B,3, H/8, W/8, C+5), (B,3, H/16, W/16, C+5), (B,3, H/32, W/32, C+5)]

# N 为 batch images 中所有目标总数。第一列记录目标所示 image index。第二列为 cls_id，后 4 列为归一化 x,y,w,h
targets: (N, 6)
```

相关代码为，

```python
pred = model(imgs)
loss, loss_items = compute_loss(pred, targets.to(device))
```


每个 scale 的特征使用对应 scale 的 3 个 anchors 进行监督。例如第一个特征 P3：

1. 其 anchors size 为

    ```sh
    [[10/8, 13/8],  [16/8, 30/8],  [33/8, 23/8]]  # W H of anchors, based on the feature
    ```

2. 所有 gt boxes resized 到基于特征 size

    $$w = w _ n * W/8, \quad h = h _ n * H/8$$

    其中 $w _ n, h_n$ 是 gt box 归一化宽度和高度，$W, H$ 是模型输入 size，8 是 P3 特征的 stride，$w, h$ 是 anchor box 在特征平面上的 size。

> 以下内容来自 https://mmyolo.readthedocs.io/zh_CN/latest/recommended_topics/algorithm_descriptions/yolov5_description.html

## 4.1 bbox 编解码

yolov3 中，回归公式为 

$$\begin{aligned}b _ x &= \sigma(t _ x) + c _ x
\\\\ b _ y &= \sigma(t _ y) + c _ y
\\\\ b _ w &= a _ w \cdot e ^ {t _ w}
\\\\ b _ h &= a _ h \cdot e ^ {t _ h}
\end{aligned} \tag{1}$$

其中 $c _ x , c _ y, a _ w, a _ h$ 表示负责预测 gt box 的 anchor box 的中心坐标和宽高。$b _ x, b _ y, b _ w, b _ h$ 是 gt box 的中心点和宽高（训练阶段），或者预测 box 的中心点和宽高（test 阶段）。

yolov5 中，回归公式为

$$\begin{aligned}b _ x &= [2 \cdot \sigma(t _ x) - 0.5] + c _ x
\\\\ b _ y &= [2 \cdot \sigma(t _ y) - 0.5] + c _ y
\\\\ b _ w &= a _ w \cdot [2 \cdot \sigma(t _ w)] ^ 2
\\\\ b _ h &= a _ h \cdot [2 \cdot \sigma(t _ h)] ^ 2
\end{aligned} \tag{2}$$

yolov3 中，损失有 3 项：

1. 分类损失：**正例** 分类损失之和
  
    **正例** 的分类损失，target 为 `1`（one hot 向量），预测为 `C` 长度向量，使用 target 所属分类 `c` 处的预测得分，计算这个得分与 `1` 的差的平方

2. 坐标损失：**正例** 的坐标损失之和

    target 值使用匹配的 (gt box, anchor box) pairs 计算

3. 置信度损失：**正例** 置信度和 **负例** 置信度加权和

    **正例：** 计算 anchor boxes（M 个） 与 gt boxes（N 个）的 IOU 矩阵，shape 为 $(M, N)$，每列求最大值，最大值所在的行 index 所对应的 anchor box 就是 **正例** 。（正例也有使用 pred boxes 与 gt boxes 计算出 $(M, N)$，然后类似处理，得到哪些位置第几个预测为正例）

    **负例** ： 除去上述 **正例** 的所有其他 anchor boxes 均为负例。当然，还有一种处理是，计算 pred boxes 与 gt boxes 的 IOU 矩阵，shape 也是 $(M, N)$，按行求最大值，找出每行最大 IOU > 0.5 的行，这些行不作为负例，即，从前述负例中去掉这些行。

    正例置信度：1 或者正例位置处的 pred box 与正例位置处匹配的 gt box 之间的 IOU 。

    负例置信度：0


回归公式从 (1) 改为 (2)，有什么好处？ 

1. 中心点坐标偏差（offset: $b_x - c_x$）范围从 `(0, 1)` 变成 `(-0.5, 1.5)` ，如图 3 所示，在 `0` 点附近更陡（毕竟斜率变成 2 倍），更能精确的预测目标中心坐标，注意 `-0.5` 这个 bias，是为了确保 `0` 点（$t_x=0$ 位置处）偏差值固定在 `0.5` 不变。

    ![](/images/obj_det/yolov5_3.png)

    <center>图 3.</center>

    由于 offset 范围从 `(0, 1)` 变成 `(-0.5, 1.5)` ，所以可以预测 box 中心点的范围从图 4 黄色区域扩大到黄色+蓝色区域，

    ![](/images/obj_det/yolov5_4.png)

    <center>图 4. 预测 box 中心点的范围</center>

2. 宽比/高比（预测 box 宽与 anchor box 宽的比值）范围从 $(0, +\infty)$ 改为 $(0, 4)$，如图 5 所示，

    ![](/images/obj_det/yolov5_5.png)

    <center>图 5. </center>

    yolov3 中宽比为 $\exp (t _ w)$，这会导致梯度过大导致训练不稳定

## 4.2 确定正负例

yolov3 中使用最大 IOU 确定正例 。yolov5 使用 shape match 确定正例，过程如下。

**shape 比例 比较**

记下标 gt 表示 gt box，下标 pt 表示 pred box（or anchor box？），按以下流程匹配：

$$\begin{aligned} r _ w &= w _ {gt} / w _ {pt}
\\\\  r _ h &= h _ {gt} / h _ {pt}
\\\\ r _ w ^ m &= \max (r _ w , 1 / r _ w )
\\\\ r _ h ^ m &= \max (r _ h, 1/ r _ h)
\\\\ r ^ m &= \max (r _ w ^ m, r _ h ^ m)
\\\\ \text{if} \ r ^ m & < \text{prior\_match\_thr: match}
\end{aligned} \tag{3}$$

其中 `prior_match_thr` 是一个预先设置的比例阈值，可取值 `4` 。(3) 式表示 $r _ w ^ m < 4$ 并且 $r _ h ^ m < 4$，以宽度比为例，

$$r _ w < 4 \ , \ 1/r _ w < 4 \Rightarrow \frac 1 4 < r _ w < 4$$

(3) 式中是 gt box 与 pred box 宽度比还是 pred box 与 gt box 宽度比，其实无所谓，因为对称，所以 $1/4 < 1/ r _ w < 4$ ，比例范围仍是 $(1/4, 4)$ 。**结论** 就是，

**pred box 与 gt box 边长比例均在 $(1/4, 4)$ 范围内的则是正例。**

例如：anchor box 为 (10,13)，一个 gt box 为 (20, 24)，显然宽比和高比均在 $(1/4, 4)$ 范围内，那么两者匹配，如果 gt box 为 (60, 40)，那么高比在这个范围内，而宽比不在这个范围内，所以不匹配。

前面根据 (2) 式可知，我们预测的 size 比例范围为 $(0, 4)$，这两个范围就比较接近，当然了，我们也可以将 (2) 式中 $\sigma (t _ w)$ 前面的系数 `2` 改成其他值，也可以调整 `prior_match_thr` 阈值，一切都根据实际任务情况做调整。


**正例 location 确定**

yolov3 中，gt box 中心落于哪个 location，那么这个 location 上 B 个 anchor boxes 中有着最大 IOU 的那个 anchor box 才是正例。这里需要搞清楚几点：

1. feature map 上每个 location 均使用相同宽高的 B 个 anchor boxes 进行预测。
2. 由于 anchor boxes 是密集的，每个 gt box 上均有 B 个 anchor boxes ，记 gt boxes 总共 N 个，那么计算它们的 IOU 矩阵，shape 为 $(B, N)$，按列求最大值所在行号，就表示每个 gt box 与这 B 个 anchor boxes 中，哪个最匹配，这个最匹配的 anchor box 的中心由 gt box 的坐标确定。

确定一个正例 anchor box，需要其中心坐标，以及其是 B 个 anchors 中的哪一个。前者确定正例 anchor 的 location，后者确定 anchor 的 shape，后者通过与 gt box 有最大 IOU 确定，前者通过这个 gt box 中心点坐标确定。

对这里的 **yolov5** 而言，我们前面通过 shape 比例匹配可以确定 B 个 anchors 中哪些 anchors 可作为正例，注意 yolov3 是采用最大 IOU 策略，所以一个 gt box 有且仅有一个 anchor 作为正例，由于 anchors 的密集性，我们先不考虑 anchor 的中心点坐标。而 yolov5 采用 shape 比例在范围 $(1/4, 4)$ 的策略，所以会匹配到 B 个 anchors 中的一个或多个 anchoors 作为正例。

好了，现在来说说 yolov5 中如何确定正例的 location ，如图 6 所示，

![](/images/obj_det/yolov5_6.png)

<center>图 6.</center>

一个 gt box，记其中心落入 grid cell `(i,j)` ，将这个 cell 划分为 4 个象限，如果 gt box 中心点坐标位于 cell 左上角区域，那么除了 `(i,j)` 这个 location，还有 `(i-1, j)` 和 `(i,j-1)` 两个 location 均被看作是正例。对于其他三个象限情况类似处理，如图 6 。一个特殊的情况是，如果正好位于 cell 中心，那么就只有 `(i,j)` 这个 location 为正例。

这种方法引入邻域 cell 是为了增强正样本，一定程度地缓解正负样本不平衡问题。

需要注意，对于图 6 的左边 4 种考虑邻域 cell 的情况，不要忘了一个前提条件：

1. 图 6 左边起第一个子图，当 $i \ge 1$ 时才能利用其左侧的 `(i-1,j)` cell ，即 gt box 中心点的 x 坐标 $x _ {gt} \ge 1.0$ 且 $x _ {gt} \\% 1 < 0.5$ 。$i$ 与 $x _ {gt}$ 的关系满足 $\lfloor x _ {gt} \rfloor = i$ 。

2. 其他情况的前提条件类似。

**总结**：

1. 一个 gt box，对应的正例 anchors 的 location 为这个 gt box 中心所在 cell 以及其邻域的 cell

2. 一个 gt box，对应的正例 anchors 为： B 个 anchors 中与这个 gt box 宽比和高比均在 $(1/4, 4)$ 范围内的 anchors

所以，一个 gt box，其匹配的正例对应一个或多个 location，每个 location 对应一个或多个 anchors 。

**负例**：除了正例之外的全部为负例。

## 4.3 损失计算

YOLOv5 中总共包含 3 个 Loss，分别为：

1. Classes loss：使用的是 BCE loss。只针对 **正例** 计算。

    对于一个 正例而言，分类损失 $l = -\sum _ {i=1} ^ C q _ i \log p _ i$，target 值为 one-hot 向量 $\mathbf q$，预测分类得分概率为 $\mathbf p$ 。通常，也会对 one-hot 向量的 target 做 soft label 处理，参考论文 [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/pdf/1902.04103.pdf) 中的公式 (3) 。

2. Objectness loss：使用的是 BCE loss 。考虑了 **正例** 和 **负例** 。第 `n` 个样本的损失为

    $$l _ n =-[y_n \log \sigma(x_n) +(1-y_n) \log (1-\sigma(x_n))]$$

    对于正例，$y _ n= iou$ ；对于负例，$y _ n = 0$

3. Location loss：使用的是 CIoU loss 。只针对 **正例** 计算

三个 loss 各自求了均值（置信度损失除以总样本数，其余两个损失除以正样本数），然后按照一定比例汇总：

$$L = \lambda _ 1  L _ {cls} + \lambda _ 2 L _ {obj} + \lambda _ 3 L _ {loc}$$

其中 Objectness loss 是三个 scale 特征平面计算的 Objectness loss 加权和，另外两种损失则是直接计算和，

$$L _ {obj} = 4.0 \cdot L _ {obj} ^ s + 1.0 \cdot L _ {obj} ^ m + 0.4 \cdot L _ {obj} ^ l$$