---
title: YOLO
date: 2020-04-20 13:58:33
p: obj_det/YOLO
tags: object detection
mathjax: true
---

# 1 YOLOv1
## 1.1 简介
one-stage 检测方法，从图像像素到bbox 坐标和分类概率，一步到位。优点为：
1. 快。YOLO 每秒可处理 45 帧，可实时处理视频流。
2. 从图像全局出发，作出预测。这种策略有效利用了上下文信息。
3. 能学习目标的泛化特征。基于天然图像数据集上的训练，用于人工图像数据集上的测试，也能取得较好的结果。

<!-- more -->

## 1.2 方法
YOLOv1 网络结构如图1，采用的是 VOC 数据集。
![](/images/obj_det/YOLOv1_fig1.jpg) <center>图 1 YOLOv1 网络结构</center>

输入 shape : `448,448,3`，
输出 shape ：`7,7,35`

### 1.3 检测思想

1. 将输入图像划分为 `SxS` 大小的网格，如果目标中心落于某个 grid cell，那么这个 grid cell 负责检测这个目标。对于 VOC 数据集，使用 `S=7`，这是综合考虑，平衡了计算量和准确率。
3. 每个 grid cell 预测 `B` 个 box。这里取 `B=3`。
4. 对每个 box 预测，需要 5 个数据 `(x,y,w,h,IOU)`（全部都是归一化的）。
5. VOC 数据集的分类数量 `C=20`。每个 grid cell 处预测 `C` 个分类概率，即，每个 grid cell 处的 `B` 个 box 共享这 `C` 个分类概率（因为实际上，`B` 个预测 box 中只有 一个 box 负责预测）。
2. 从 `448` 到 `7`，网络的下采样率为 `448/7=64`。从图 1 也能看出，具有 `s-2` 字样的 layer 共有 6 个。输出 map 的 spatial size 变成 `7x7`，channel 变成 `35`，这是因为每个空间位置处需要 `B*5+C=3*5+20` 个预测数据。
6. 训练阶段，计算损失，并求梯度，然后更新，具体参见下文详细分析。
7. 测试阶段，共检测出 `S*S*B` 个 box，每个 box 有 4 个坐标值，1 个 IOU 以及 C 个分类概率。对分类概率进行一个阈值截断，阈值默认为 0.2。分别针对每个分类，根据分类概率倒序排列，对 box 进行非极大抑制（设置被抑制 box 的当前分类的概率为 0），非极大抑制阈值默认为 0.4。最后，筛选出所有检测 box 中具有大于阈值（0.2）的分类概率，为最终检测结果。

__思考：__ 为什么每个 grid cell 处预测不是 1 个 box 而是多个 box？

__答：__ 我们假定不会有多个目标的中心落入同一个 grid cell，如果确实存在（这种概率很低），那么只有第一个目标的数据`x,y,w,h,IOU,class`会写入 gt `Y` 中。每个 grid cell 仍然预测多个 box，这是因为这些不同的预测 box 将具有不同的 size 或 aspect ratio。如果目标中心落入某个 grid cell，那么其上的 `B` 个预测 box 中，只有与目标 IOU 最大的预测 box 才负责预测。例如，某个预测 box 适合预测高的目标（人），而另一个预测 box 可能适合预测宽的目标（车）。

### 1.4 损失

$$\begin{aligned} L&=\lambda_{coord} \sum_{i=1} ^ {S ^ 2}\sum_{j=1} ^ B \mathbf 1_{ij} ^ {obj}(x_i-\hat x_i) ^ 2+(y_i-\hat y_i) ^ 2 \\\\ &+ \lambda_{coord} \sum_{i=1} ^ {S ^ 2}\sum_{j=1} ^ B \mathbf 1_{ij} ^ {obj}(\sqrt {w_i}- \sqrt {\hat w_i}) ^ 2+(\sqrt {h_i}- \sqrt {\hat h_i}) ^ 2 \\\\ &+ \sum_{i=1} ^ {S ^ 2}\sum_{j=1} ^ B \mathbf 1_{ij} ^ {obj} (C_i-\hat C_i) ^ 2 \\\\ &+ \lambda_{noobj} \sum_{i=1} ^ {S ^ 2}\sum_{j=1} ^ B \mathbf 1_{ij} ^ {noobj}(C_i-\hat C_i) ^ 2 \\\\ &+ \sum_{i=1} ^ {S ^ 2} \mathbf 1_i ^ {obj} \sum_{c \in classes}\left(p_i( c )-\hat p_i( c )\right) ^ 2 \end{aligned}
$$

__分析：__

带 `^` 的为网络输出，不带 `^` 则为 ground truth 值。`x, y, w, h` 为中心点坐标，`C` 为 IOU，`pi(c)` 为分类 `c` 的概率。

$\mathbf 1_{ij}^{obj}$ 表示第 `i` 个 grid cell 有目标（中心），且此 grid cell 上第 `j` 个预测 box 与 gt box 有最大 IOU，即，第 `j` box 负责预测。

对于较大 box 和 较小 box，在相同偏差$\Delta w, \ \Delta h$ 下，较大 box 的损失应该比较小 box 的损失更小才合理，然而两者平方差损失相同，所以我们对宽高 `w,h`，先求平方根，再求平方差，这在一定程度上降低了这种不合理性。

$\mathbf 1_{ij} ^ {noobj}$ 表示： 第 `i` 个 grid cell 无目标（中心），或者有目标（中心），但是第 `j` 个预测 box 不负责预测（即，与 gt box 的 IOU 不是 `B` 个预测 box 中最大的）。

$\mathbf 1_i^{obj}$ 表示第 `i` 个 grid cell 有目标（中心）。

坐标损失与分类损失分属两类损失，需要进行平衡。此外，由于大部分预测 box 其实并不负责预测，来自这部分预测 box 的 IOU 损失（损失公式中第四行）将会压制负责预测的 box 的坐标损失和 IOU 损失（损失公式中前三行），所以需要提升被压制的那部分的损失贡献。综合考虑，设置 $\lambda_{coord}=5, \ \lambda_{noobj}=0.5$。

## 1.5 细节
1. GT label 数据的 size 为 `S*S*(5+20)`，其中 `5` 包含了 4 个坐标值，1 个 IOU，20 为表示分类 id 的 one-hot vector 的长度。维度从高到低为 `(S,S,5+20)`，最低维数据顺序为 IOU, class id, x,y,w,h。
2. 网络输出 size 为 `S*S*(B*5+20)`，维度从高到低为 `(5+20,S,S)`，通道顺序为 class id, IOU, x,y,w,h。
3. GT label 数据中， x,y,w,h 先进行归一化（除以图像宽/高），然后 `x=x*S-(int)x*S, y=y*S-(int)y*S`。
4. 网络输出中的 x,y 与 GT label 中含义一致，表示相对于 grid cell 的（归一化）偏差，而 w,h 则是经过了平方根处理。


# 2. YOLOv2
## 2.1 简介
YOLOv2 是对 YOLOv1 的改进，包括：
1. 利用现有的分类数据集来扩展检测数据集，使得检测目标的分类种数更多。
2. 增加 Batch Normalization
3. 检测小目标更准确

在分类集上预训练时，就使用较大分辨率的图像。YOLOv1 中使用 `224x224` 在分类集上预训练，然后直接将 `448x448` 大小的检测数据训练集喂给网络，这让网络同时适应高分辨率的图像以及学习目标检测，难免压力山大。YOLOv2 中，每隔十次 batch 训练，变更一次网络输入 size。

## 2.2 方法
以 VOC 数据集为例，YOLOv2 的网络结构可以从配置文件 `cfg/yolov2-voc.cfg` 中获取。
### 2.2.1 实现细节
1. 输入 size 每隔 10 个 batch 变更一次，从 `320, 352, ..., 608` 这十个值中随机选择。记输入大小为 `(3,d,d)`。
2. 网络整体下采样率为 32，输出大小为 `(125, d/32, d/32)`。其中，`(d/32,d/32)` 与 YOLOv1 中类似，可以看作原图像上的 grid cell 数量 `S=d/32`。如果目标的中心落入某个 grid cell，那么这个 grid cell 负责预测目标。每个 grid cell 上有 `5` 个预测 box，每各 box 有 `1` 个 IOU 以及 `4` 个坐标值，每个 box 独立拥有 `20` 个分类得分，故输出 channel 为 `125=5*(1+4+20)`。注意，YOLOv1 中每个 cell 上的 `B` 个预测 box 共享 `20` 个分类得分。

3. 人为假设每个图像中目标数量最多为 30，所以 GT label 大小为 `30x5`，其中 `5` 包含了 4 个坐标值以及 1 个分类 id。最低维数据顺序为 x,h,w,h,class id。GT label 靠前存储。
4. `(route:-1，-4)` 层将浅层特征（高分辨率）与高层特征（低分辨率）融合，类似于 ResNet 中的 identity mapping，这种更细粒度的特征将有助于小目标的检测。

### 2.3 损失
损失包括：分类损失，置信度损失，坐标损失三部分。

$$L=L_p+L_{box}+L_C$$

__分类损失__

$$L_p=\sum_{i=1} ^ {S ^ 2} \sum_{j=1} ^ B \sum_{c=1} ^ {20} \mathbf 1_{ij} ^ {obj} [\hat p_{ij}(c)-p_{ij}(c)] ^ 2$$

__坐标损失__

$$\begin{aligned}L_{box}&=\lambda_{obj} ^ {coord} \sum_{i=1} ^ {S ^ 2} \sum_{j=1} ^ B \mathbf 1_{ij} ^ {obj} (\hat x_{ij} - x_{ij}) ^ 2 + (\hat y_{ij} - y_{ij}) ^ 2+ (\hat w_{ij} - w_{ij}) ^ 2+ (\hat h_{ij} - h_{ij}) ^ 2 
\\\\ &+ \lambda_{noobj} ^ {coord} \sum_{i=1} ^ {S ^ 2} \sum_{j=1} ^ B \mathbf 1_{ij} ^ {noobj} (\hat x_{ij} - x_{ij} ^ a) ^ 2 + (\hat y_{ij} - y_{ij} ^ a) ^ 2+ (\hat w_{ij} - w_{ij} ^ a) ^ 2+ (\hat h_{ij} - h_{ij} ^ a) ^ 2 \end{aligned}$$

__置信度损失__

$$\begin{aligned}L_C &=\lambda_{obj} ^ {conf}\sum_{i=1} ^ {S ^ 2} \sum_{j=1} ^ B \mathbf 1_{ij} ^ {obj}[\hat C_{ij}-iou(\hat {\text{box}} _ {ij}, \text{box} _ {ij})] ^ 2  
\\\\ &+ \lambda_{noobj} ^ {conf}\sum _ {i=1} ^ {S ^ 2}\sum_{j=1} ^ B \mathbf 1_{ij} ^ {noobj}[\hat C_{ij}-0] ^ 2 \end{aligned}$$

以上，带 ^ 表示 network 输出，带 a 表示 anchor，不带这两个修饰的表示 GT label。

__分析：__

网络输出 shape 从高维到低维为，`batch, B, 4+1+C, S, S`（其实无论几维，在内存中都是一维）。这里假设了输出 feature map 的 height 和 width 相等，均为 `S` （grid size），且 `4` 表示 4 个坐标，`1` 表示 IOU，`C` 表示分类数量。

与 YOLOv1 中类似，目标中心落入某个 grid cell，那么这个 grid cell 负责预测目标。每个 grid cell 有 `B=5` 个预测 box，具有不同的 size。使用 5 组 anchor box 帮助预测，参考 yolov2-voc.cfg 文件中最后一个 layer 配置中 `anchors` 的值，给了 5 组 width height 的值，这些值基于输出 feature map 的 size `SxS`，即，并没有归一化。anchor box 的中心为所在 grid cell 坐标加 0.5，即 `(i,j)` 处 grid cell 的 anchor box 中心为 `(i+0.5, j+0.5)`。

网络输出坐标 `x,y,h,w` 的具体含义，如图 2，
![](/images/obj_det/YOLOv2_fig2.png) 

<center>图2 预测 box 与 anchor box 的关系</center>

网络输出坐标实际含义就是 $\sigma(t_x), \sigma(t_y), t_w, t_h$。

一幅图像的 GT label 的 size 为 `30*5`，低维数据排列顺序为 `x,y,w,h, class id`，其中 `x,y,w,h` 是基于 original image 的 size 进行了归一化（`x,y` 与 YOLOv1 中稍有不同）。


坐标损失中 $x_{ij}, y_{ij}, w_{ij}, h_{ij}$ 使用的是 $\sigma(t_x), \sigma(t_y), t_w, t_h$，对于网络输出，不用做任何修改，而对于 GT box 以及 anchor box，则需要做变换，也就是说，将预测 box 分别替换为 GT box 和 anchor box 来计算 $\sigma(t_x), \sigma(t_y), t_w, t_h$。

位于某 location `(i,j)` 处，将 `B` 个预测 box 与 GT label 中所有目标 box 两两求 IOU，最后得到一个最大 IOU，如果这个最大 IOU 大于阈值 0.5，那么 $\mathbf 1_{ij} ^ {noobj}=0$，此时置信度损失中第二项为 0。

对于每个 GT box，找出与这个 GT box 有最大 IOU 预测 box，注意这个 IOU 没有阈值限制，然后设置 $\mathbf 1_{ij} ^ {obj}=1$（每个 GT box 有且只有一个负责预测的 box），此时置信度损失中第一项非零，且分类损失非零，此时计算分类损失时，$\sum_{c=1} ^ C$ 求和中，当且仅当 `c` 等于 GT label 中的 class id 时，$p_{ij}(c)=1$，其余 `C-1` 种情况 $p_{ij}(c)=0$。

# 3. YOLOv3
在 YOLOv2 基础上做了修改：
1. 三个 scale 的输出 feature maps。每组 feature maps 的大小为 `NxNx[3*(4+1+C)]`，三个不同的 `N`，依次增大 2 倍。
2. 使用 `9` 个不同 scale 的 anchor box 帮助预测。由于有 `3` 个 scale 的 feature maps，所以实际上，每个 scale 大小的 feature maps 上每个 grid cell 仅使用 `9/3=3` 个 anchor box。

以 VOC 数据集为例，网络结构参见 `cfg/yolov3-voc.cfg`。

1. 特征抽取网络的下采样率为 `32`。如果输入图像的大小为 `(h,w)`，那么输出feature map 大小为 `(h/32,w/32)`，另外两个 scale 的 feature maps 的大小则为 `(h/16,w/16)` 和 `(h/8, w/8)`。
2. 单个图像的 GT label 大小 为 `90*5`。这表示单个图像中目标数量最大不超过 `90`。
3. 大量使用 Residual layer。
4. anchor size 为 `anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326`，先 w 后 h，前 3 个 anchor size 对应 stride=8 的特征平面，中间 3 个 anchor size 对应 stride=16 的特征平面，最后 3 个 anchor size 对应 stride=32 的特征平面，

![](/images/obj_det/YOLO_3.png)

<center>图 3. Darknet53 结构</center>

backbone 三个输出分支分别为：
1. 第三个 residual block 的输出，图 3 中的 `32x32x256`
2. 第四个 residual block 的输出，图 3 中的 `16x16x512`
3. 最后的 residual block 的输出，图 3 中的 `8x8x1024`