---
title: yolov8 解读
date: 2023-08-11 17:51:10
tags: object detection
mathjax: true
---

源码：[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

# 1. 网络结构

yolov8 与 yolov5 一样，不仅用于检测，还可以用于分类，分割等任务。本文主要讨论目标检测任务。

yolov8 有 5 个 size 不同的模型，结构相似，在模型 depth，width，max_channels 三个维度上不同，所有模型配置位于文件 `ROOT/cfg/models/v8/yolov8.yaml` 。

```sh
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024]  # YOLOv8n summary: 225 layers
  s: [0.33, 0.50, 1024]  # YOLOv8s summary: 225 layers
  m: [0.67, 0.75, 768]   # YOLOv8m summary: 295 layers
  l: [1.00, 1.00, 512]   # YOLOv8l summary: 365 layers
  x: [1.00, 1.25, 512]   # YOLOv8x summary: 365 layers
```

其中 depth 和 width 与 yolov5 中一样，一个是 block 中 layer 的数量因子，一个是输出 channel 因子。max_channels 则限制了在使用 width 因子之前的输出 channel 值，如下代码所示，

```python
c1, c2 = ch[f], args[0]
if c2 != nc:  # nc 是分类数量
    c2 = make_divisible(min(c2, max_channels) * width, 8)
```

目标检测模型类位于文件 `nn/tasks.py`，为 `DetectionModel`，其他任务如分割，分类，姿态估计等模型类也位于文件 `nn/tasks.py` 中。

以 yolov8l 为例说明，因为 `l` 这个模型的 depth 和 width 的缩放比例均为 `1` ，网络配置为，

```sh
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]  # 9

# YOLOv8.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f, [512]]  # 12

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f, [256]]  # 15 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]  # cat head P4
  - [-1, 3, C2f, [512]]  # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]  # cat head P5
  - [-1, 3, C2f, [1024]]  # 21 (P5/32-large)

  - [[15, 18, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5)
```

网络的整体结构与 yolov5 类似，其中 `C3` 换成了 `C2f` 模块，并且去掉了 head 中 upsample 前面的 conv 。

**几个特殊的 模块**

**C2f**

```sh
    +-------+                                    +-------+
--->| Conv =+==+-------------------------------->|       |
    +-------+  |                                 |       |
               +----+--------------------------->|       |
                    |        +-------------+     |       |
                    |   +--->| Bottleneck -+---->|       |
                    |   |    +-------------+     |       |
                    |   |        .        ------>| Conv -+--->
                    +---+        .        ------>|       |
                        |        .        ------>|       |
                        |    +-------------+     |       |
                        +--->| Bottleneck -+---->|       |
                             +-------------+     +-------+       (C2f)
```

```python
class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))   # chunk：沿着 dim=1 均匀分割为 2 个 chunks
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
```

其中 Bottleneck 的网络结构为，

```sh
# c1          c_          c2        c2  
    +------+    +------+    +---+
-+->| Conv +--->| Conv +--->| a |
 |  +------+    +------+    | d +--->
 +------------------------->| d |
                            +---+

# 通常 c_ 要小于 c1，shortcut 的一个前提添加是：c1==c2
# 但是 yolov8 中，c_==c1
```

**SPPF**

```sh
-> conv -+-> maxpool -+-> maxpool -+-> maxpool -> c
         |            |            +------------> o ->
         |            +-------------------------> n
         +--------------------------------------> v         (SPPF)
```

```python
class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
```

整个网络结构如图 1，模型输入 size 为 640（见源码的 cfg/default.yaml 中的 imgsz）

![](/images/obj_det/yolov8_1.jpg)

<center>图 1</center>

输出部分如下图所示，

![](/images/obj_det/yolov8_2.png)

图 2 中浅蓝色的特征为网络的输出特征，与 FPN 的主要区别是：每个 scale 的输出特征既有高层语义又有低层语义，而 FPN 通过自上而下的融合，使得小 scale 的输出特征仅有高层语义。

预测包含了 3 个 scale 的特征

```sh
# name  下采样率 stride
P3      8
P4      16
P5      32
```

网络结构中 Detect 将分类和坐标检测头分开（解耦），且使用 anchor free ，如图 2 所示，

![](/images/obj_det/yolov8_1.png)

<center>图 2. 图源：参考文章</center>

> 本文参考了文章 https://mmyolo.readthedocs.io/zh_CN/latest/recommended_topics/algorithm_descriptions/yolov8_description.html，记为参考文章

图 2 中， yolov5 输出三个 scale 的特征，每个特征有独立的 Conv2d，然后输出经过 Loss 计算损失。yolov8 则对每个 scale 的特征有两组 Conv，对应图 2 中 Yolov8 的上下两行，表示 bbox 和 cls。



训练阶段，`Detect` 对 3 个 scale 的预测特征平面分别处理，每个 scale 特征均经过图 2 下方所示的检测头，检测头有一个分类分支和一个回归分支，分别经过 3 个 conv，得到分类输出 shape 为 `(b, nc, h, w)`，其中 `nc` 表示前景分类数量。回归输出 shape 为 `(b, 4 * reg_max, h, w)`，其中 `reg_max=16` ，两者 concat 为 shape  `(b, 4 * reg_max + nc, h, w)` ，这就是某个 scale 对应的预测输出。 

相关代码如下，

```python
def forward(self, x):
    """x: A list of 3 tensors, and each tensor's shape is (b, c, h, w)"""
    shape = x[0].shape  # BCHW
    for i in range(self.nl):
        # cv2 output is for box regression: (b, 4*reg_max, h, w)
        # cv3 output is for classification: (b, nc, h, w)
        x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    if self.training:
        return x # x 是一个list，包含 3 个 tensor，每个 tensor 形如 (b,4*reg_max+nc,h,w)
```

我们需要知道：yolov8 是 anchor free 模型，所以每个 anchor point 处均需要预测 4 个坐标值以及分类得分。yolov8 预测每个坐标的分布而非一个确定的值，将坐标分布离散化，使用 `reg_max=16` 个离散值，每个离散值预测对应的概率，计算分布的期望作为坐标的预测值，参见下文 (2) 式。

# 2. Loss 计算

loss 计算包括 如何分配正负样本和如何计算损失两部分。

yolov8 使用动态分配策略：TOOD 的 TaskAlignedAssigner，TaskAlignedAssigner 策略为：根据分类与回归的分数加权和选择正样本。

$$t = s ^ {\alpha} \times u ^ {\beta}\tag{1}$$

其中 $s$ 表示标注类别对应的预测得分，即，每个 anchor point 取所在 gt box 的分类得分，如果 anchor point 不在任何 gt box 内部，那么 $s=0$。

如果 anchor point 在 n 个 gt box 内部，那么保留这 n 个 gt box 对应的分类得分，因为实现过程中，使用一个得分矩阵，矩阵行数为 gt box 数量，矩阵列数为 anchor point 数量，故这个 anchor point 所在列将有 n 个值非零。

$u$ 是预测 box 与 gt box 的 IoU。$\alpha, \beta$ 为两个控制参数，分别控制 $s$ 和 $u$ 对 t 值指标的影响。

这里 $t$ 是一个指标，用于衡量分类任务的 anchor point 和定位任务的 anchor point 对齐程度，一个 anchor，如果既能有大的分类得分预测，又能有精确的定位，那么这个 anchor 就是很好对齐的。

基于 $t$ 指标，为每个 gt box 选择 top-K 大的 t 值对应的 anchor points 作为正样本，其他 anchor points 作为负样本。

代码中相关超参数选择为，

```python
self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
```

**损失** 包含分类损失和回归损失，没有置信度（objectness）损失。

1. 分类损失使用 BCE 损失（二值交叉熵损失），计算对象：所有 anchor points
2. 回归损失使用  DFL 损失（distribution focal loss），还使用了 CIoU 损失。

**问题**

1. 为什么取消了置信度预测之和，分类数量没有包含背景，即，为什么不使用 `nc+1`?
2. 为什么 `reg_max` 取值 16，即，如何选择合适的 `reg_max` 值？

这两个问题等下文介绍了损失如何计算之后再来解答。

## 2.1 动态分配样本

根据 (1) 式动态分配样本。

有 3 个 scale 的 feat maps，但是可以通过 concatenate 操作，看作是 $h _ 1 w _ 1 + h _ 2 w _ 2 + h _ 3 w _ 3$ 个 anchor points，本节为了表示简洁，记总共 anchor points 数量为 $3hw=h _ 1 w _ 1 + h _ 2 w _ 2 + h _ 3 w _ 3$ 。（anchor free 模型，每个特征平面的 point 可看作是一个 anchor）。

batch 数据中，目标 gt boxes 的 x1y1x2y2 坐标，数据 shape 为 `(b, max_obj_num,  4)`，其中 `b` 是 batch size，`max_obj_num` 是这批数据中单个图像的目标数量的最大值，显然这是经过填充的，使用一个 `mask_gt` 作为真实目标 box 坐标数据的掩码，易知掩码 shape 为 `(b, max_obj_num, 1)` 。

gt boxes 的分类 labels 的 shape 为 `(b, max_obj_num, 1)` ，前景分类 index 为 `0 ~ nc-1`。

<font color="magenta">模型预测输出点距离左上右下的 distance 数据 shape 为 `(b, 3hw, reg_max * 4)`</font>。这里使用 [DFL](/2023/08/13/obj_det/GFL)，box 坐标预测值是一个分布，离散化处理，则是 `reg_max` 个离散值，每个值对应一个概率，所以回归分支实际上预测的是每个离散值的概率，例如 anchor point 距离 box 左边的距离是一个离散型随机变量，取值范围为 `0, 1, ..., reg_max-1`，模型预测输出距离左边 distance 为一个 `reg_max` 长度的向量，表示各个离散值的概率，那么距离左边 distance 的预测结果为 

$$\hat y = \sum _ {x=0} ^ {reg \_ max -1} p(x) x \tag{2}$$

距离其余三个边的 distance 类似处理，处理后得到预测 boxes 的 x1y1x2y2 坐标，shape 为 `(b, 3hw, 4)` ，即，每个 anchor point 处有一个预测 box，需要注意，这里得到的预测 box 的坐标是基于各个 scale 特征平面的，也就是说，相对于原输入 image size，box 的坐标缩小到 `1/stride`，所以对于 P3 特征平面，输入图像上的目标 box size 的范围为 0 到 $reg \_ max*8$，这里取 reg_max=16，即 box size 最大为 128。对于 P5 特征平面预测的 box size 最大为 $16*32=512$。

<font color="magenta">模型预测分类得分 `pd_scores` 的 shape 为 `(b, 3hw, nc)`</font> 。

相关代码：

```python
class v8DetectionLoss:
    def __call__(self, preds, batch):
        feats = preds   # list of 3 tensors, 每个 tensor: (b, nc+reg_max*4, hi*wi)
        # pred_distri: (b, reg_max*4, 3*h*w)
        # pred_scores: (b, nc, 3*h*w)
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        # 生成密集 anchor 中心点坐标，偏移 0.5
        # anchor_points: (3*h*w, 2)     xy
        # stride_tensor: (3*h*w, 1)     每个anchor的stride
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        # 将归一化的 xywh 坐标转为非归一化的 xyxy 坐标（based on imgsz）
        # targets: (b, max_obj_num, 5)  5 表示 cls_id, x1, y1, x2, y2
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        # gt_labels: (b, max_obj_num, 1)
        # gt_bboxes: (b, max_obj_num, 4)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        # batch 中各个图像中的目标数量不等，全部对齐到 max_obj_num，才能组成 tensor
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0) # (b, max_obj_num, 1)
        # pred_bboxes 表示每个 anchor 处的预测 box 的 xyxy 坐标，坐标值范围：[0,reg_max-1]
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # (b, 3*h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_boxes, mask_gt
        )
        ...

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:    # True
            b, a, c = pred_dist.shape   # (b, 3*h*w, reg_max*4)
            # self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
            # 参见上文 (2) 式
            # pred_dist: (b, 3*h*w, 4) 这就是每个 anchor 处预测的anchor中心与 ltrb 四个边的距离
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

```

**# 动态分配**

有了以上数据说明，接下来看如何动态分配样本。调用语句为

```python
# pred_scores: 所有锚点处的预测分类归一化得分，(b, 3*hw, nc)
# pred_bboxes: 所有锚点处的预测 box xyxy坐标，基于相应 scale 特征平面即，但是乘上 stride，则映射到基于模型输入 imgsz。(b, 3*hw, 4)，4 表示 xyxy
# anchor_points: 所有anchor中心坐标，基于相应 scale 特征平面即，但是乘上 stride，则映射到基于模型输入 imgsz。(3*hw, 2)  2 表示 xy，注意batch中每个图像均使用相同的 anchor_points
# gt_labels: gt box 的分类 id，(b, max_obj_num, 1)
# gt_bboxes: gt box 的坐标，基于 image size 640，(b, max_obj_num, 4)，4 表示 xyxy
# mask_gt: gt box 的 mask，因为部分 gt box 是填充的，(b, max_obj_num, 1)
_, target_bboxes, target_scores, fg_mask, _ = self.assigner(
    pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
    anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)
```

下文我们谈到 anchor 与 gt box 的关联矩阵时，记住关联矩阵的 shape 为 `(b, max_obj_num, 3*h*w)`，其中行数为 `max_obj_num`，列数为 `3*h*w`，`b` 为 batch size。

1. 根据 gt boxes 的 x1y1x2y2 和 `3hw` 个 anchor points，筛选出位于 gt boxes 内部的 anchor points，得到筛选掩码 `mask_in_gt`，其 shape 为 `(b, max_obj_num, 3hw)`，再与 `mask_gt` 相乘，得到 <font color="magenta">最终掩码</font>（即，位于有效非填充 gt boxes 内部的 anchor points），shape 依然为 `(b, max_obj_num, 3hw)`，参考下方代码中的 `mask_in_gts * mask_gt`。

    相关代码为

    ```python
    # utils/tal.py
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        '''选择中心点坐标位于 gt box 内部的 anchors'''
        n_anchors = xy_centers.shape[0]     # 3*h*w
        bs, n_boxes, _ = gt_bboxes.shape
        # lt: (b*max_obj_num, 1, 2)
        # rb: (b*max_obj_num, 1, 2)
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        # (b*max_obj_num, 3*h*w, 4) -> (b, max_obj_num, 3*h*w, 4)
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(eps) # (b, max_obj_num, 3*h*w)
    
    # class TaskAlignedAssigner
    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)   # (b, max_obj_num, 3*h*w)
        _, _ = self.get_box_metrics(..., mask_in_gts * mask_gt)
        ...
    ```

2. 从预测得分中根据 gt 分类 label 取值，然后使用<font color="magenta">最终掩码</font>提取有效 anchor points 的预测分类得分。

    对 batch 中某个 image 其预测得分矩阵 shape 为 `(3hw, nc)`，这个 image 有 `max_obj_num` 个 gt boxes，每个 gt box 有一个分类 index，预测 anchor 与 gt box 的组成一个预测得分的关联矩阵，其 shape 为 `(max_obj_num, 3hw)`，每个 (anchor, gt box) pair 的预测分类得分从 `nc` 的得分中取 gt box 所属分类对应的那个得分。`b` 个 images 一共取得的预测得分为 `(b, max_obj_num, 3hw)` 。这个预测分类得分还需要使用最终掩码提取有效的 (anchor, gt box) pair，相关代码为

    ```python
    # class TaskAlignedAssigner
    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        ...
        na = pd_bboxes.shape[-2]    # 3*h*w
        # (b, max_num_obj, 3*h*w)
        mask_gt = mask_gt.bool()    # 最终掩码，(b, max_obj_num, 3*h*w)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)
        # (b, max_num_obj)  
        ind[0] = torch.arange(self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        # (b, max_num_obj)
        ind[1] = gt_labels.squeeze(-1)
        # 首先根据 pd_scores 和 gt_labels 组成一个关联矩阵 (b, max_obj_num, 3*h*w)
        # 注意 pd_scores[ind[0], :, ind[1]] 的结果，其 shape 为 (*ind[0].shape, pd_scores.shape[1])
        # 若是 pd_scores[ind[0], :, :, ind[1]]，其 shape 为 (*ind[0].shape, pd_scores.shape[1], pd_scores.shape[2])
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]
    ```

    预测分类得分提取结果记为 `bbox_scores`，参见上面代码，这个 `bbox_scores` 就是 (1) 式中的 $s$ 。

3. 类似第 `2` 步，提取预测 boxes 坐标，然后再根据最终掩码提取有效位置处的预测坐标，其 shape 为 `(N, 4)`，N 为最终掩码中 true 元素数量。gt boxes 坐标数据 shape 为 `(b, max_obj_num, 4)`，通过扩充（repeat）操作得到 `(b, max_obj_num, 3hw, 4)`，然后同样的取最终掩码为 true 位置处的值，得到 `(N, 4)`。

    batch 数据中总共 `(b, max_obj_num)` 个 gt boxes，每个 gt boxes 提取 `(3hw, 4)` 的预测 boxes 坐标，那么一共 `(b, max_obj_num, 3hw, 4)` 的提取数据（anchor 与 gt boxes 的关联矩阵），然后取最终掩码 `(b, max_obj_num, 3hw)` 中 true 位置处的值。

    ```python
    # class TaskAlignedAssigner/def get_box_metrics

    # mask_gt: (b, max_obj_num, 3*h*w)
    # (b, 3*h*w, 4) -> (b, 1, 3*h*w, 4) -> mask 筛选 -> (N, 4)
    pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
    # (b, max_obj_num, 4) -> (b, max_obj_num, 1, 4) -> mask 筛选 -> (N, 4)
    gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
    ```

4. 第 `3` 步提取的预测 boxes 与 gt boxes，计算 CIoU，由于两个 tensor shape 均为 `(N, 4)`，计算出来的 CIoU 为 `(N, )` 向量。创建一个 `overlaps`，其 shape 为 `(b, max_obj_num, 3hw)`，根据最终掩码为 true 的位置，将 CIoU 设置到 `overlaps` 中。这个 `overlaps` 就是 (1) 式中的 $u$ 。

    ```python
    overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
    # gt_boxes 和 pd_boxes 均为 (N, 4)，均基于模型输入尺寸 imgsz(640)，4 表示 xyxy 坐标
    overlaps[mask_gt] = bbox_iou(gt_boxes, pd_boxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)
    ```

5. 根据 (1) 式计算 $t$ 值，使用超参数 $\alpha=0.5, \ \beta=6.0$ ，代码如下：

    ```python
    # (b, max_obj_num, 3hw)
    align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
    ```

    易知， `align_metric` 的 shape 为 `(b, max_obj_num, 3hw)`，每个 gt box 有 `3hw` 个 $t$ 值，最终掩码 true 值位置处值有效，其他值均为 0 。

6. 将 `align_metric` 沿着 `dim=-1` 取 topK（源码中 topk 为 10） 大的值，以及对应的位置，即，每个 gt box 取 topK 大 $t$ 值对应的 anchor points 作为正样本。

    取出来的 topK 大 anchor points 实际上还需要使用 `mask_gt` 做掩码过滤，因为 `align_metric` 有 `(b, max_obj_num)` 个 gt boxes，显然要过滤掉填充的假 gt boxes 。最后所选择的 topK 的 anchor points 也使用掩码表示，记为 `mask_topk`，其 shape 为 `(b, max_obj_num, 3hw)`，一共 `b * max_obj_num` 个 gt boxes，每个 gt box 的 `3hw` 长度向量 mask 中，有 K 个 元素为 true 或者全部为 false（这种对应填充的假 gt box，见下面代码中的 `count_tensor.masked_fill_` 方法）。

    ```python
    def select_topk_candidates(self, metrics, largest=True, top_mask=None):
        '''
        metrics: 就是前面的 align_metric，表示 t 值，(b, max_obj_num, 3*h*w)
        top_mask: gt box 掩码，见上文的 mask_gt，(b, max_obj_num, 10)，但是最后一维 expand 为 topk=10
        '''
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)
        topk_idxs.masked_fill_(~topk_mask, 0)   # 没有 gt box 的地方，topk 的 anchor index 全部为 0

        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # 每次在 b*max_obj_num 个元素位置上累加 1
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k:k+1], ones)
        # > 1 表示无效，重置为 0，只有等于 1 的才有效
        count_tensor.masked_fill_(count_tensor > 1, 0)
        return count_tensor.to(metrics.dtype)
    ```

7. `mask_topk` 与上述 `mask_in_gt` 和 `mask_gt` 三者按 elementwise 相乘，得到用于提取 **正样本** 的掩码，记为 `mask_pos`，其 shape 为 `(b, max_obj_num, 3hw)` 。

    ```python
    # (b, max_obj_num, 3*h*w)，属于 topk 的 anchor，其在关联矩阵中值为 1，否则为 0
    mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
    # 三个关联矩阵的掩码相乘，得到正样本的掩码，即
    # 一个 anchor 对 一个 gt box 而言，是否是正样本
    mask_pos = mask_topk * mask_in_gts * mask_gt    
    ```
8. 上一步得到 anchor 对 gt box 而言是否是正样本，但是某个 anchor 中心可能被多个 gt boxes 包含，那么与这个 anchor 的预测 box 有最大 CIoU 的 gt box 被选中。

    ```python
    # mask_pos: (b, max_obj_num, 3*h*w) 正样本掩码的关联矩阵
    # overlaps: (b, max_obj_num, 3*h*w) CIoU 的关联矩阵
    # self.n_max_boxes: 就是 max_obj_num
    target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)
    ```

    以某一个 image 为例，那么正样本掩码 shape 为 `(max_obj_num, 3hw)`，这是一个矩阵，即上面的 `mask_pos`，对此矩阵按行求和，那么得到每个 gt boxes 所对应的正样本数量。按列求和，那么能判断每个 anchor point 被用作了几次正样本，或者是，有几个 gt boxes 将这个 anchor point 作为正样本，如果某个 anchor point 被不止 1 个 gt boxes 看作正样本，那么这个 anchor point 只选择具有最大 CIoU 的那个 gt box，也就是说其余 gt box 不再将这个 anchor point 看作正样本，经过这样的调整后的正样本掩码仍用变量 `mask_pos` 存储 ，相关代码为，

    ```python
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        # (b, 3*h*w)
        fg_mask = mask_pos.sum(-2)  # 关联矩阵，沿列求和，得到每个 anchor 是多少个 gt box 的正样本
        if fg_mask.max() > 1:  # 存在现象：one anchor is assigned to multiple gt_bboxes
            # (b, max_obj_num, 3*h*w) anchor 是多重正样本的掩码
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
            # CIoU 关联矩阵，对列求最大值，得到每个 anchor 的最大 CIoU 对应的 gt box index
            max_overlaps_idx = overlaps.argmax(1)  # (b, 3*h*w)
            # (b, max_obj_num, 3*h*w)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            # 对 dim=1 散列赋值，这是因为 max_voerlaps_idx 是最大 CIoU 对应的 gt box index
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            # 如果 anchor 是多重正样本，那么只有最大 CIoU 的 gt box 被选中为 1，其他 gt box index 位置处为 0
            # 如果 anchor 不是多重正样本，那么根据前面 t 值决定 anchor 是否是正样本
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, max_obj_num, 3*h*w)
            fg_mask = mask_pos.sum(-2)  # 此时，fg_max 最大值不超过 1，因为每个 anchor 最多只作为一个 gt box 的正样本
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)，正样本 anchor 对应的 gt box index
        return target_gt_idx, fg_mask, mask_pos
    ```

    此时，我们得到正样本掩码的关联矩阵 `mask_pos`，其中每列最多只有一个 true 值，`fg_mask` 表示每个 anchor 作为正样本时与之关联的 gt box 数量，显然 `fg_mask` 最大值为 1，所以 `fg_mask` 也表示 anchor 是否是正样本的掩码，`target_gt_idx` 表示正样本 anchor 对应的 gt box index。

9. 根据上一步得到的 gt box index，以及 gt box 的坐标和分类 id，组装成 target，用于计算 loss

    ```python
    def get_targets(self, gt_labels, gt_boxes, target_gt_idx, fg_mask):
        # (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        # 正样本对应的 gt box index 本来只是单个图像内的 index，范围 0~max_obj_num-1
        # 现在将 gt box index 映射到整个 batch，即第一个图像中的范围 0~max_obj_num-1
        # 第二个图像中的范围 max_obj_num~2*max_obj_num-1，依次类推
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, 3*h*w)
        # gt_labels: (b, max_obj_num, 1)
        # target_labels: 正样本对应 target 分类 id
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, 3*h*w)
        # (b, 3*h*w, 4) 正样本对应的 target 坐标，基于 imgsz(640)
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]    
        target_labels.clamp_(0)
        target_scores = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.num_classes), ...)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes) # 正样本分类得分的掩码
        # 这里使用正样本得分掩码进行过滤。为什么？
        # 这是因为 target_gt_idx 的值范围 0~max_obj_num-1，这表示每个 anchor 都关联一个 gt box，
        # 显然对于非正例 anchor，没有对应的 gt box，所以 target_labels 中负例 anchor 对应的 target 分类 id 错了，
        #从而 target_scores 中负例 anchor 对应的 target 分类得分也错了，需要用 fg_scores_mask 纠正过来
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)
        # (b, 3*h*w), (b, 3*h*w, 4), (b, 3*h*w, nc)
        return target_labels, target_bboxes, target_scores
    ```


## 2.2 计算损失

有了以上说明，我们来看如何计算损失：

1. 分类损失为所有样本（**包括正负样本，所有 anchor points**）的 BCE 损失。因为取消了置信度损失，所以这里分类损失除了考虑正样本，还需要考虑负样本。
    
    `target_scores`: shape 为 `(b, 3*hw, nc)`，为每个 anchor point 设置 gt label。只有正样本 anchor points 处有 one-hot 向量，此向量长度为 `nc`，向量中 正样本对应的 gt box 分类 id 处的元素值为 1，但是这里实际上不使用 `1` 作为 target 值，而是 

    $$\hat t _ {ij} \cdot \max(CIoU _ i) = \frac {t _ {ij}}{\max (\mathbf t _ i)} \cdot \max (CIoU _ i)\tag{3}$$

    以单个 image 为例理解上式更容易些，$t$ 值关联矩阵 shape 为 `(max_obj_num, 3*hw)`，对于每一个 gt box 编号为 `i` 分布进行归一化得到 $\hat t$ ，即关联矩阵的每一行单独进行归一化后得到 $\hat t$。然后 CIoU 关联矩阵 shape 也是 `(max_obj_num, 3*hw)`，一个 gt box 对应多个正样本 anchor points，自然就有多个 CIoU 值，求出最大的 CIoU ，记为 $CIoU _ i$ ，即 CIoU 关联矩阵求每一行的最大值，以此值为这个 gt box 的分类得分基准，那么与这个 gt box 匹配的正样本 anchor points 的分类得分 target 就是这个基准乘上 $\hat t$ 。负样本 anchor point 的分类得分向量则是全 0 向量。相关代码为，

    ```python
    # class TaskAlignedAssigner

    # (3) 式中的 t_{ij}
    align_metric *= mask_pos    # 对 t 关联矩阵应用正样本掩码，这与负样本的 t 值均为 0
    pos_align_metrics = align_metric.amax(dim=-1, keepdim=True) # t 关联矩阵每行求最大值，即 (3) 式中 max(t_i)
    # CIoU 关联矩阵求每行最大值，即 (3) 式中的 max(CIoU_i)
    pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
    # 按 (3) 式计算后 (b, max_obj_num, 3*h*w)，再求每一列最大值，(b, 3*h*w, 1) 得到每个 anchor 的归一化 t 值
    norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
    target_score = target_scores * norm_align_metric    # 正样本 anchor 对应的 gt 分类 index 处的归一化 t 值被保留，作为分类得分的 target，其余分类 index 处以及负样本所有分类 index 处的 target 值均为 0
    ```
    
    分类损失使用 BCE，正负样本均参与计算，相关代码为，

    ```python
    target_scores_sum = max(target_scores.sum(), 1) # 计算正样本分类得分的 target 值的总和
    # pred 和 target 分类得分 (b, 3*h*w, nc) 计算 BCE 损失之和，然后除以正样本分类得分的 target 值的总和
    # 注意，不是对 BCE 的结果求平均
    loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
    ```

2. 坐标损失，只考虑正样本的坐标损失。损失包含两种：

    - CIoU 损失：$1-CIoU$ 。CIoU 损失还考虑了权重，使用分类得分 `target_scores` 作为权重，分类得分 target 实际上考虑了定位质量，值越大，越是需要注重学习优化，所以使用 `target_scores` 作为权重是合理的 。计算加权平均。

        ```python
        # (b, 3*h*w) -> (N, 1)  N 表示这个 batch 中所有 gt box 数量之和
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)   # 每个正样本的权重 
        # target_bboxes 除以了 stride，即映射到基于特征平面 size，pred_bboxes 也是基于特征平面 size
        # 计算 CIoU (N, N)，第一个维度表示 anchor，第二个维度表示 gt box
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        # 计算加权平均
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        ```

    - DFL 损失：

    $$DFL(\mathcal S _ i, \mathcal S _ {i+1}) =-((y _ {i+1}-y)\log \mathcal S _ i + (y - y _ i)\log \mathcal S _ {i+1}) \tag{4}$$
    
    上式就是 [GFL](/2023/08/13/obj_det/GFL) 一文中的 (6) 式。下面给出计算 DFL 的代码，然后根据代码进行说明。

    ```python
    # class BboxLoss

    # 根据 anchor 中心点和 gt box 的 x1y1x2y2 值，计算 anchor 中心与 gt box ltrb 之间的距离
    # 距离值 clamp(0, 15-0.01)，这里 self.reg_max=16-1
    # anchor_points: (3*h*w, 2) 中心点 xy 坐标
    # target_bboxes: (b, 3*h*w, 4)，所有 anchor 对应的 gt box x1y1x2y2 值，负样本对应的 gt box 坐标值为 0
    target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
    # pred_dist: (b, 3*h*w, 16*4) 预测的与 ltrb 距离的值的分布（0~15的各个值的概率）
    # fg_mask: (b, 3*h*w) 正样本掩码
    # loss_dfl: (N, 1)
    loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max+1), target_ltrb[fg_mask])*weight
    loss_dfl = loss_dfl.sum() / target_scores_sum   # 加权平均

    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # pred_dist: (N * 4, 16)
        # target: (N, 4)， left top right bottom distance
        # 返回值：(N, 1)

        tl = target.long()  # target left，向下取整
        tr = tl + 1  # target right，向上取整
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)
    ```

    这个函数的参数为正例的 box 预测值，实际上是正例 anchor point 距离预测 box 左上右下 4 个 distance 的离散概率分布的预测，即一个 anchor point 预测 4 个向量，每个向量表示一个 distance 的概率分布，记 batch 中一共 `N` 个正样本。4 个 distance 的 target 值为变量 `target`，其 shape 为 `(N, 4)`，也就是 (4) 式中的 $y$ 。`tl` 表示 (4) 式中的 $y _ i$，`tr` 为 $y _ {i+1}$ ，`wl` 为 $y _ {i+1}-y$， `wr` 为 $y - y _ i = y - (y _ {i+1}-1)=1-(y _ {i+1}-y)$，这两个权重值与线性插值的权重思想相同。

    中心点与 ltrb 的距离值的预测概率分布为 `pred_dist`，target 值本来是浮点数，取其左右两个相邻的整数，作为新的 target，然后这就是一个分类问题，类别为 `0,1,...,15` 共 16 个分类，计算交叉熵损失。与 ltrb 的四个距离，每个距离的预测使用左右两个整数距离进行监督，对两个交叉熵损失计算加权平均（类似于线性插值）。

    然后计算得到的 DFL 损失 与 CIoU 损失一样，使用正样本的归一化 t 值做加权平均，即代码中的 `target_scores`。

## 2.3 总结

对于一个 image，其中有 `max_obj_num` 个 gt boxes，`3*h*w`（实际上是 $\sum_{i=1}^3 h_i w_i$） 个 anchor points，那么可以计算 CIoU 关联矩阵，shape 为 `max_obj_num x 3hw`，每行表示一个 gt box，每列表示一个 anchor point，实际上图像中的 gt box 数量可能小于 `max_obj_num`，所以有一个 gt mask。

需要注意，这里回归分支预测输出不是 anchor points 距离预测 box 的左上右下 4 个 distance 具体的值，而是 4 个离散 distance 的概率分布，distance 就是一个离散随机变量， 取值范围为 `0 ~ reg_max-1` 是固定的预先设置好的，预测的是 distance 的概率分布，分布的期望，作为 distance 的预测值。

正样本不仅仅是指某个 anchor point ，还包含与之相关联的 gt box，即，这个 anchor point 位于这个 gt box 内部，也可以称作正样本对 `(anchor point, gt box)` 。那么如果 anchor point 位于多个 gt boxes 内部呢？根据 CIoU 矩阵这个 anchor point 所在列求最大值，最大值所在行，决定了 anchor point 关联到哪个 gt box 上，其他 gt boxes 不再与这个 anchor point 关联。

上面正样本的掩码关联矩阵中，仅保证了每列最多一个 true 值，但是每行可能有多个 true 值，这表示 一个 gt box 有多个 anchor 与之关联作为正样本，通过计算 (1) 式每行选择 top-k 的 anchor points 为正样本，其余正样本改为负样本（原来负样本 anchor 的还是负样本）。


### 2.3.1 分类

正样本 anchor point 的分类 target 是一个 one-hot 向量，向量长度为分类总数 `nc` ，向量中 `1` 值所在位置就是 anchor point 关联的 gt box 的分类 id （范围为 `0 ~ nc-1`）。负样本分类 target 则是全 `0` 向量。

由于分类预测最佳 anchor point 与回归预测最佳 anchor point 不对齐，那么 NMS 时会导致差的预测 anchor point 抑制好的预测 anchor piont，所以为了对齐，正样本分类 target 的 one hot 向量中的 `1` 值替换为 (3) 式计算结果。(3) 式基于三点考虑：

1. 同一 gt box 内部不同正样本 anchor points 的分类 target 应该与各自的 CIoU 成正比，这样 CIoU 最大的 anchor point 既是分类预测最佳 anchor point，又同时是回归预测最佳 anchor point。作者实现中，使用的是 $t$ 值，即，分类 target 与 $t$ 值成正比，为了防止 t 值超过 1，对 t 值按行归一化为 $\hat t$。

2. 不同 gt boxes 对应的正样本 anchor points，其分类 target 也应该不同。为每个 gt box 设置一个分类 target 基准，对 CIoU 关联矩阵按行取最大值，那么每行最大值就是这个 gt box 的基准。

所有样本的分类 target 应该是 shape 为 `(h*w, nc)` 的矩阵，每行是一个类似 one-hot 向量（其中 `1` 由 (3) 式替代）

### 2.3.2 回归

与分类损失不同，**回归损失仅考虑正样本**

回归损失包含两部分：CIoU 损失和 DFL 损失。

CIoU 损失为 $1-CIoU$，根据前述的 CIoU 矩阵，取正样本的 CIoU 代入计算即可。

DFL 损失根据 (4) 式，也很容易计算。

CIoU 损失和 DFL 损失均使用加权平均，权重为正样本的分类 target，这在上一小节 `### 2.3.1` 已讲。


## 2.4 预测

调用示例，

```python
from ultralytics import YOLO
# Load a model
model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model
# Run batched inference on a list of images
results = model(["im1.jpg", "im2.jpg"])  # return a list of Results objects
```

### 2.4.1 LetterBox 处理

原始 image 的 高宽为 $H_0, W_0$，模型输入 size 为 $H, W$，计算最小比例

$$f=\min(H/H_0 , W/ W_0 )$$

这样可以保证经过 scale 之后的 image 高宽分别不超过 $H, W$，经过 scale 之后的 size 为

$$H_1 = f H_0, \quad W_1 = f W _ 0$$

当然我们还要对其进行四舍五入，毕竟 size 值必须是整数，

$$H_1 := \lfloor H_1+0.5 \rfloor, \quad W_1 := \lfloor W_1 + 0.5 \rfloor$$

这与模型输入 size 的差为

$$\Delta W = W - W_1 , \quad \Delta H = H - H _ 1$$

本来是要 padding，使得 image size 为模型输入 size，padding 的大小就是 $\Delta W, \ \Delta H$，但实际上我们不需要保证模型输入 size 必须是 $H, W$，也可以比 $H, W$ 小，由于模型的最大 stride 为 32，所以我们需要 padding，使得 $H_1, W_1$ 为 32 的整数倍，padding size 变为

$$\Delta W := \Delta W \\% 32 , \quad \Delta H := \Delta \\% 32$$

左右两侧和上下两侧进行 padding，每侧 padding size 为

$$\Delta W := \Delta W / 2, \quad \Delta H := \Delta H / 2$$

注意，上式可能又引入了小数，而 padding size 必须要是整数，所以采取措施，left 和 top 的 padding size 为

$$l =\lfloor \Delta W - 0.1 + 0.5 \rfloor, \quad t = \lfloor \Delta H - 0.1 + 0.5 \rfloor$$

right 和 bottom 的 padding size 为

$$r =\lfloor \Delta W + 0.1 + 0.5 \rfloor, \quad b = \lfloor \Delta H + 0.1 + 0.5 \rfloor$$

当 $\Delta W$ 是整数时，$l = r$。

当 $\Delta W$ 是小数时，其小数部分是 0.5，此时 $r=l+1$

$\Delta H$ 情况完全相同。

所以 letterbox：先将图像 resize 到 $(H_1, W_1)$，然后在 ltrb 四个边进行填充。

### 2.4.2 Detect head

预测阶段，模型前向传播到 Detect head 之后，还要进行一些处理：

1. 生成 anchor 中心点和每个 anchor 对应的 stride，这是因为每次预测的模型输入 size 可能不同，原因见上面 letterbox。

    ```python
    # class Detect
    
    # (2, 3*h*w), (1, 3*h*w)
    self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
    self.shape = shape  # (b, c, h1, w1) 记录 P3 的特征 size，下次若模型输入 size 相同，那么不必调用 make_anchors
    ```
2. 模型输出分成坐标和分类两个部分

    ```python
    # x_cat: (b, no, 3*h*w)
    x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
    # (b, reg_max*4, 3*h*w), (b, nc, 3*h*w)
    box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    ```

3. 坐标预测值是 4 个距离的概率分布，计算 4 个距离的期望，作为 4 个距离的预测值，并联合 anchors 计算预测框的 xywh 坐标

    ```python
    dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
    ```

    这里 `self.dfl` 其实是一个卷积，卷积权重就是 `0~reg_max-1`，这样卷积结果就是求期望，卷积结果的 shape 为 `(b, 4, 3*h*w)` 。

    然后使用 `dist2bbox`，根据 anchors 中心坐标和预测的与 ltrb 4 个距离值，计算出预测框的 xywh 坐标，结果 shape 为 `(b, 4, 3*h*w)`

4. 分类得分预测值进行归一化

    ```python
    y = torch.cat((dbox, cls.sigmoid()), 1)     # (b, 4+nc, 3*h*w)
    ```

### 2.4.3 postprocess

**# 非极大抑制 NMS**

上一步我们得到的预测数据是 xywh 的坐标（非归一化，基于模型输入 size）和分类得分，tensor shape 为 `(b, 4+nc, 3*h*w)`。

1. 使用 conf 阈值初筛（conf 默认值为 0.25）

    ```python
    # prediction: (b, 4+nc, 3*h*w)
    xc = prediction[:, 4:].amax(1) > conf_thres # (b, 3*h*w)
    ```
2. xywh 转 xyxy

    ```python
    prediction = prediction.transpose(-1, -2)   # (b, 3*h*w, 4+nc)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    ```

3. batch 中每个图像单独做 NMS。

    ```python
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]   # (n, 4+nc)  n 是满足 conf 阈值的预测框数量
        if not x.shape[0]: continue     # 没有预测到目标

        box, cls, mask = x.split((4, nc, nm), 1)    # 这里 mask 列数为 0，故不用考虑 mask
        conf, j = cls.max(1, keepdim=True)  # 只考虑分类得分最大的那个类
        x = torch.cat((box, conf, j.float(), mask), 1)  # xyxy，score，cls_id
        n = x.shape[0]
        if n > max_nms: # 超过 30000，则仅保留分类预测得分 top30000 的预测结果
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]   # top30000 从大到小排序
        # 这里 NMS 要按分类来，即只有相同分类的预测框才会做 NMS 操作
        # 所以将 xyxy 值根据分类 id 映射到一个范围，不同分类 id 的范围没有交集，从而保证不同分类的预测框
        # 不会被 NMS 筛掉
        # max_wh = 7680，约定模型输入的 w 和 h 均不可能超过 7680
        c = x[:, 5:6] * (0 if agnostic else max_wh) # 这里 agnostic = False，表示只有相同分类内做 NMS
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        i = i[:max_dec] # 这一步仅保留 top300
        output[xi] = x[i]
    return output
    ```

**# scale back to origin size**

前面 letterbox 中将图像 scale 到接近模型输入 size，预测出结果后，还需要 scale back 到原图 size。记此次预测模型的输入 size 为 $H, W$，原图 size 为 $H_0, W_0$，这 4 个值是已知的

根据 letterbox 操作，

$$H= f H_0 + t + b, \quad W = f W _ 0 + l + r$$

这里，$f, l, t, r, b$ 分别表示 scale ratio，ltrb 四个方向上的 padding size，如果保存了这 5 个值，那么很容易将预测框映射回原图，记预测框坐标为 $(x_1,y_1,x_2,y_2)$：

1. 去掉 padding 后预测框坐标为

    $$x_1:=x_1-l, y_1:=y_1-t, x_2:=x_2-l, y_2:=y_2-t$$

2. scale back

    $$x_1:=x_1/f, y_1:=y_1/f, x_2:=x_2/f, y_2:=y_2/f$$

如果没有保存这 5 个值，也可以计算出来：

1. 计算 scale factor

    $$f' = \min(H/H_0, W/W_0)$$

2. 计算 padding size

    $$l' = \lfloor(W - f' W_0)/2 - 0.1 + 0.5\rfloor, \ t' = \lfloor(H - f' H_0)/2 - 0.1 + 0.5\rfloor$$

3. 映射回原图

    $$x_1:=(x_1-l')/f', \ y_1:=(y_1-t')/f' \\\\ x_2:=(x_2-l')/f', \ y_2:=(y_2-t')/f'$$

这里的 $f', l', t'$ 与原来的 $f, l, t$ 其实是相等的，所以映射回原图的坐标是准确的，这是因为：

letterbox 计算图像矩形的宽高与模型输入宽高的两个比例，选择其中一个较小比例 $f$ 进行缩放，记这个比例对应的边为`边1`，那么 `边1` 是不需要 padding 的，而`边2` 则可能需要 padding，这样 `边2` 的缩放比例更大，所以 $f'$ 仍然使用 `边1` 对应的缩放比例，即 $f'=f$ ，于是计算 $l', t'$ 与原来的 $l, t$ 也分别相等。

# 3. 分割任务