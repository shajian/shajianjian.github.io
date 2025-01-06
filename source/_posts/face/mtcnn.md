---
title: MTCNN 论文解读
date: 2024-04-17 09:22:37
tags: face detection
---

论文：[Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878)


# 1. 简介

本文提出了一种级联的多任务框架，实现人脸检测和对齐，此方法可以挖掘这两个任务之间的联系，从而提高性能。

级联的 CNN 网络包含三个阶段：

1. 通过一个较浅 CNN 快速生成候选窗口
2. 通过一个较复杂 CNN 去掉大部分不包含人脸的窗口
3. 使用一个更强的 CNN 精调窗口并输出人脸 landmarks

本工作的主要贡献：

1. 提出级联 CNN 用于人脸检测和对齐，设计轻量 CNN 框架达到 real time 性能
2. 使用在线难例挖掘，提高性能
3. 执行大量实验，在 benchmark 上展示了明显的性能提升

# 2. 方法

## 2.1 框架

整个方法管线如图 1 所示，

![](/images/face/mtcnn_1.png)


对于一个图片，首先 resize 到不同尺度，建立一个图像金字塔，这就是 3-stage 级联网络的输入。

**stage 1** 

一个全卷积网络，称作 P-Net (Proposal Network) 用于获取候选窗口，以及它们的 bbox 回归向量，然后使用这些 bbox 回归向量校正候选窗口，再使用 NMS 筛选候选窗口。

**stage 2**

将候选窗口喂给另一个 CNN，称作 R-Net (Refine Network)，进一步过滤掉了大量的错误候选窗口，再进行 bbox 回归校正和 NMS 筛选。

**stage 3**

与 stage 2 类似，但是这个阶段得到更详细的人脸信息。

整个网络结构如图 2 所示，

![](/images/face/mtcnn_2.png)

P-Net 输入 size `12x12x3`

R-Net 输入 size `24x24x3`

O-Net 输入 size `48x48x3`

## 2.2 训练

根据三个任务：人脸/非人脸 分类、bbox 回归和人脸 landmark 定位来训练网络。

**# 人脸/非人脸 分类**

记样本为 $x _ i$，使用交叉熵作为损失函数

$$L _ i ^{det} = -(y _ i ^ {det} \log p _ i + (1- y _ i ^ {det})(1 - \log p _ i)) \tag{1}$$

其中 $p _ i$ 为预测得分，$y _ i ^ {det} \in \{0, 1\}$ 为 gt label。

**# bbox 回归**

对于每个候选窗，预测 box 与最近 gt box 对应，损失函数为，

$$L _ i ^ {box} = ||\hat y _ i ^ {box} - y _ i ^ {box}|| _ 2 ^ 2 \tag{2}$$

其中 $\hat y _ i ^ {box}$ 为 bbox 回归预测，$y _ i ^ {box}$ 为 gt box，包括 left top，height 和 width 。

**# 5 landmarks**

损失函数为

$$L _ i ^ {landmark} = ||\hat y _ i ^ {landmark} - y _ i ^ {landmark}|| _ 2 ^ 2 \tag{3}$$

其中 $\hat y _ i ^ {landmark}$ 为 landmark 回归预测，$y _ i ^ {landmark} \in \mathbb R ^ {10}$ 为 gt landmark。

**# 多源训练**

训练过程中有不同类型的训练图片：人脸、非人脸和部分对齐的人脸，所以损失函数 (1)~(3) 中有的损失函数用不到。例如对于背景窗口，仅仅计算分类损失 $L _ i ^ {det}$，而两个回归损失则不需要，使用样本类型指示器，那么总的训练目标为

$$\min \ \sum _ {i=1} ^ N \sum _ {j \in \{det, box, landmark\}} \alpha _ j \beta _ i ^ j L _ i ^ j \tag{4}$$

其中 $N$ 是训练样本数量，$\alpha _ j$ 为损失类型的权重因子。

本文设置 P-Net 和 R-Net 中使用 $\alpha _ {det} = 1, \alpha _ {box} = 0.5, \alpha _ {landmark} = 0.5$，而 O-Net 中设置 $\alpha _ {det} = 1, \alpha _ {box} = 0.5, \alpha _ {landmark} = 1$ 以便获得更精确的 landmarks 坐标。

$\beta _ i ^ j \in \{0, 1\}$ 是样本类型指示器。

**# 在线难例挖掘**

在人脸分类任务中进行在线难例挖掘。

每个 minibatch 中，根据分类损失排序，选择 top 70% 作为难例，然后使用这 70% 的样本的梯度进行反向传播。实验显示这个策略效果很好。

# 3. 实验

## 3.1 训练数据

1. 负样本

    与所有 gt box 的 IOU 均小于 0.3 的 box

2. 正样本

    与某个 gt box 的 IOU 大于 0.65 的 box

3. part face

    0.4 <= IOU <= 0.65

4. landmark face

    有 5 个 landmark 的人脸

负样本和正样本用于人脸分类任务。正样本和 part face 用于 bbox 回归任务。landmark face 用于 landmark 定位任务。

1. P-Net 

    从 WIDER FACE 数据集的图片中随机 crop 若干 patches，得到正负样本和 part face。从 CelebA 数据集中 crop faces 作为 landmark face。

    - 负样本。 在范围 `[12, min(im_h, im_w)/2)` 中随机选择 size，然后随机选择 crop 区域的左上角。计算这个 crop box 与图像中所有 gt boxes 的 IoUs，如果均小于 0.3，那么这个 crop box 为负样本，但是需要将 crop box resize 到 (12, 12) 。

    - 正例周围的负样本（难样本）。在每个 gt box 附近随机 crop 5 个 boxes。gt box 的 size 必须大于 20，否则太小不使用。在 [12, min(im_h, im_w)/2] 范围中随机选择 crop size。记 gt box 坐标为 (x1, y1, w, h)，那么在范围 `[max(-size, -x1), w)` 中随机选择一个值，作为 x1 的偏差值，即 crop box 的 left 与 gt box left 的偏差，这个范围左侧值保证了 crop box 的 left 值 > 0，范围右侧值保证了 crop box left < gt box right，y1 的偏差也是如此，这样就保证了 crop box 与 gt box 有交叠 intersection，从而 IOU > 0。计算这个 crop box 与所有 gt boxes 的 IoUs，如果最大 IoU < 0.3，那么 crop box 为负样本，同样需要 resize 到 (12, 12)。

    - 正样本/part face。在范围 `[min(w, h) * 0.8, 1.25*max(w, h))` 范围中随机取值作为 crop size，这里 w 和 h 是当前 gt box 的 size，这个范围保证了 crop size 比较接近 gt box size。crop center 与 gt box center 的偏差则分别为 w 和 h 的 0.2 倍内上下浮动。计算 crop box 与当前 box 的 IoU，如果 >= 0.65，则是正样本，如果 >= 0.4，则是 part face。这里没有计算 crop box 与所有 gt boxes 的 IoUs，因为会遍历每一个 gt box ，通过此方法寻找相应的正样本和 part face，所以对每个当前 gt box，我们只要寻找这个 gt box 的正样本和 part face 即可，故只要求 crop box 与当前 gt box 的 IoU，判断随机 crop box 是否是当前 gt box 的正样本/part face，如果是，那么 target 为：

        ```python
        delta_x = npr.randint(-w * 0.2, w * 0.2)    # 中心 x 坐标偏差
        delta_y = npr.randint(-h * 0.2, h * 0.2)

        # x1+w/2： gt box 中心 x 坐标
        # x1+w/2+delta_x: crop box 中心 x 坐标
        nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))  # crop box left
        ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))  # crop box top
        nx2 = nx1 + size    # crop box right
        ny2 = ny1 + size    # crop box bottom
        # target：
        offset_x1 = (x1 - nx1) / float(size)
        offset_y1 = (y1 - ny1) / float(size)
        offset_x2 = (x2 - nx2) / float(size)
        offset_y2 = (y2 - ny2) / float(size)
        ```

        left top right bottom 偏差除以 size，以相对值作为 target，因为训练阶段，crop image 需要 resize 到 (12, 12)，但是这个偏差比例不变，用作 target 是合适的。

        注意：gt box 图像没有直接作为正样本（其实是可以作为正样本的，不知道为啥没有使用）

    - landmark 训练数据的准备。标注数据的格式为，

        ```sh
        # 图像文件路径  人脸 gt box 坐标    5 个 landmarks 坐标
        im_path left right top bottom x1 y1 x2 y2 x3 y3 x4 y4 x5 y5
        ```
        根据人脸 gt box 坐标 crop 出人脸图像，然后 landmarks 坐标归一化如下，然后作为 landmark 任务的 target：
        
        ```python
        rx = (x1 - left) / (right - left)
        ry = (y1 - top) / (bottom - top)
        ```

        人脸需要 resize 到 (12, 12)，target 是比例，其值不再需要调整。landmark 数据增强：在 gt box 附近随机 crop 一个大小差不多的区域，然后计算 crop box 与 gt box 的 IoU，如果 > 0.65，那么这个 crop box 也作为一个正样本，按上式计算 landmark 坐标到 crop box 边缘的偏差相对值作为 target，然后 crop box 图像需要 resize 到 (12, 12)。然后分别按 0.5 的概率对 crop box 图像进行其他增强：镜像、旋转、逆时针旋转。

        - 镜像，将 crop image 左右翻转，target 值则变成 (1-target) 。
        - 旋转。将原图 im 绕 crop 图像的中心旋转 5°，然后 crop 区域不变，仍为原来的 crop box 坐标决定，这是因为旋转角度 5° 比较小，但是 landmarks 的 target 值按如下修改：

            ```python
            landmark_ = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                 rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmark])
            ```

            用数学式表达就是

            $$\mathbf x' = \mathbf M \mathbf x=\begin{bmatrix}m_{00} & m_{01} & m_{02} \\\\ m_{10} & m_{11} & m_{12} \\\\ 0 & 0 & 1\end{bmatrix} \begin{bmatrix}x \\\\ y \\\\ 1\end{bmatrix}$$

            landmark target 值是归一化的，需要先进行逆归一化，得到 landmark 点的齐次坐标 $[x, y, 1]^{\top}$，旋转之后坐标值就是左乘一个旋转矩阵，最后在计算 target 值。旋转之后 crop 图像依然要 resize 到 (12, 12)。
        
        landmarks 的训练样本全部为正样本，所以上面过程中得到的 landmarks 的 target 值 <= 0 或者 >=1 的全部过滤掉。
    
    - 将上面 PNet 的 pos，neg，part face 数据和 landmark 数据合并到一个文件中，其中 label 值为

        ```sh
        pos: 1      l,t,r,b 偏差相对值
        neg: 0
        part: -1    l,t,r,b 偏差相对值
        landmark: -2    地标点坐标与 l,t 的 偏差相对值（归一化）
        ```

2. R-Net。训练好 PNet 之后，使用 PNet 生成 RNet 的训练数据。
    
    使用 first stage 对 WIDER FACE 数据集进行检测，收集正负样本和 part face，对 CelebA 数据集进行检测，收集 landmark face。

    调用 `gen_hard_example.py` 文件中的 `t_net` 函数，

    ```python
    t_net(..., test_mode='PNet', ...)
    ```

    见下文 4.2.2 一节。

3. O-Net

    与 R-Net 类似，但是使用前两个 stage 对数据集检测从而收集 O-Net 的训练数据

# 4. 代码解读

由于官方代码使用 matlab 实现，这里我选择 [MTCNN-Tensorflow](https://github.com/AITTSMD/MTCNN-Tensorflow) 进行讲解。

## 4.1 训练数据

准备训练数据的步骤见项目的说明文档。[数据集 WIDERFACE](http://shuoyang1213.me/WIDERFACE/)。

对于负样本，gt label 为 `img_path 0`

对于正样本，gt label 为

```sh
img_path 1 offset_x1 offset_y1 offset_x2 offset_y2
```

对于 part face，gt label 为

```sh
img_path 1 offset_x1 offset_y1 offset_x2 offset_y2
```

准备 P-Net 分类任务和 bbox 回归任务的训练数据关键代码

```python
# prepare_data/gen_12net_data.py
for annotation in annotations:  # 遍历每一个图片的标注信息
    annotation = annotation.strip().split(' ')
    im_path = annotation[0] # 文件名
    bbox = list(map(float, annotation[1:])) # bbox 坐标
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4) # (n, 4)，n 是此图片中人脸数量
    img = cv2.imread(os.path.join(im_dir, im_path + '.jpg'))
    neg_num = 0
    while neg_num < 50:     # 每个图片上 crop 50 个负样本
        crop_box = np.array([nx, ny, nx+size, ny+size]) # 随机生成的 crop 坐标
        Iou = IoU(crop_box, boxes)

        cropped_im = img[ny:ny+size, nx:nx+size, :] # crop 的图片 patch
        resized_im = cv2.resize(cropped_im, (12, 12))

        if np.max(Iou) < 0.3:   # crop 了一个负样本
            f2.write('../../DATA/12/negative/%s.jpg'%n_idx + ' 0\n')    # 负样本的标注
            cv2.imwrite(save_file, resized_im)  # 保存负样本图片
            ...
    for box in boxes:   # 遍历每个 gt box
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        # gt box 附近再 crop 5 个 patch，如果 IOU < 0.3，那么保存为负样本
        for i in range(5):
            ...
            crop_box = np.array([nx1, ny1, nx1+size, ny1+size]) # 随机生成的 crop 坐标
            Iou = IoU(crop_box, boxes)
            cropped_im = img[ny1:ny1+size, nx1:nx1+size, :]
            resized_im = cv2.resize(cropped_im, (12, 12))
            if np.max(Iou) < 0.3:
                f2.write('../../DATA/12/negative/%s.jpg'%n_idx + ' 0\n')    # 负样本的标注
                cv2.imwrite(save_file, resized_im)  # 保存负样本图片
                ...
        # 生成正样本和 part face
        for i in range(20):
            ...
            nx2 = nx1 + size
            ny2 = ny1 + size
            crop_box = np.array([nx1, ny1, nx2, ny2])   # 随机，但是 crop 的中心位于 gt box 中心附近 delta_x 和 delta_y 距离处
            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)
            cropped_im = img[ny1:ny2, nx1:nx2, :] # crop 的图片 patch
            resized_im = cv2.resize(cropped_im, (12, 12))
            box_ = box.reshape(1, -1)
            iou = IoU(crop_box, box_)
            if iou >= 0.65:     # 正样本
                f1.write(...)
                cv2.imwrite(...)
            elif iou >= 0.4:    # part face
                f3.write(...)
                cv2.imwrite(...)
```

注意上面计算正样本和 part face 的 bbox target: `(x1 - nx1) / float(size)`，即，坐标偏差，然后除以边长。学习模型然后预估偏差与边长的比例，而非坐标偏差的绝对值。由于 P-Net 训练阶段的模型输入 size 为 `12x12`，所以还需要将 crop out 的图像 resize 到 `12x12` 。

准备 P-Net landmark 回归任务的训练数据关键代码

landmark 的 gt label 格式为

```sh
img_path -2 x1 y1 x2 y2 ... x5 y2
```

1. 先根据 gt bbox crop 出人脸 patch
2. 根据 gt bbox 重新计算 landmark 点坐标，并归一化
3. 在 patch 内根据 patch 中心随机微扰得到新的中心，以这个新中心 crop 出新 region，并计算相应的 landmark 点坐标


```python
# gen_landmark_aug_12.py
size = 12
# 遍历数据集中的：图像文件路径，gt box 坐标 x1y1x2y2，(5,2) 的 landmark 坐标
for (imgPath, bbox, landmarkGt) in data:
    F_imgs = []
    F_landmarks = []
    img = cv2.imread(imgPath)   # 加载图片数据
    # 此数据集中，一个图片只有一个 gt box，以及 5 个 landmark points
    f_face = img[bbox.top:bbox.bottom+1, bbox.left:bbox.right+1]
    f_face = cv2.resize(f_face, (size, size))
    landmark = np.zeros((5, 2))

    for index, one in enumerate(landmarkGt):    # 遍历 5 个 landmark 点坐标
        rv = ((one[0] - bbox.left)/(bbox.right-bbox.left),
              (one[1] - bbox.top) /(bbox.bottom-bbox.top))  # 归一化 landmark 坐标
        landmark[index] = rv
    F_imgs.append(f_face)
    F_landmarks.append(landmark.reshape(10))
    landmark = np.zeros((5, 2))
    if argument:    # True，实现数据增强
        x1, y1, x2, y2 = bbox
        gt_w = x2 - x1 + 1
        gt_h = y2 - y1 + 1

        for i in range(10): # 对应上面的第 3 点
            bbox_size = npr.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))    # 随机选择 size
            delta_x = npr.randint(-gt_w*0.2, gt_w*0.2)  # 微扰 offset
            delta_y = npr.randint(-gt_h*0.2, gt_h*0.2)
            # nx1 + bbox_size/2 是 crop 的中心 x 坐标，等于 patch 中心 + delta_x
            nx1 = int(max(x1+gt_w/2-bbox_size/2+delta_x, 0))
            ny1 = int(max(y1+gt_h/2-bbox_size/2+delta_y, 0))
            nx2 = nx1 + bbox_size
            ny2 = ny1 + bbox_size

            crop_box = np.array([nx1, ny1, nx2, ny2])
            cropped_im = img[ny1:ny2+1, nx1:nx2+1, :]
            resized_im = cv2.resize(cropped_im, (size, size))

            iou = IoU(crop_box, np.expand_dims(gt_box, 0))
            if iou > 0.65:
                F_imgs.append(cropped_im)
                # 基于 crop region 归一化 landmark 坐标
                ...
                # 数据增强
                if random.choice([0, 1]) > 0:   # mirror
                    # 将 resized_im flip
                    ...
                if random.choice([0, 1]) > 0:   # rotate
                    ...
```

这里有两个数据集，第一个收集了正负样本和 part face，第二个收集了 landmark face。

## 4.2 训练

## 4.2.1 P-Net

训练入口为 `train_PNet.py` 。

获取 batch 数据，

```python
# read_tfrecord_v2.py/read_single_tfrecord
...
label = tf.reshape(label, [batch_size]) # (B,) 0/1/-1/-2
roi = tf.reshape(roi, [batch_size, 4])  # (B, 4)
landmark = tf.reshape(landmark, [batch_size, 10])   # (B, 10)
```

如果某类型样本缺乏某个数据，那么对应的 target 值 为 0，例如正样本没有 landmark 点坐标，或者 landmark face 样本缺乏 bbox offset（即 `roi`） 数据。

P-Net 网络的结构比较简单，

```sh
                                               +----+
                                           +-->|conv|--> (pred_cls)
                                           |   +----+
          +----+   +--+   +----+   +----+  |   +----+
(input)-->|conv|-->|mp|-->|conv|-->|conv|--+-->|conv|--> (pred_box)
          +----+   +--+   +----+   +----+  |   +----+
                                           |   +----+
                                           +-->|conv|--> (pred_landmark)
                                               +----+
# 各 layer 的输出 size
(12x12)  (10x10)   (5x5)  (3x3)    (1x1)
```

计算 loss 的代码，如下

```python
# mtcnn_model.py/P_Net
cls_prob = tf.squeeze(conv4_1, [1, 2], name='cls_prob') # (B,1,1,2)->(B,2) 参见图 2
# label: 分类 gt ，(B,)
cls_loss = cls_ohem(cls_prob, label)
```

分类任务的在线难例挖掘代码，与上文 2.2 一节中关于 OHEM 的说明一致。需要注意的是，上文准备训练数据时，标注值为

```sh
# [path to image][cls_label][bbox_label][landmark_label]
  
pos：cls_label=1,bbox_label(calculate),landmark_label=[0,0,0,0,0,0,0,0,0,0].

part：cls_label=-1,bbox_label(calculate),landmark_label=[0,0,0,0,0,0,0,0,0,0].
  
landmark：cls_label=-2,bbox_label=[0,0,0,0],landmark_label(calculate).  
  
neg：cls_label=0,bbox_label=[0,0,0,0],landmark_label=[0,0,0,0,0,0,0,0,0,0].  
```

但是在训练 PNet 时，只用到 pos 和 neg，其余两个分类数据均不使用，见下方代码中的 `valid_inds` 。

```python
# mtcnn_model.py

num_keep_radio = 0.7 # 在线难例挖掘，仅保留损失 top 70% 大的样本，其他则不参与分类损失计算
def cls_ohem(cls_prob, label):
    '''
    cls_prob: (n, 1, 2)
    label: (n, 1)
    '''
    zeros = tf.zeros_like(label)
    # pos->1, neg->0, others(part, landmark)->0
    label_filter_invalid = tf.where(tf.less(label, 0), zeros, label)
    ...
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))  # 根据 gt 获取对应的预测得分，part 和 landmark 与 neg 一样均使用 p_i(0)，pos 使用 p_i(1)
    loss = -tf.log(label_prob + 1e-10)  # -\log p_i
    ...
    valid_inds = tf.where(label < zeros, zeros, ones)   # 只考虑 pos 和 neg
    num_valid = tf.reduce_sum(valid_inds)   # pos neg 的数量
    keep_num = tf.cast(num_valid * num_keep_radio, dtype=tf.int32)  # top 70%
    loss = loss * valid_inds    # 提取 pos 和 neg 的 loss
    loss, _ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_sum(loss)
```

bbox 回归任务的 OHEM 代码如下，实际上是使用了全部的 pos 和 neg，丢弃了 part face 和 landmark face。

```python
# mtcnn_model.py
def bbox_ohem(bbox_pred, bbox_target, label):
    # 只考虑 pos(1) 和 part(-1)
    valid_inds = tf.where(tf.equal(tf.abs(label), 1), ones_index, zeros_index)
    square_error = tf.square(bbox_pred - bbox_target)   # L2 范数, (B, 4)
    square_error = tf.reduce_sum(square_error, axis=1)  # (B,)
    num_valid = tf.reduce_sum(valid_inds)   # pos neg 的数量
    keep_num = tf.cast(num_valid, dtype=tf.int32)  # 保留全部 pos 和 neg
    square_error = square_error * valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)
```

landmark 回归任务的 OHEM 则类似的，使用了全部的 landmark face，丢弃了正负样本和 part face。

三种损失加权求和，然后再与模型权重的 L2 正则损失相加，得到最终的损失。


### 4.2.2 R-Net

训练数据准备

使用 P-Net 检测 Wider face 数据集，从而收集训练 R-Net 的数据。具体步骤：

1. 加载 Wider face 所有图片路径
2. 加载 P-Net 模型，为 Wider face 每个图片进行预测，根据预测值生成 box。注意这里是逐步缩小原始图片尺寸，然后喂给 P-Net（此即图片金字塔）。 

    ```python
    # mtcnn_model.py/P_Net
    # test 模式下，batch size 为 1
    cls_pro_test = tf.squeeze(conv4_1, axis=0)  # (h, w, 2) 
    bbox_pred_test = tf.squeeze(bbox_pred, axis=0)  # (h, w, 4)
    landmark_pred_test = tf.squeeze(landmark_pred, axis=0)  # (h, w, 10)
    ```

    注意 train 模式下，由于 network input size 为 crop 的 `12x12`，输出 size 为 `1x1`，但是 test 模式下，输入 size 是变化的，是将整个原始图片通过尺度变换生成图片金字塔，所以输入 size 大于等于 `12x12`，否则输出 size 为 0。但是 PNet 的整体下采样率为 2。
    
    图片金字塔作为输入的代码为，

    ```python
    # MtcnnDetector.py/detect_pnet 方法
    net_size = 12   # network 输入 size 为 12
    current_scale = float(net_size) / self.min_face_size    # 最小人脸尺寸，手动设置为 20
    im_resized = self.processed_image(im, current_scale)    # 将原始图片缩放 0.6
    current_height, current_width, _ = im_resized.shape
    all_boxes = list()

    while min(current_height, current_width) > net_size:    # 当前level的输入图片尺寸不小于 12
        # (h, w, 2), (h, w, 4)。分类得分中，分别表示为 neg 和 pos 的预测
        # 由于 test 阶段，输入 size 可能大于 12，所以 h w 可能大于 1。
        cls_cls_map, reg = self.pnet_detector.predict(im_resized)
        # 根据预测值生成 box（需要先对预测得分进行阈值筛选）
        # proposal box x1 y1 x2 y2，score，以及 dx1, dy1, dx2, dy2， (n, 9)，n 为预测为正例的数量
        boxes = self.generate_bbox(cls_cls_map[:,:,1], reg, current_scale, self.thresh[0])
        current_scale *= self.scale_factor  # 进一步缩小图片（从而得到图片金字塔）
        im_resized = self.processed_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape
        ...
        keep = py_nms(boxes[:, :5], 0.5, 'Union')   # NMS 筛选
        boxes = boxes[keep]
        all_boxes.append(boxes)
    # 所有金字塔结构的图片的预测结果再进行 NMS 筛选
    keep = py_nms(all_boxes[:, :5], 0.5, 'Union')
    all_boxes = all_boxes[keep]
    boxes = all_boxes[:, :5]
    bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1 # proposal box width
    bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1 # proposal box height

    # refine the boxes, x1' = x1 + dx1 * w
    boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                            all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                            all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                            all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                            all_boxes[:, 4]])
    boxes_c = boxes_c.T
    # (N, 9), (N, 5), 无landmark 预测故为 None
    # N 表示各个缩放 level 对应的检测出正样本 ni 之和
    return boxes, boxes_c, None
    ```

    每个图片经过上述过程，得到 `boxes_c` 的列表，`boxes_c` 包含了某个图片经过 PNet 预测的（经过预测得分大于阈值的筛选之后的）所有人脸的预测框 x1y1x2y2 坐标和预测得分。这些预测框将作为 RNet 的训练样本，但是我们还需要处理一下，哪些是正样本哪些是负样本。

3. 经过 P-Net 预测出来的 box 即，对预测得分进行阈值筛选之后的预测框，再作为 R-Net 的训练样本，经过 R-Net 的预测，从而达到 refine 的目的。

    P-Net 预测出来的 box，再根据与 gt box 的 IoU 判断出是正样本还是负样本，还是 part face，判断出来之后保存到文件，作为 R-Net 的训练数据。代码见 `gen_hard_example.py/save_hard_example` 方法。

    ```python
    # 图像文件路径、PNet 预测、gt box 坐标
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):
        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)    # (n, 4) gt box: x1y1x2y2
        img = cv2.imread(im_idx)
        dets = convert_to_square(dets)  # 将预测框的短边增大到与长边一样，从而变成方形预测框
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1    # 预测框的 width
            height = y_bottom - y_top + 1   # 预测框的 height

            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue    # 预测框 size 至少为 20，且坐标位于图像内（即，完全的人脸展示出来）
            Iou = IoU(box, gts) # 当前预测框与所有 gt boxes 的 IoU
            cropped_im = img[y_top:y_bottom+1, x_left:x_right+1, :]
            resized_im = cv2.resize(cropped_im, (img_size, img_size))   # resize 到 (24, 24)
            if np.max(Iou) < 0.3 and neg_num < 60:  # 保存为负样本
                ...
            else:
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt
                # 计算坐标偏差的相对值
                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                if np.max(Iou) >= 0.65: # 保存为正样本
                    pos_file.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                elif np.max(Iou) >= 0.4:# 保存为 part face
                    part_file.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
    ```
    以上，就将 RNet 的训练数据的 pos，neg，part 三种样本均准备好。

4. 使用 `gen_landmark_aug_24.py` 生成 R-Net 的 landmark 训练数据。逻辑与生成 PNet 的 landmark 数据完全一致，只是 crop 图像需要 resize 到 (24, 24) 。

5. 最后所有的数据一起，作为 R-Net 的训练数据，这一步使用 `gen_imglist_rnet.py` 完成。训练 RNet 与训练 P-Net 基本一致。加载数据集时，使用 `read_multi_tfrecords`，这么做是为了将 pos neg part 和 landmark 四种样本分别以不同的 batch size 加载，即加载的一个 batch 中，其中 pos part 和 landmark 各占 1/6，neg 样本占 3/6。

**RNet 网络**

RNet 网络输入 size 为 (24, 24)，经过中间卷积层后输出特征 size 为 (3, 3)，然后 flatten 后再经过一个 fc 层，得到特征维度为 (128, )，然后分别经过 3 个 fc 层，调整输出维度为 2, 4, 10，分别表示 label，box，landmark 预测，计算这三种损失，与训练 PNet 的一样，参考上面的分析。

### 4.2.3 O-Net

生成 O-Net 训练数据的过程与 R-Net 的类似，代码位于 `gen_hard_example.py`，下面我们列出其中的几个关键变量的值，

```python
# gen_hard_example.py
net = 'RNet'    # 生成 R-Net 训练数据时设置
net = 'ONet'    # 生成 O-Net 训练数据时设置
test_mode = 'PNet'  # 生成 R-Net 训练数据时设置
test_mode = 'RNet'  # 生成 O-Net 训练数据时设置
slide_window = False    # 设置为 False 就好（for R-Net and O-Net）
```

上一节讲到生成 R-Net 的训练数据时，使用 P-Net 对图片进行预测，那么生成 O-Net 训练数据时，除了第一步使用 P-Net 对图片进行预测之外，还需要第二步，使用 R-Net 对 P-Net 的预测再次进行预测，

```python
# MtcnnDetector.py/detect_face
if self.pnet_detector:
    _, boxes_c, landmark = self.detect_pnet(im)

if self.rnet_detector:
    _, boxes_c, landmark = self.detect_rnet(im, boxes_c)
```

其中 `detect_rnet` 方法所作的事情为：

1. 根据 P-Net 的预测 box，对原始图片 `im` 进行 crop，并调整 crop 后的图片 size 为 `24x24`

    - 将将 P-Net 的预测 box 调成方形（固定住中心，短边增大到长边），记预测 box size 为 (crop_h, crop_w)。然后根据预测 box 从源图像中 crop 放到目标图像，源图像 size 为 (im_h, im_w)，目标图像 size 为 (crop_h, crop_w) ，如果预测 box 完全包含在源图中，那么直接 crop 即可，否则仅 crop 包含在源图中的那一部分，如下图。

    ![](/images/face/mtcnn_3.jpg)

    - crop 之后再 resize 到 (24, 24)，然后归一化到 (-1, 1) 之间。stack 所有这样的 crop 归一化图像，其 shape 为 (n, 24, 24, 3)，其中 n 为此源图中 PNet 预测数量。

2. 使用 R-Net 再进行预测，根据一个预测得分阈值进行筛选，然后使用 NMS 筛选，然后使用 R-Net 的预测 bbox offset 对 P-Net 的预测 box 进行校正

    R-Net 预测包含 (n, 2) 的分类预测、(n, 4) 的 box 坐标偏差相对值预测以及 landmark 坐标偏差相对值预测。

    - 根据分类为 1 的得分，进行阈值筛选，然后再进行 NMS。

        ```python
        cls_scores, reg, _ = self.rnet_detector.predict(cropped_ims)
        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > self.thresh[1])[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds] # dets 是 PNet 预测结果
            boxes[:, 4] = cls_scores[keep_inds] # 使用 RNet 预测得分更新 PNet 的预测得分
            reg = reg[keep_inds]
        keep = py_nms(boxes, 0.6)   # 使用 RNet 的预测得分，重新做 NMS
        boxes = boxes[keep]
        boxes_c = self.calibrate_box(boxes, reg[keep])  # 使用 RNet 预测的坐标偏差相对值更新预测 box
        ```
    - 将 `boxes_c` 值保存起来。要得到训练 O-Net 的标注数据，还需要使用 `save_hard_example` 将 RNet 预测结果与 gt boxes 对比，区分出 pos，neg 和 part face 的样本。

3. O-Net 除了网络结构上与 R-Net 有一些不同之外，训练过程是一样的，O-Net 输出分类得分、box 坐标偏差相对值和 landmark 坐标偏差相对值三个预测。

## 4.3 检测人脸

```python
# one_image_test.py

all_boxes, landmarks = mtcnn_detector.detect_face(test_data)
```

与上一节生成 O-Net 的训练数据类似，在此基础上，即 R-Net 对 P-Net 的预测进行 refine 之后，再经过 O-Net 预测，得到最终的预测 box，`detect_face` 方法且预测人脸 box，没有给出 landmark 预测值，要返回 landmark 预测值，使用 `detect_single_image` 方法。整个预测过程是：

1. 不同 scale level 的金字塔图像，分别经过 PNet 的预测，阈值筛选，NMS，然后所有 level 的预测结果合并之后再 NMS

    PNet 是 FCN 全卷积网络，所以输入 size 不需要与训练阶段的 (12, 12) 相同。

2. PNet 预测结果经过 RNet 预测，对 PNet 的预测结果重新进行 预测得分更新，阈值筛选，NMS

    RNet 不是 FCN，所以根据 PNet 预测结果在源图像上 crop 之后，还要 resize 到 (24, 24)，才能喂给 RNet 。

3. RNet 预测结果再经过 ONet 预测，对 PNet 的预测结果重新进行 预测得分更新，阈值筛选，NMS，过程与 RNet 类似

    ONet 的预测结果中保留了 landmark 的预测，而 PNet 和 RNet 的 landmark 预测都丢弃了。landmark 坐标预测为

    $$r = (x - x_1) / w \Rightarrow x = r * w + x_1$$

    其中 $r$ 是 landmark 坐标偏差相对值，$w$ 是人脸 gt box 宽，$x_1$ 是 gt box 的左侧坐标，$x$ 是地标点的横坐标。

    ```python
    # width
    w = boxes[:, 2] - boxes[:, 0] + 1   # boxes 是 RNet 的预测 box 坐标
    # height
    h = boxes[:, 3] - boxes[:, 1] + 1
    landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
    landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
    ```