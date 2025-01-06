---
title: Character Region Awareness for Text Detection 论文解读
date: 2024-04-08 08:56:55
tags: 
    - ocr
    - text detection
---

论文：[Character Region Awareness for Text Detection](https://arxiv.org/abs/1904.01941)

# 1. 简介

本文提出 CRAFT (Character Region Awareness For Text detection)，用于检测文本，定位每一个字符，并将检测的字符联系起来成为一个文本实例。

CRAFT 生成字符区域 score 和关系 score，前者用于定位单独的字符，后者将独立的字符进行分组得到文本实例。通过定位单独的字符，可以检测任意形状的文本实例。

# 2. 方法

由于没有公开的基于字符标注的数据集，本文模型使用弱监督方法。

## 3.1 框架

基于 VGG-16，采用了 batch normalization，在解码部分，使用了 skip connection，类似于 U-net，输出包含两个 score maps：region score 和 affinity score，整个框架如图 1，

![](/images/ocr/craft_1.png)
<center>图 1 </center>

## 3.2 训练

### 3.2.1 gt label 生成

我们为每个训练图片生成 region score 和 affinity score 的 gt label。region score 表示像素是字符中心的概率，affinity score 表示是两个相邻字符中间空白的概率。

在二值分割 map 中，每个像素的 label 是独立的，但是在本文方法中，字符中心概率使用高斯 heatmap 表示，如图 2，

![](/images/ocr/craft_2.png)
<center>图 2. gt label 生成示意图</center>

图 2 中，根据字符标注，得到绿色的字符框，然后两条对角线将字符框分为 4 个三角形，上下两个三角形的中心为蓝色 +，然后两个相邻字符的 4 个蓝色 + 构成 affinity box 的顶点，这样就得到 region box 和 affinity box 的标注框。将标准 2D 高斯分布根据标注框进行透视变换，就得到 region score 和 affinity box 的 gt map 。

本文方法着重学习字符内部和字符之间（affinity）的信息，而不是整个文本实例。

### 3.2.2 弱监督学习

公开数据集通常都是单词级别的标注，故本文使用弱监督生成字符框，如图 3，

![](/images/ocr/craft_3.png)
<center>图 3</center>

对一个单词级别标注的图片，一个临时学习好的模型预测字符 region score 以得到字符级别的标注框，即，得到字符伪 gt label 。

将图片的每个单词 crop 出来作为输入，得到 region score map，根据切分得到预测的字符，为了反映这个临时模型的可靠性，计算每个 word box 的置信度，其值为 所检测出的字符数量除以 gt 字符数量，置信度值作为训练中学习的权重。

图 4 是切分字符的整个过程，

![](/images/ocr/craft_4.png)
<center>图 4. 字符切分过程。1. crop 出 word；2. 预测 region score；3. 使用 watershed algorithm；4. 获取字符框；5.unwarp 字符框</center>

切分得到字符的步骤：

1. 将原图中的单词部分 crop 出来
2. crop 出来的单词图片喂给最新训练出来的模型，得到 region score
3. 使用分水岭算法 watershed 将字符区域切分出来。watershed 算法用于获取字符bbox
4. 将字符 box 的坐标转换为原图中
5. 根据图 2 得到 region score 和 affinity score 的伪 gt label。

当模型使用弱监督训练时，我们不得不使用伪 gt 来训练，如果模型用了不准确的 region score 来训练，那么输出在字符 region 内是模糊的，为了阻止这个现象，本文作者计算了由模型生成的伪 gt 的质量，也就是置信度，因为单词级别的标注中，单词长度是一个很强的信息。

假设数据集中一个单词级别的标注样本 $w$，其 word bbox 为 $R(w)$，word 长度为 $l(w)$，通过字符切分过程，我们得到预测的字符 bbox 和字符数量 $l ^ c(w)$，那么置信度为

$$s _ {conf} (w) = \frac {l(w) - \min(l(w), |l(w) - l ^ c (w)|)} {l(w)} \tag{1}$$

(1) 式分子的 $\min$ 是避免分子为负。那么一个图像的像素置信度 map（见图 3 中的 Confidence map）为

$$S _ c (p) = \begin{cases} s _ {conf} (w) & p \in R(w) \\ 1 & \text{o.w.} \end{cases} \tag{2}$$

置信度作为学习的权重，那么目标函数为

$$L=\sum _ p S _ c (p) \cdot (||S _ r(p) - S _ r ^ *(p)|| _ 2 ^ 2 + ||S _ a (p) - S _ a ^ * (p)|| _ 2 ^ 2) \tag{3}$$

其中 $S _ r ^ * (p)$ 和 $S _ a ^ * (p)$ 分别为 region score 和 affinity score 的伪 gt 。当使用合成数据训练时，由于知道真实的 gt，故 $S _ c(p)=1$ 。

随着训练的进行，CRAFT 模型预测字符越发准确，置信度得分 $s _ {conf}(w)$ 逐渐增大，如图 5 所示，

![](/images/ocr/craft_5.png)
<center>图 5</center>

如果置信度低于 0.5，那么预测的字符 bbox 应该被忽略，因为此时伪 gt 已经错误较大，模型学习拟合一个较大错误的伪 gt 肯定不合适，故对于置信度低于 0.5 的情况，我们假定每个字符的宽度是相同的，故直接根据字符数量 $l(w)$ 对 word region $R(w)$ 进行切分，得到字符预测，然后置信度设置为 0.5 。

## 3.3 推断

推断阶段，最终的输出形式是多种多样的，例如 word box 或者字符 box，以及多边形。数据集 ICDAR 的评估方式是 word 级别的 IoU，故本文作者给出了如何根据 $S _ r$ 和 $S _ a$ 得到 word 级别的 box ，这是一个四边形 QuadBox。

首先，与 image 相同 size 的二值 map $M$ 初始化为 0，如果 $S _ r(p) > \tau _ r$ 或者 $S _ a(p) > \tau _ a$，那么设置 $M(p) = 1$，其中 $\tau _ r, \ \tau _ a$ 分别是 region 阈值和 affinity 阈值。

其次，对 $M$ 执行 Connected Component Labeling(CCL)

最后，通过寻找最小面积的矩形得到 QuadBox，这个矩形要包含 connected components，可以使用 opencv 的 `connectedComponents` 和 `minAreaRect` 两个函数实现。

