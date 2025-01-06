---
title: 相机模型（二）
date: 2024-05-06 14:35:27
tags: camera calibration
categories: 相机标定
mathjax: true
---

# 2. 投影相机

一般投影相机将世界点 $\mathbf X$ 变为图像点 $\mathbf x = P \mathbf X$ 。本文解剖相机模型。

## 2.1 相机解剖

相机模型可表示为 $P=[M|\mathbf p _ 4] \in \mathbb R ^ {3 \times 4}$，如果 $M$ 非奇异，那么这是一个有限相机。

**相机中心**

矩阵 $P \in \mathbb R ^ {3 \times 4}$ 的秩为 3，故其核空间维度为 1，记核空间的一个非零向量为 $\mathbf C \in \mathbb R ^ {4 \times 1}$（即，基向量），那么 $P \mathbf C =\mathbf 0$。以下我们将会说明 $\mathbf C$ 为相机中心，这是一个齐次 4 维坐标向量。

考虑包含 $\mathbf C$ 和 3D 空间中其他任意点 $\mathbf A$ 的直线，那么直线上任意一点可表示为

$$\mathbf X(\lambda)=\lambda \mathbf A + (1-\lambda)\mathbf C \tag{1}$$

根据 $\mathbf x = P \mathbf X$ 以及 $P \mathbf C=\mathbf 0$，得

$$\mathbf x = P\mathbf X(\lambda)=\lambda P \mathbf A + (1-\lambda)P \mathbf C = \lambda P \mathbf A \tag{2}$$

故直线上所有点均映射到 $P\mathbf A$，这里 $\lambda$ 值不影响，因为要将 $\mathbf x$ 的 z 坐标归一化，$\lambda$ 就被归约掉了，所以直线 $\mathbf {CA}$ 经过相机中心，而 $\mathbf A$ 可以任意选择，所以 $\mathbf C$ 是相机中心。

当然，$\mathbf C$ 要作为相机中心的齐次坐标表示，那么第 4 个元素必须为 1，于是 $\mathbf C$ 就固定住且唯一，其他 $\mathbf C'=k\mathbf C$ 虽然满足 $P\mathbf C'=\mathbf 0$，但是不满足第 4 个元素为 1。

$\mathbf C$ 是相机中心并不意外，因为图像点 $(0,0,0)^{\top}=P\mathbf C$ 未定义（因为图像点的齐次坐标中 z 轴需要为 1，所以 $(0,0,0)^{\top}$ 是未定义，也可以说是图像中无穷远处），而相机中心是空间中唯一的成像点未定义。

如果 $P$ 的左侧 $3 \times 3$ 子矩阵 $M$ 是奇异的，那么 $\mathbf C=(\mathbf d ^{\top}, 0) ^{\top}$，其中 $M \mathbf d=\mathbf 0$，此时相机中心在无穷远处。

> 齐次坐标向量中第 4 维值为 1，如果为 0，表示在无穷远处，下文 X Y Z 轴方向的坐标向量第 4 维值也是 0，表示坐标轴的尽头

**列向量**

射影相机矩阵的列向量是 3D 向量 $\mathbf p _ i, i=1,\ldots,4$，那么 $\mathbf p _ 1, \mathbf p _ 2, \mathbf p _ 3$，分别对应世界坐标系 X Y Z 轴的消影点（vanishing point），例如 X 轴方向 $\mathbf D=(1,0,0,0) ^ {\top}$，而 $\mathbf p _ 1 = P \mathbf D$。

$\mathbf p _ 4$ 是世界坐标系原点 $(0,0,0,1)^{\top}$ 的成像点 。

**行向量**

射影相机矩阵的行向量是 4D 向量 $P ^ {1\top}, P ^ {2\top}, P ^ {3\top}$，行向量在几何上解释为特殊的世界平面，如图 1，

![](/images/dip/camera_model_4.jpg)

<center>图 1</center>

**主平面**

主平面经过相机中心且平行与图像平面。主平面内点 $\mathbf X$ 满足 $P\mathbf X=(x,y,0)^{\top}$，在图像坐标系中第 3 维元素值必须为 0，表示在无穷远处，因为相机中心与 $\mathbf X$ 的连线平行于图像平面，所以认为连线与图像平面的交点在无穷远。于是 $P ^ {3\top}\mathbf X=0$，也就是说 $P ^ 3$ （$P$ 的第 3 行写成列向量形式）表示主平面的向量（平面法向量，由于是齐次表示，实际上法向量是 $P ^ 3$ 的前 3 维子向量）。

**轴平面**

考虑平面 $P ^ 1$ 上的点 $\mathbf X$，满足 $P ^ {1\top}\mathbf X=0$，那么成像坐标为 $P\mathbf X=(0,y,w)^{\top}$，这表示图像坐标系的 y 轴，并且根据 $P\mathbf C=\mathbf 0$ 可知 $P ^ {1\top}\mathbf C=0$，于是相机中心 $\mathbf C$ 也在平面 $P ^ 1$ 上，因此平面 $P ^ 1$ 由相机中心和图像中 $x=0$ 的线定义。也就是说，平面 $P ^ 1$ 的点与相机中心的连线与图像平面相交在 y 轴上。类似地 $P ^ 2$ 由相机中心和 $y=0$ 定义。

与主平面不同，轴平面依赖于图像的 x y 轴，也就是说，与图像坐标系的选择有关。平面 $P ^ 1$ 和 $P ^ 2$ 的交线是相机中心和图像坐标原点的连线。

**主点**

主轴经过相机中心并垂直主平面 $P ^ 3$，主轴与图像平面交于主点。

平面 $\boldsymbol {\pi}=(\pi _ 1,\pi _ 2,\pi _ 3,\pi _ 4)^{\top}$ 的法向量是 $(\pi _ 1,\pi _ 2,\pi _ 3) ^{\top}$，也可以使用平面上无穷远处的点 $(\pi _ 1, \pi _ 2, \pi _ 3, 0)^{\top}$ 表示（把齐次坐标的最后一维的值改成 0 就表示无穷远）。对于主平面 $P ^ 3$，也可以使用无穷远点 $\hat {\mathbf P} ^ 3 = (p_{31}, p_{32}, p_{33}, 0) ^ {\top}$ 表示，此点经过相机矩阵 $P$ 转换为 $P \hat {\mathbf P} ^ 3$ 就是主点，这里 $P=[M|\mathbf p_4]$，所以主点为 $\mathbf x _ 0=M \mathbf m ^ 3$，其中 $\mathbf m ^ 3$ 是 $M$ 的第三行的转置。

## 2.2 射影相机作用到点上

**点反向投影为射线**

给定图像上一点 $\mathbf x$，如何确定是空间中的哪些点的投影。这些点构成一个射线，且经过相机中心，这些点全部投影到图像上的点 $\mathbf x$。

我们知道两个点，一个是相机原点 $\mathbf C$，还有一个是 $P ^ + \mathbf x$，其中 $P ^ +$ 是 $P$ 的伪逆 $P ^ + = P ^ {\top} (P P ^ {\top}) ^ {\top}$，这样 $P P ^ + = I$。点 $P ^ + \mathbf x$ 位于射线上，因为 $P(P ^ + \mathbf x)=I \mathbf x=\mathbf x$，那么射线可以用两点表示

$$\mathbf X(\lambda)=P ^ + \mathbf x + \lambda \mathbf C$$

如果是有限相机，$P=[M|\mathbf p_4]$，其中 $M$ 是非奇异矩阵，那么相机中心为 $\tilde {\mathbf C}=-M ^ {\top} \mathbf p _ 4$，然后图像上点 $\mathbf x$ 反向投影的射线与无穷远平面交点为 $\mathbf D = ((M ^ {-1}\mathbf x) ^ {\top}, 0)^{\top}$（无穷远那么最后一维元素值为 0，然后验证 $P\mathbf D=M(M ^ {-1}\mathbf x) + 0 \cdot \mathbf p _ 4=\mathbf x$），那么根据相机中心和 $\mathbf D$ 点得到射线为

$$\mathbf X(\mu)=\mu \begin{pmatrix}M ^ {-1} \mathbf x \\ 0\end{pmatrix}+\begin{pmatrix}-M ^ {-1} \mathbf p _ 4 \\ 1\end{pmatrix}=\begin{pmatrix}M^{-1}(\mu \mathbf x - \mathbf p _ 4) \\ 1\end{pmatrix}$$

## 2.3 点的深度

我们考虑点在相机主平面前面或者后面的距离。相机矩阵 $P=[M|\mathbf p _ 4]$，将点 $\mathbf X=(X,Y,Z,1)^{\top}=(\tilde {\mathbf X} ^ {\top}, 1) ^ {\top}$ 投影到图像上 $\mathbf x = w(x,y,1) ^ {\top}= P\mathbf X$，记相机中心为 $\mathbf C = (\tilde {\mathbf C}, 1)^{\top}$，根据 $P\mathbf C = \mathbf 0$，得到

$$w=P ^ {3\top} \mathbf X=P ^ {3\top} (\mathbf X - \mathbf C)$$

由于 $\mathbf X - \mathbf C$ 的最后一维元素值为 0，那么上式变成

$$w=\mathbf m ^ {3 \top}(\tilde {\mathbf X} -\tilde {\mathbf C})$$

其中 $\mathbf m ^ {3 \top}$ 是主轴方向。

如果相机矩阵已经归一化，那么 $\det M > 0$ 且 $||\mathbf m ^ 3||=1$，那么 $\mathbf m ^ 3$ 是一个单位向量且指向正轴方向，那么 $w$ 就表示点 $\mathbf X$从相机中心 $\mathbf C$ 沿着主轴方向的深度，如图 2 所示，

![](/images/dip/camera_model_5.jpg)

<center>图 2</center>

虽然相机矩阵总是可以归一化，但是我们可以通过以下方法计算非归一化相机矩阵的点深度。

记 $\mathbf X=(X,Y,Z,T)^{\top}$ 为 3D 点，有限相机矩阵为 $P=[M|\mathbf p_4]$，假设 $P(X,Y,Z,T)^{\top}=w(x,y,1)^{\top}$，那么

$$\text{depth}(\mathbf X;P)=\frac {\text{sign}(\det M)w}{T||\mathbf m ^ 3||}$$

表示点 $\mathbf X$ 在相机主平面前面的深度。

上式可以有效地判断点 $\mathbf X$ 是否位于相机前面。可以验证如果 $\mathbf X$ 或者 $P$ 乘上一个因子 $k$，深度 $\text{depth}(\mathbf X;P)$ 也保持不变，因此深度与 $\mathbf X$ 和 $P$ 的具体齐次表征无关。

## 2.4 相机矩阵的分解

一般射影相机矩阵 $P$，我们希望根据 $P$ 获得相机中心，相机朝向以及相机内参。

**相机中心**

根据 $P\mathbf C=\mathbf 0$ 求相机中心 $\mathbf C$，即 $\mathbf C$ 是 $P$ 特征值为 0 对应的特征向量。 对 $P$ 做 SVD，那么最小特征值对应的右特征向量就是 $\mathbf C$ 。

根据代数计算，可得 $\mathbf C=(X,Y,Z,T)^{\top}$ 为

$$X=\det([\mathbf p _ 2, \mathbf p_ 3, \mathbf p _ 4]) \quad Y=\det([\mathbf p _ 1, \mathbf p_ 3, \mathbf p _ 4])
\\ Z=\det([\mathbf p _ 1, \mathbf p_ 2, \mathbf p _ 4]) \quad T=\det([\mathbf p _ 1, \mathbf p_ 2, \mathbf p _ 3])$$

**相机朝向和内参**

由于

$$P=[M|-M\tilde {\mathbf C}]=K[R|-R\tilde {\mathbf C}]$$

容易看出将 $M$ 进行 QR 分解为 $M=KR$，那么相机朝向为 $R$，内参为 $K$ 。QR 分解中我们要求 $K$ 的对角线元素为正。

> 正交分解，表示将矩阵分解为一个正交矩阵 Q 和一个上三角矩阵 R 的乘积

$K$ 的形式为

$$K = \begin{bmatrix}\alpha _ x &s & x _ 0\\ & \alpha _ y & y _ 0 \\ & & 1 \end{bmatrix}$$

- $\alpha _ x$ 为 x 轴的尺度因子
- $\alpha _ y$ 为 y 轴的尺度因子
- $s$ 为偏度
- $(x _ 0, y _ 0) ^ {\top}$ 表示主点坐标（在像素坐标系下）

**当 $s\ne 0$ 时：**

如果 $s \ne 0$，那么 x 轴和 y 轴不垂直，这通常不太可能发生。

如果对着一个图像进行拍照，那么可能出现 $s \ne 0$ 。

