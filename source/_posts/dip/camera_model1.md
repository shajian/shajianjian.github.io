---
title: 相机模型（一）
date: 2024-05-06 14:35:27
tags: camera calibration
categories: 相机标定
mathjax: true
---

相机将 3D 现实世界中的目标射影到 2D 图像平面。本文主要考虑中心射影。根据相机中心在有限远处和无限远处，分为有限相机和无限相机。

# 1. 有限远相机

## 1.1 针孔相机模型

如图 1 所示，

![](/images/dip/camera_model_1.jpg)
<center>图 1</center>

射影中心作为欧氏空间坐标系原点，图像（成像）平面位于焦距处，空间中一点 $\mathbf X=(X,Y,Z)^{\top}$ 在图像中对应为 $\mathbf x$，即直线 $C\mathbf X$ 与图像面的交点，根据相似三角形性质，可知

$$(X,Y,Z)^{\top} \rightarrow (f\cdot X/Z, f\cdot Y/Z) ^ {\top} \tag{1}$$

我们省去了射影点的 Z 坐标 $f$，因为图像中的坐标是 2D 。

**概念：**

射影中心称作相机中心，也叫做光学中心。

相机中心到图像平面的垂线称作相机的主轴或主线。

主轴与图像平面的交点称为主点。

经过相机中心且与图像平面平行的平面称作主平面。

## 1.2 齐次坐标的中心射影

使用齐次坐标表示世界点和图像点，那么中心射影可使用一个线性映射表示，

$$\begin{bmatrix}X \\ Y \\ Z \\ 1\end{bmatrix} \rightarrow \begin{bmatrix} fX \\ fY \\ Z \end{bmatrix}=\begin{bmatrix}f & & & 0\\ & f & & 0\\& & 1 & 0\end{bmatrix}\begin{bmatrix} X \\ Y \\ Z \\ 1\end{bmatrix} \tag{2}$$

上式其实少了一个因子 $1/Z$ 。

(2) 式可以写为

$$\mathbf x = P \mathbf X \tag{3}$$

其中 $P \in \mathbb R ^ {3 \times 4}$ 表示齐次相机射影矩阵，

$$P = \text{diag}(f, f, 1)[I|\mathbf 0] \tag{4}$$

其中 $P \in \mathbb R ^ {3 \times 4}$，(4) 式的写法表示两个矩阵相乘。

(1) 式中，图像坐标系原点位于主点 p 处，如果图像坐标系原点不在 p 处，那么需要增加一个位移，

$$(X, Y, Z) ^ {\top} \rightarrow (fX/Z + p_x, fY+Z+p_y) \tag{5}$$

其中 $(p _ x, p _ y)$ 是主点坐标，如图 2 所示，主点从左下角移动到 p 处。

![](/images/dip/camera_model_2.jpg)
<center>图 2</center>

此时齐次相机射影矩阵为

$$\begin{bmatrix}X \\ Y \\ Z \\ 1\end{bmatrix} \rightarrow \begin{bmatrix} fX+Zp _ x \\ fY+Z p _ y \\ Z \end{bmatrix}=\begin{bmatrix}f & & p _ x & 0\\ & f & p _ y & 0\\& & 1 & 0\end{bmatrix}\begin{bmatrix} X \\ Y \\ Z \\ 1\end{bmatrix} \tag{6}$$

射影矩阵写为

$$K = \begin{bmatrix}f & & p _ x \\ & f & p _ y \\& & 1 \end{bmatrix} \tag{7}$$

$K$ 称为相机标定矩阵。

射影变换形式为

$$\mathbf x = K[I|\mathbf 0] \mathbf X _ {cam} \tag{8}$$

其中 $\mathbf X _ {cam}$ 是相机坐标系中点的坐标，故 $K$ 表示从相机坐标系到图像坐标系的变换矩阵。

## 1.3 相机旋转和平移

空间中的一点可以使用不同的欧氏坐标系表达，即世界坐标系，世界坐标系与相机坐标系之间通过旋转和平移联系起来，如图 3，

![](/images/dip/camera_model_3.jpg)
<center>图 3</center>

记世界坐标系中一点的非齐次坐标（3元素向量表示）为 $\tilde {\mathbf X}$，对应地在相机坐标系中坐标为 $\tilde {\mathbf X} _ {cam}$，那么有关系

$$\tilde {\mathbf X} _ {cam} = R (\tilde {\mathbf X} - \tilde {\mathbf C}) \tag{9}$$

其中 $\tilde {\mathbf C} \in \mathbb R ^ {3\times 1}$ 表示相机坐标系原点在世界坐标系中的坐标，$R \in \mathbf R ^ {3 \times 3}$ 为旋转矩阵，表示相机坐标系的朝向，于是变换过程为

$$\mathbf X _ {cam} = \begin{bmatrix}R & -R \tilde {\mathbf C} \\ \mathbf 0 & 1 \end{bmatrix}\begin{bmatrix} X \\ Y \\ Z \\ 1\end{bmatrix}=\begin{bmatrix}R & - R \tilde {\mathbf C} \\ \mathbf 0 & 1\end{bmatrix} \mathbf X \tag{10}$$

与 (8) 式一起可得，

$$\mathbf x = KR[I |-\tilde {\mathbf C}]\mathbf X \tag{11}$$

(11) 式就是针孔相机的一般映射矩阵，$P = KR[I |-\tilde {\mathbf C}]$，自由度为 9，其中 $K$ 自由度为 3，包括 $f, p _ x, p _ y$，$R$ 自由度为 3，包括 X Y Z 三个方向的旋转角度，$\tilde {\mathbf C}$ 自由度为 3，包含 X Y Z 三个方向平移量。

$K$ 称为相机内参。$R$ 和 $\tilde {\mathbf C}$ 称为相机外参，表示相机在世界坐标系中的朝向和位置。

不明确表示相机中心会给我们带来方便，世界到相机的转换表示为 $\tilde {\mathbf X} _ {cam} = R \tilde {\mathbf X} + \mathbf t$，其中 $\mathbf t = -R \tilde {\mathbf C}$，那么世界到图像的变换矩阵为

$$P=K[R|\mathbf t] \tag{12}$$

## 1.4 CCD 相机

针孔相机模型中，图像坐标系为欧氏坐标系，其中两个轴向具有相同的尺度。CCD 相机中，每个方向的尺度不等，x 和 y 方向的单位距离分别有 $m_x$ 和 $m _ y$ 个像素，那么变换矩阵需要左乘一个矩阵 $\text{diag}(m _ x, m _ y, 1)$，相当于最后从图像坐标系再变换到像素坐标系，于是 CCD 相机的标定矩阵为

$$K=\begin{bmatrix}\alpha _ x & & x _ 0\\ & \alpha _ y & y _ 0 \\ & & 1 \end{bmatrix} \tag{13}$$

其中 $\alpha _ x = f m _ x, \ \alpha _ y = f m _ y$ 表示相机在像素维度的焦距，类似地，$\tilde {\mathbf x} _ 0=(x _ 0, y _ 0)$ 表示的是主点在像素维度上的坐标，即 $x _ 0 = m _ x p _ x, \ y _ 0 = m _ y p _ y$ 。

## 1.5 有限射影相机

一般地，增加一个参数 $s$ 表示 skew (偏度)参数，那么标定矩阵变为

$$K = \begin{bmatrix}\alpha _ x &s & x _ 0\\ & \alpha _ y & y _ 0 \\ & & 1 \end{bmatrix} \tag{14}$$

对于大多数相机，skew 参数为 0，某些特殊情况下，skew 非 0。

增加 skew 参数后，射影矩阵 $P=KR[I|-\tilde {\mathbf C}]$ 对应的相机称为有限射影相机。自由度为 11，因为 $K$ 的自由度从 3 变成 5，所以相机射影矩阵的自由度从 9 变成 11。

注意到 $P$ 的左侧子矩阵 $KR$ 是非奇异的。反过来，任意 $3 \times 4$ 矩阵其中左侧 $3 \times 3$ 子矩阵非奇异，那么这个矩阵就是某个有限射影相机的矩阵。

## 1.6 一般射影相机

最后，我们移除左侧 $3 \times 3$ 子矩阵的非奇异限制。一般射影相机由任意秩为 3 的齐次 $3 \times 4$ 矩阵表示，具有 11 个自由度。

要求秩为 3，这是因为如果秩小于 3，那么矩阵映射的值域变成一条线或者一个点，而非一个面，从而不是一个 2D 图像。

**ref**

1. Multiple View Geometry in computer vision
