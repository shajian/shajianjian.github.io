---
title: 2D 单应变换（一）
date: 2024-05-08 10:35:27
tags: computer vision
mathjax: true
---

2D 单应变换：从一个图像中的点映射到另一个图像中的点。

## 1. 2D 单应变换

给定一个图像中的点集 $\mathbf x _ i \in \mathbb P ^ 2$，以及对应的另一个图像中的点集 $\mathbf x _ i' \in \mathbb P ^ 2$，计算投影变换 $H:\mathbf x _ i \rightarrow \mathbf x _ i '$，使得 

$$\mathbf x _ i' = H \mathbf x _ i, \ \forall i \tag{1}$$

均成立。

使用齐次向量表示，由于是图像中的点坐标，所以 $\mathbf x, \mathbf x '$ 均为 3D 向量，所以 $H \in \mathbb R ^ {3 \times 3}$。

实际上 $H \mathbf x _ i$ 与 $\mathbf x _ i '$ 其实并不严格相等，它们方向相同，而幅度相差一个非零因子 $\lambda$，即

$$\mathbf x _ i' =\lambda H \mathbf x _ i$$

因为对于不同的 $\lambda$ 值，显然均保持着两个点之间的对应关系。

所以我们不使用 (1) 式表示，而是改用叉乘表示，因为叉乘结果为 $\mathbf 0$ 表示方向相同，而幅度不一定相同，

$$\mathbf x _ i' \times H \mathbf x _ i = \mathbf 0 \tag{2}$$

由于

$$H\mathbf x _ i = \begin{pmatrix}\mathbf h ^ {1\top} \mathbf x _ i \\ h ^ {2\top} \mathbf x _ i \\ h ^ {3\top} \mathbf x _ i \end{pmatrix}$$

其中 $\mathbf h ^ {j\top}$ 表示 $H$ 的第 j 行。 所以叉乘表示为

$$\mathbf x _ i ' \times H\mathbf x _ i = \begin{pmatrix}y _ i' \mathbf h ^ {3\top} \mathbf x _ i -w _ i' \mathbf h ^ {2\top} \mathbf x _ i \\ w _ i ' h ^ {1\top} \mathbf x _ i - x _ i' \mathbf h ^ {3\top} \mathbf x _ i \\ x _ i 'h ^ {2\top} \mathbf x _ i - y _ i' \mathbf h ^ {1\top} \mathbf x _ i \end{pmatrix}$$

其中 $\mathbf x _ i' = (x _ i', y _ i', w _ i') ^ {\top}$ 。由于 $\mathbf h ^ {j \top} \mathbf x _ i = \mathbf x _ i ^ {\top} \mathbf h ^ j$，改写上式为

$$\begin{bmatrix}\mathbf 0 ^ {\top} & -w _ i' \mathbf x _ i ^{\top}& y _ i' \mathbf x _ i ^{\top} \\ w _ i' \mathbf x _ i ^ {\top} & \mathbf 0 ^ {\top} & - x _ i ' \mathbf x _ i ^ {\top} \\ -y _ i' \mathbf x _ i ^ {\top} & x _ i' \mathbf x _ i ^ {\top} & \mathbf 0 ^ {\top} \end{bmatrix}\begin{pmatrix}\mathbf h ^ 1 \\ \mathbf h ^ 2 \\ \mathbf h ^ 3\end{pmatrix}=\mathbf 0 \tag{3}$$

(3) 式左侧是 $3 \times 9$ 的矩阵与 $9 \times 1$ 的矩阵相乘，表示 3 个等式，实际上这 3 个等式只有两个是线性无关的，第三个等式可由其他两个等式表示，所以进一步表示为

$$\begin{bmatrix}\mathbf 0 ^ {\top} & -w _ i' \mathbf x _ i ^{\top}& y _ i' \mathbf x _ i ^{\top} \\ w _ i' \mathbf x _ i ^ {\top} & \mathbf 0 ^ {\top} & - x _ i ' \mathbf x _ i ^ {\top} \end{bmatrix}\begin{pmatrix}\mathbf h ^ 1 \\ \mathbf h ^ 2 \\ \mathbf h ^ 3\end{pmatrix}=\mathbf 0 \tag{4}$$

简记为 

$$A _ i \mathbf h = \mathbf 0 \tag{5}$$

其中 $A _ i \in \mathbb R ^ {2 \times 9}$ 对应第 i 个3D 空间点和像点，$\mathbf h \in \mathbb R ^ {9 \times 1}$。


**求解 H**

每个点对有 2 个独立的方程，给定 4 个点对，我们可以得到 8 个方程，使用矩阵表示为 $A \mathbf h = \mathbf 0$，其中 $A$ 大小为 $8 \times 9$，$A$ 的秩为 8，那么 $A$ 的核空间是 1 维，$\mathbf h$ 就是核空间中的一个向量。

**基本 DLT 算法**

Direct Linear Transformation 算法总结如下：

目标：

给定 $n \ge 4$ 2D 到 2D 点的对应 $\{\mathbf x _ i \leftrightarrow \mathbf x _ i '\}$，求 2D 单应矩阵 $H$ 使得 $\mathbf x _ i' = H \mathbf x _ i$。

算法：

1. 为每个点对，计算 $A _ i$
2. 将 $n$ 个 $A _ i$ 组合为 $A$，size 为 $2n \times 9$
3. 对 $A$ 进行 SVD，最小奇异值对应的单位奇异向量就是 $\mathbf h$。

    具体而言，记 $A=UDV ^ {\top}$，其中 $D$ 中奇异值降序排列，那么 $\mathbf h$ 等于 $V$ 中最后一个列向量。

## 2. 归一化变换

1. 将图像平移，使得点集的中心位于原点
2. 将点的坐标进行伸缩变换，使得到原点的距离的平均等于 $\sqrt 2$，即这个“平均点”位于 $(1,1,1)^{\top}$
3. 两个图像分别独立的进行这种归一化变换

记点集 $\{\mathbf x _ i\} _ {i=1} ^ n$，那么均值点为

$$\overline {\mathbf x} = \frac 1 n \sum _ i ^ n \mathbf x _ i$$

平均距离为

$$d = \frac 1 n \sum _ {i=1}^n ||\mathbf x _ i- \overline {\mathbf x}|| _ 2$$

要得到归一化后的点集 $\{\tilde {\mathbf x} _ i \} _ {i=1} ^ n$，使其满足

$$\overline {\tilde {\mathbf x}}=\mathbf 0, \quad \tilde d = \sqrt 2$$

我们作变换

$$\tilde {\mathbf x} _ i=(\mathbf x _ i - \overline {\mathbf x}) \cdot \frac {\sqrt 2} d$$

那么

$$\overline {\tilde {\mathbf x}}=\frac 1 n \cdot \frac {\sqrt 2} d  \sum _ i (\mathbf x _ i - \overline {\mathbf x})=\mathbf 0$$

$$\tilde d = \frac 1 n \sum _ i ||\tilde {\mathbf x} _ i|| _ 2 = \frac {\sqrt 2} d \cdot \left(\frac 1 n \sum _ i ||\mathbf x _ i - \overline {\mathbf x}|| _ 2\right)= \frac {\sqrt 2} d \cdot d=\sqrt 2$$

显然这个变换满足要求。

以上说明中，坐标是 2D。现在使用齐次坐标，那么 $\mathbf x _ i = (x _ i , y _ i, 1), \ \tilde {\mathbf x}_i=(\tilde x _ i, \tilde y _ i, 1)$，均值点为 $\overline {\mathbf x}=(\overline x, \overline y, 1)$，变换为

$$\begin{bmatrix}\tilde x _ i \\ \tilde y _ i \\ 1\end{bmatrix}= \begin{bmatrix}\frac {\sqrt 2} d & 0 & -\overline x \cdot \frac {\sqrt 2} d \\ 0 & \frac {\sqrt 2} d  & - \overline y \cdot \frac {\sqrt 2} d \\ 0 & 0 & 1\end{bmatrix}\begin{bmatrix}x _ i \\ y _ i \\ 1\end{bmatrix}$$

下面给出归一化变换的源码

```python
import numpy as np
h, w = 640, 640     # 图像的高宽
n = 6               # 点数量
hs = np.random.randint(0, h, (n, ))
ws = np.random.randint(0, w, (n, ))
ps = np.stack((hs, ws), axis=-1)        # (n, 2)

m = np.mean(ps, axis=0)                 # (2, )    均值
d = ps - m                              # (n, 2)    
d = np.mean(np.linalg.norm(d, axis=1))  # scalar    距离平均值
scale = np.sqrt(2) / d
T = np.array([
    [scale, 0, -m[0] * scale],
    [0, scale, -m[1] * scale],
    [0, 0, 1]
])
ps = np.hstack((ps, np.ones((n, 1)))).T # (3, n)
ps2 = T.dot(ps)         # (3, n) 归一化后的点
```

**# 归一化 DLT 算法**

目标：

给定 $n \ge 4$ 的 2D 到 2D 的点对 $\{\mathbf x _ i \leftrightarrow \mathbf x _ i '\}$，确定 2D 单应变换，使得 $\mathbf x _ i' = H \mathbf x _ i$ 。

算法：

1. 归一化 $\mathbf x$，如上所说，变换为 $\tilde {\mathbf x} _ i = T \mathbf x _ i$
2. 归一化 $\mathbf x'$，如上所说，变换为 $\tilde {\mathbf x} _ i '= T' \mathbf x _ i'$
3. DLT，使用基本 DLT 算法得到 $\tilde {\mathbf x} _ i \leftrightarrow \tilde {\mathbf x} _ i'$ 的单应变换 $\tilde H$
4. 逆归一化，$H=T'^{-1}\tilde H T$

