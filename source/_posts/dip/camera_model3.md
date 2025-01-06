---
title: 相机模型（三）
date: 2024-05-08 14:35:27
tags: camera calibration
categories: 相机标定
mathjax: true
---

# 3. 相机矩阵的计算

根据一组 3D 空间点和对应的像点，使用数值法估计相机投影矩阵。当 $\mathbf X _ i \leftrightarrow \mathbf x _ i$ 足够多时，可以确定相机矩阵 $P$。

这里我们假设从 3D 空间到图像的映射是线性的（如果存在透镜畸变，那么这种假设是无效的）。

## 3.1 基本方程

给定 $n$ 个对应点 $\mathbf X _ i \leftrightarrow \mathbf x _ i$，要求估计 $P \in \mathbb R ^ {3 \times 4}$，使得 $\mathbf x _ i = P \mathbf X _ i, \ \forall i$ 均成立。

这与 [2D 单应](/2024/05/08/dip/2D_homography) 的情况很类似，所以类似地可以写出下式，

$$\begin{bmatrix}\mathbf 0 ^ {\top} & -w _ i \mathbf X _ i ^{\top}& y _ i \mathbf X _ i ^{\top} \\ w _ i \mathbf X _ i ^ {\top} & \mathbf 0 ^ {\top} & - x _ i  \mathbf X _ i ^ {\top} \\ -y _ i \mathbf X _ i ^ {\top} & x _ i \mathbf X _ i ^ {\top} & \mathbf 0 ^ {\top} \end{bmatrix}\begin{pmatrix}\mathbf P ^ 1 \\ \mathbf P ^ 2 \\ \mathbf P ^ 3\end{pmatrix}=\mathbf 0 \tag{1}$$

由于第三个等式可由另外两个等式表示，即只有两个线性无关的等式，那么进一步有

$$\begin{bmatrix}\mathbf 0 ^ {\top} & -w _ i \mathbf X _ i ^{\top}& y _ i \mathbf X _ i ^{\top} \\ w _ i \mathbf X _ i ^ {\top} & \mathbf 0 ^ {\top} & - x _ i  \mathbf X _ i ^ {\top} \end{bmatrix}\begin{pmatrix}\mathbf P ^ 1 \\ \mathbf P ^ 2 \\ \mathbf P ^ 3\end{pmatrix}=\mathbf 0 \tag{2}$$

其中 $P ^ {i\top}$ 是 $P$ 第 i 行向量，维度为 4，所以矩阵 $A$ 大小为 $2n \times 12$，简记为

$$A \mathbf p = \mathbf 0 \tag{3}$$

**# 最小解**

$P$ 有 12 个元素，自由度为 11，所以需要有 11 个方程来解 $P$。由于每个点对有 2 个方程，那么需要 5.5 个点对，其中 0.5 表示对于第 6 个点对，仅需要其中一个方程，也就是说只需要知道第 6 个点对的 x 坐标或 y 坐标即可。

给定最小数量的点对，那么可以精确的得到 $P$，也就是解 $A \mathbf p = \mathbf 0$，此时使用 5.5 个点对，$A$ 大小为 $11\times 12$，通常 $A$ 的秩为 11，$\mathbf p$ 是 $A$ 的核空间向量。

**# 超定解**

由于噪声，数据并不精确，给定 $n \ge 6$ 的点对，那么 $A\mathbf p=\mathbf 0$ 的解并不精确，我们根据代数或者几何误差来估计 $P$ 。

根据代数误差，求解目标为

$$\min \ ||A \mathbf p||
\\ s.t. \quad ||\mathbf p||=1, \ ||\hat {\mathbf p} ^ 3||=1$$

其中 $\hat {\mathbf p}^3=(p _ {31}, p _ {32}, p _ {33}) ^ {\top}$，即 $P$ 的最后一行的前 3 个元素。

与 2D 单应一样，使用 DLT 算法求解（SVD 分解中最小右奇异向量）。

## 3.2 几何误差

假设世界点 $\mathbf X _ i$ 比图像点远远准确，那么图像上几何误差为

$$\sum _ i d (\mathbf x _ i, \hat {\mathbf x} _ i) ^ 2$$

其中 $\mathbf x _ i$ 是测量点，$\hat {\mathbf x} _ i = P \mathbf X _ i$。如果测量误差是高斯型，那么求解目标为对 $P$ 的最大似然估计，

$$\min _ P \ \sum _ i d (\mathbf x _ i, P \mathbf X _ i)^2$$

**# 黄金标准算法**

目标：

给定 $n \ge 6$ 世界点与图像点的对应，求解 $P$ 的最大似然估计。

算法：

1. 线性解。使用归一化 DLT 算法，估算结果作为 $P$ 的初值

    - 归一化。使用一个相似变换 $T$ 将图像点归一化，另一个相似变换 $U$ 将空间点归一化，记为 $\tilde {\mathbf x}_i=T\mathbf x _ i$，$\tilde {\mathbf X} _ i=U\mathbf U _ i$

    - DLT。$A$ 大小为 $2n \times 12$，根据 $\tilde {\mathbf X}_i \leftrightarrow \tilde {\mathbf x}_i$ 的两个方程得到，解的 $\mathbf p$ 对应 $\tilde P$ 的元素。

2. 最小化几何误差。

    使用 Levenberg-Marquardt 迭代算法求解最小化

    $$\sum _ i d (\tilde {\mathbf x}_i, \tilde P \tilde {\mathbf X}_i) ^ 2$$

    $\tilde P$ 的初始值为上一步线性解。

3. 去归一化。

    $$P=T ^ {-1} \tilde P U$$

以上是估计从世界到图像的转换 $P$ 的黄金标准算法。

**# 世界点的误差**

世界点的 3D 几何误差为

$$\sum _ i d(\mathbf X _ i, \hat {\mathbf X} _ i) ^ 2$$

其中 $\hat {\mathbf X} _ i$ 是空间中最靠近 $\mathbf X _ i$ 且满足 $\mathbf x _ i = P \hat {\mathbf X} _ i$。

## 3.3 仿射相机的估计

仿射相机指的是投影矩阵的最后一行为 $(0,0,0,1)$。

假设点经过归一化，为 $\mathbf X _ i=(X _ i, Y _ i, Z _ i, 1)^{\top}$，$\mathbf x _ i=(x _ i, y_i,1)^{\top}$，并设是仿射相机，那么 (2) 式变为

$$\begin{bmatrix}\mathbf 0 ^ {\top} & -\mathbf X _ i ^{\top} 
\\  \mathbf X _ i ^ {\top} & \mathbf 0 ^ {\top} \end{bmatrix}
\begin{pmatrix}\mathbf P ^ 1 
\\ \mathbf P ^ 2\end{pmatrix} + \begin{pmatrix}y _ i \\ - x _ i\end{pmatrix}
=\mathbf 0 \tag{4}$$

那么代数误差为

$$||A \mathbf p||^2=\sum _ i(x _ i - \mathbf P ^ {1\top}\mathbf X _ i) ^ 2 + (y _ i - \mathbf P ^ {2\top}\mathbf X _ i) ^ 2=\sum _ i d(\mathbf x _ i, \hat {\mathbf x} _ i) ^ 2$$

即，代数误差等于几何误差。

以下给出从世界到图像的仿射相机矩阵 $P _ A$ 的估计方法（黄金标准算法）

目标：

给定 $n \ge 4$ 世界到图像的点对 $\{\mathbf X _ i \leftrightarrow \mathbf x _ i\}$，求仿射相机矩阵 $P _ A$ 的最大似然估计，即

$$\min _ P \ \sum _ i d(\mathbf x _ i, P\mathbf X _ i)^2, \ s.t. \ \mathbf P ^ {3\top}=(0,0,0,1) ^ {\top}$$

算法：

1. 归一化。使用一个相似变换 $T$ 归一化图像点，使用另一个相似变换 $U$ 归一化空间点，$\tilde {\mathbf x} _ i=T\mathbf x _ i$，$\tilde {\mathbf X} _ i=U\mathbf X _ i$，使得归一化后的坐标向量最后一个元素为 1。

2. 将每个归一化后的点对 $\{\tilde {\mathbf X} _ i \leftrightarrow \tilde {\mathbf x} _ i\}$ 根据 (4) 式得到
    $$\begin{bmatrix}\tilde {\mathbf X} _ i ^ {\top} & \mathbf 0 ^ {\top} 
    \\ \mathbf 0 ^ {\top} & \tilde {\mathbf X} _ i ^{\top} 
     \end{bmatrix}
    \begin{pmatrix}\tilde {\mathbf P} ^ 1 
    \\ \tilde {\mathbf P} ^ 2\end{pmatrix} = \begin{pmatrix} \tilde x _ i\\ \tilde y _ i \end{pmatrix}$$

    将 $n$ 个上式堆起来，得到 $A _ 8 \mathbf p _ 8=\mathbf b$，其中 $\mathbf p _ 8$ 是 $\tilde P _ A$ 的前两行组成的 8D 向量，$A _ 8$ 大小为 $2n \times 8$，$\mathbf b$ 大小是 $2n \times 1$ 由归一化图像点坐标组成。

3. 计算 $A _ 8$ 的伪逆 $A _ 8 ^ +$，那么 $\mathbf p _ 8=A _ 8 ^ +\mathbf b$，得到 $\tilde P _ A$ 的前两行，而第三行为 $(0,0,0,1)$

4. 去归一化。相机矩阵计算如下

    $$P _ A = T ^ {-1} \tilde P _ A U$$



**# 矩阵的伪逆**

一个对角方阵 $D$，其伪逆也是一个对角方阵 $D ^ +$，满足

$$D _ {ii} ^ + = \begin{cases} 0 & D _ {ii}=0 \\ D _ {ii} ^ {-1} & \text{o.w.}\end{cases}$$

对于一个 $m \times n$ 的矩阵，其中 $m \ge n$，计算 SVD 即 $A=UDV ^ T$，定义 $A$ 的伪逆为

$$A ^ + = VD ^ + U ^ {\top}$$

