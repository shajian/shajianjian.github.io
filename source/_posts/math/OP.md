---
title: 正交多项式
date: 2024-11-28 11:42:33
tags: math
mathjax: true
---

# 1. 简介

正交多项式（Orthogonal Polynomials）是数学中一类具有特殊性质的多项式。假设我们有一个权重函数 $w(x)$，在区间 $[a,b]$ 上非负。对于两个函数 $f(x), g(x)$，定义它们的加权内积为

$$\langle f,g\rangle=\int_a^b f(x) g(x) w(x)dx \tag{1}$$

这里 $w(x)$ 可以是概率密度函数，也可以不是。

__定义 1：__ 如果对于两个不同的多项式 $p_n(x), p_m(x)$，当且仅当 $m \ne n$ 时满足

$$\langle p_n, p_m \rangle=0 \tag{2}$$

称这两个多项式相互正交。

正交多项式序列 $p_n(x)$ 是一组多项式，其中 $p_n(x)$ 是 n 次的。

一些著名的正交多项式：

1. 勒让德多项式（Legendre Polynomials）：权重函数为

    $$w(x)=\begin{cases} 1 & x \in [-1,1]\\\\ 0 & \text{o.w.} \end{cases} \tag{3}$$
2. 拉盖尔多项式（Laguerre Polynomials）：权重函数为

    $$w(x)=\begin{cases} e^{-x} & x \in [0,\infty)\\\\ 0 & \text{o.w.} \end{cases} \tag{4}$$

3. 切比雪夫多项式（Chebyshev Polynomials）：权重函数为

    $$w(x)=\begin{cases} (1-x^2)-\frac 1 2 & x \in [-1,1]\\\\ 0 & \text{o.w.} \end{cases} \tag{5}$$

4. 赫米特多项式（Hermite Polynomials）：权重函数为

    $$w(x)=e-x^2, \quad x \in \mathbb R \tag{6}$$

# 2. 勒让德多项式

$$P_n(x)=\frac 1 {2^n n!} \frac {d^n}{d x^n} \left[(x^2-1)^n \right] \tag{7}$$

其中 $n \in \mathbb Z_+$ 。

## 2.1 性质

### 2.1.1 正交性

由于权重函数 $w(x)=1, \ x \in [-1, 1]$，根据 (2) 式计算得，

$$\int _{-1} ^ 1 P_m(x) P_n(x) dx=\begin{cases} 0 & m \ne n \\\\ \frac 2 {2n+1} & m=n \end{cases} \tag{8}$$

### 2.1.2 递归公式

$$(2n+1) P_n =P_{n+1}' - P_{n-1}' \tag{9}$$
$$P'_{n+1}=(n+1)P_n + x P_n' \tag{10}$$

<details>
<summary>证明</summary>

根据 (7) 式和导数定义易知，

$$\begin{aligned}P'_{n+1}(x)&=\frac 1 {2^{n+1} (n+1)!} \frac {d^{n+2}}{d x^{n+2}} \left[(x^2-1)^{n+1} \right]
\\\\ &=\frac 1 {2^{n+1} (n+1)!}  \frac {d^{n+1}}{d x^{n+1}}\left[(n+1)(x^2-1)^n \cdot 2x \right] 
\\\\ &=\frac 1 {2^n n!}  \frac {d^{n+1}}{d x^{n+1}}\left[(x^2-1)^n \cdot x \right]
\\\\ &=\frac 1 {2^n n!}  \frac {d^n}{d x^n}\left[n(x^2-1)^{n-1} \cdot 2x \cdot x + (x^2-1)^n \right]
\\\\ &=\frac 1 {2^{n-1} (n-1)!}\frac {d^n}{d x^n} \left[(x^2-1)^{n-1} x^2\right] + P_n(x)
\\\\ &=\frac 1 {2^{n-1} (n-1)!}\frac {d^n}{d x^n}\left[(x^2-1)^n + (x^2-1)^{n-1}\right] + P_n(x)
\\\\ &= \frac 1 {2^{n-1} (n-1)!} \frac {d^n}{d x^n}\left[(x^2-1)^{n-1} \right] + 2n P_n(x) + P_n(x)
\\\\ &= P'_{n-1}(x) + (2n+1) P_n(x)
\end{aligned}$$

(9) 式得证。接下来证明 (10) 式，易知

$(x^2-1)^n=\sum_{i=0}^n C_n^i (-1)^i x^{2(n-i)}$

其 n 阶导为

$$\frac d {dx} (x^2-1)^n=\sum_{i=0}^{n-1} [2(n-i)]C_n^i (-1)^i x^{2(n-i)-1}$$

$$\frac {d^2} {dx^2} (x^2-1)^n=\sum_{i=0}^{n-1} [2(n-i)][2(n-i)-1]C_n^i (-1)^i x^{2(n-i)-2}$$

$$\frac {d^n} {dx^n} (x^2-1)^n=\sum_{i=0}^{\lfloor n/2\rfloor} \prod_{j=0}^{n-1} [2(n-i)-j] \cdot C_n^i (-1)^i x^{2(n-i)-n}$$

于是有

$$P_n=\frac 1 {2^n n!} \sum_{i=0}^{\lfloor n/2\rfloor} \prod_{j=0}^{n-1} (2n-2i-j) \cdot C_n^i (-1)^i x^{n-2i}$$

$$P'_n=\frac 1 {2^n n!} \sum_{i=0}^{\lfloor (n-1)/2\rfloor} \prod_{j=0}^{n} (2n-2i-j) \cdot C_n^i (-1)^i x^{n-2i-1}$$

$$xP'_n=\frac 1 {2^n n!} \sum_{i=0}^{\lfloor (n-1)/2\rfloor} \prod_{j=0}^{n} (2n-2i-j) \cdot C_n^i (-1)^i x^{n-2i}$$

根据上式容易写出

$$P'_{n+1}=\frac 1 {2^{n+1} (n+1)!} \sum_{i=0}^{\lfloor n/2\rfloor} \prod_{j=0}^{n+1} (2n+2-2i-j) \cdot C_{n+1}^i (-1)^i x^{n-2i} \tag{11}$$

推导如下，

$$(n+1)P_n+xP'_n=\frac 1 {2^n n!}\sum_{i=0}^{\lfloor (n-1)/2\rfloor} \prod_{j=0}^{n-1} (2n-2i-j)\cdot (n+1+n-2i)\cdot C_n^i (-1)^i x^{n-2i} + \frac {n+1} {2^n n!}\sum_{i=\lfloor (n+1)/2\rfloor}^{\lfloor n/2\rfloor} \prod_{j=0}^{n-1} (2n-2i-j) \cdot C_n^i (-1)^i x^{n-2i}
\\\\ =\frac 1 {2^n n!}\sum_{i=0}^{\lfloor (n-1)/2\rfloor} \prod_{j=-1}^{n-1} (2n-2i-j)\cdot C_n^i (-1)^i x^{n-2i} + \frac {n+1} {2^n n!}\sum_{i=\lfloor (n+1)/2\rfloor}^{\lfloor n/2\rfloor} \prod_{j=0}^{n-1} (2n-2i-j) \cdot C_n^i (-1)^i x^{n-2i} \tag{12}$$

当 $n$ 为奇数时，上式 $+$ 号右侧项不存在，当 $n$ 为偶数时，右侧求和只有一项，对应为 $i=\lfloor n/2 \rfloor$。



$\forall i \in [0, \lfloor (n-1)/2\rfloor \ ]$ ，比较 (11) 和 (12) 式中的 $x^{n-2i}$ 的系数，

$$\begin{aligned}\frac {\frac 1 {2^n n!}\prod_{j=-1}^{n-1} (2n-2i-j)\cdot C_n^i (-1)^i}{\frac 1 {2^{n+1} (n+1)!} \prod_{j=0}^{n+1} (2n+2-2i-j) \cdot C_{n+1}^i (-1)^i}&=\frac {2(n+1)\prod_{j=-1}^{n-1} (2n-2i-j) C_n^i}{\prod_{j=0}^{n+1} (2n+2-2i-j) \cdot C_{n+1}^i }
\\\\ &=\frac {2(n+1)\prod_{j=-1}^{n-1} (2n-2i-j) (n+1-i)}{\prod_{j=0}^{n+1} (2n+2-2i-j) \cdot (n+1) }
\\\\ &=\frac {\prod_{j=-1}^{n-1} (2n-2i-j)}{\prod_{j=1}^{n+1} (2n+2-2i-j)}=1
\end{aligned}$$

__case 1:__ $n$ 为奇数，此时 $\lfloor (n-1)/2\rfloor =\lfloor n/2\rfloor$

__case 2:__ $n$ 为偶数，还要再考虑一项即 $i=\lfloor n/2 \rfloor$，此时有 $2i=n$，于是 $x^{n-2i}=1$，比较 (11) 和 (12) 式中的常数项，

$$\begin{aligned}\frac {\frac {n+1} {2^n n!} \prod_{j=0}^{n-1} (2n-2i-j) \cdot C_n^i (-1)^i} {\frac 1 {2^{n+1} (n+1)!} \prod_{j=0}^{n+1} (2n+2-2i-j) \cdot C_{n+1}^i (-1)^i}&=\frac {2(n+1)^2 \prod_{j=0}^{n-1} (n-j) C_n^i}{\prod_{j=0}^{n+1} (n+2-j) C_{n+1}^i}
\\\\ &= \frac {2(n+1)^2 \prod_{j=0}^{n-1} (n-j) (n+1-i)}{\prod_{j=0}^{n+1} (n+2-j) (n+1)}
\\\\ &=\frac {2(n+1) \prod_{j=0}^{n-1} (n-j) (n+1-i)}{\prod_{j=0}^{n+1} (n+2-j) }
\\\\ &= \frac {\prod_{j=-1}^{n-1} (n-j) (2n+2-2i)}{\prod_{j=0}^{n+1} (n+2-j) }
\\\\ &=\frac {\prod_{j=-1}^{n-1} (n-j)}{\prod_{j=1}^{n+1} (n+2-j) }=1
\end{aligned}$$

故 $x^{n-2i}$ 的系数完全相同，所以 (10) 式得证。
</details>

# 3. 拉盖尔多项式

$$L_n(x) = \frac {e^x} {n!} \frac {d^n}{dx^n} (e^{-x} x^n) \tag{13}$$

## 3.1 性质

### 3.1.1 正交性

由于权重函数 $w(x)=e^{-x}, \ x \in [0, \infty]$，根据 (2) 式计算得，

$$\int _0 ^ {\infty} e^{-x} L_m(x) L_n(x) dx=\begin{cases} 0 & m \ne n \\\\ (n!)^2 & m=n \end{cases} \tag{14}$$


## 3.2 广义拉盖尔多项式

$$L_n^a (x) = \frac 1 {n!} e^x x^{-a} \frac {d^n} {dx^n} (x^{a+n} e^{-x}) \tag{15}$$

### 3.2.1 正交性

$$\int_0^{\infty} e^{-x}x^a L_n^a (x) L_m^a (x) dx = \delta _ {mn} \cdot \frac {(n+a)!}{n!} \tag{16}$$

### 3.2.2 递归公式

$$\frac d {dx} L _ n ^a (x)=-L_{n-1} ^ {a+1} (x) \tag{17}$$

$$L _ n ^{a+1} (x)=\sum _ {i=0} ^ n L_i ^ a (x) \tag{18}$$

证明：

令 

$$F_n=x^{a+n} e^{-x}=\sum_{i=0}^{\infty} \frac {(-1)^i} {i!}x^{i+a+n}$$

那么

$$\begin{aligned}\frac {d^n}{dx^n} F_n =\frac {d^n}{dx^n} x^{a+n} e^{-x}&=\frac {d^{n-1}}{dx^{n-1}} [(a+n)x^{a+n-1}e^{-x}-x^{a+n}e^{-x}]
\\\\ &=(a+n)\frac {d^{n-1}}{dx^{n-1}}F_{n-1}-\frac {d^{n-1}}{dx^{n-1}}F_n
\end{aligned}$$

$$\begin{aligned}\frac {d^{n+1}}{dx^{n+1}} F_n &=\frac {d^{n}}{dx^{n}} [(a+n)x^{a+n-1}e^{-x}-x^{a+n}e^{-x}]
\\\\ &=(a+n)\frac {d^{n}}{dx^{n}}F_{n-1}-\frac {d^{n}}{dx^{n}}F_n
\end{aligned}$$

(17) 式左边为

$$\begin{aligned} \frac d {dx} L _ n ^a (x)&=\left(\frac 1 {n!} e^x x^{-a} \right)' \frac {d^n} {dx^n} F_n+\frac 1 {n!} e^x x^{-a} \frac {d^{n+1}} {dx^{n+1}} F_n
\\\\ &=\left(\frac 1 {n!} e^x x^{-a}-\frac a {n!}e^x x^{-a-1}\right)\frac {d^n} {dx^n} F_n + \frac 1 {n!} e^x x^{-a}\left((a+n)\frac {d^{n}}{dx^{n}}F_{n-1}-\frac {d^{n}}{dx^{n}}F_n\right)
\\\\ &=\frac {a+n} {n!}e^x x^{-a} \frac {d ^n}{dx^n} F_{n-1} - \frac a {n!}e^x x^{-a-1}\frac {d^n} {dx^n} F_n
\end{aligned}$$

计算

$$\begin{aligned} \frac {\frac d {dx} L _ n ^a (x)} {L_{n-1}^{a+1}(x)} &=
\frac {\frac {a+n} {n!}e^x x^{-a} \frac {d ^n}{dx^n} F_{n-1} - \frac a {n!}e^x x^{-a-1}\frac {d^n} {dx^n} F_n}
{\frac 1 {(n-1)!} e^x x^{-a-1} \frac {d^{n-1}} {dx^{n-1}} F_n }
\\\\ &=\frac {(a+n)x\frac {d ^n}{dx^n} F_{n-1} - a\frac {d^n} {dx^n} F_n}{n \frac {d^{n-1}} {dx^{n-1}} F_n}
\end{aligned} \tag{19}$$

由于

$$\frac {d^n} {dx^n} F_n=\sum_{i=0}^{\infty} \prod_{j=0}^{n-1} (i+a+n-j) \frac {(-1)^i}{i!}x^{i+a}$$

$$\frac {d^n} {dx^n} F_{n-1}=\sum_{i=0}^{\infty} \prod_{j=0}^{n-1} (i+a+n-1-j) \frac {(-1)^i}{i!}x^{i+a-1}$$

$$\frac {d^{n-1}} {dx^{n-1}} F_{n}=\sum_{i=0}^{\infty} \prod_{j=0}^{n-2} (i+a+n-j) \frac {(-1)^i}{i!}x^{i+a+1}$$

代入 (19) 式，分子为

$$(a+n)\sum_{i=0}^{\infty} \prod_{j=0}^{n-1} (i+a+n-1-j) \frac {(-1)^i}{i!}x^{i+a}-a\sum_{i=0}^{\infty} \prod_{j=0}^{n-1} (i+a+n-j) \frac {(-1)^i}{i!}x^{i+a}$$


$x^{i+a}$ 的系数为

$$C_i = \left[(a+n)\prod_{j=0}^{n-1}(i+a+n-1-j)-a\prod_{j=0}^{n-1} (i+a+n-j)\right] \frac {(-1)^i}{i!}=ni \prod_{j=1}^{n-1}(i+a+j)\cdot \frac {(-1)^i}{i!}$$

$$C_{i+1}=n(i+1) \prod_{j=1}^{n-1}(i+a+j+1)\cdot \frac {(-1)^{i+1}}{(i+1)!}= -n\frac {(-1)^i}{i!} \prod_{j=2}^n (i+a+j)$$

分母 $x^{i+a+1}$ 系数为

$$n \frac {(-1)^i}{i!} \prod_{j=0}^{n-2} (i+a+n-j)=n \frac {(-1)^i}{i!} \sum_{j=2}^n (i+a+j)$$

显然分子分母的每个 $x^{i+a+1}$ 项的系数均只差一个 $-1$ 因子，于是 (17) 式得证。下面证明 (18) 式，

左侧为

$$L _ n ^{a+1} (x)=\frac 1 {n!} e^x x^{-a-1}\frac {d^n}{dx^n}(x^{a+n+1}e^{-x})=\frac 1 {n!} e^x x^{-a-1} \sum_{i=0}^{\infty} \prod_{j=0}^{n-1} (i+a+n+1-j) \frac {(-1)^i}{i!}x^{i+a+1}$$

右侧为

$$\begin{aligned}\sum _ {i=0} ^ n L_i ^ a (x)&=\sum_{i=0}^n \frac 1 {i!} e^x x^{-a} \frac {d^i}{dx^i}(x^{a+i}e^{-x})
\\\\ &= \sum_{i=0}^n \frac 1 {i!} e^x x^{-a} \sum_{k=0}^{\infty} \prod_{j=0}^{i-1}(k+a+i-j)\frac {(-1)^k}{k!} x^{k+a}
\end{aligned}$$


左右两侧均约去 $e^x x^{-a}$ 因子，

$$LHS=\frac 1 {n!} \sum_{i=0}^{\infty} \prod_{j=0}^{n-1} (i+a+n+1-j) \frac {(-1)^i}{i!}x^{i+a}$$

$$\begin{aligned}RHS &=\sum_{k=0}^n \sum_{i=0}^{\infty}\frac 1 {k!} \prod_{j=0}^{k-1}(k+a+i-j)\frac {(-1)^i}{i!} x^{i+a}
\\\\ &= \sum_{i=0}^{\infty} \sum_{k=0}^n \frac 1 {k!} \prod_{j=0}^{k-1}(k+a+i-j)\frac {(-1)^i}{i!} x^{i+a}
\end{aligned}$$

如果能证明

$$\frac 1 {n!}  \prod_{j=0}^{n-1} (i+a+n+1-j) =\sum_{k=0}^n \frac 1 {k!} \prod_{j=0}^{k-1}(k+a+i-j) \tag{20}$$

那么 (18) 式就得证。

(20) 式左侧表示组合数 $C_{i+a+n+1}^n$ ，右侧表示组合数之和

$$\sum_{k=0}^n C_{i+a+k} ^ k$$

根据组合数的性质（根据组合数定义很容易证明）

$$C_{n+1}^m = C_n^m + C_n ^ {m-1}, \quad m \le n$$

于是有

$$\begin{aligned}C_{i+a+n+1}^n &= C_{i+a+n}^n + C_{i+a+n}^{n-1}
\\\\ &=  C_{i+a+n}^n + C_{i+a+n-1}^{n-1} + C_{i+a+n-1}^{n-2}
\\\\ & \vdots
\\\\ &= C_{i+a+n}^n + C_{i+a+n-1}^{n-1} + C_{i+a+n-2}^{n-2} + \cdots + C_{i+a+1}^1 + C_{i+a+1}^0
\\\\ &= C_{i+a+n}^n + C_{i+a+n-1}^{n-1} + C_{i+a+n-2}^{n-2} + \cdots + C_{i+a+1}^1 + C_{i+a}^0
\\\\ &= \sum_{k=0}^n C_{i+a+k} ^ k
\end{aligned}$$

于是 (20) 式得证，从而证明 (18) 式。

# 4. 切比雪夫多项式

## 4.1 第一类切比雪夫多项式

第一类切比雪夫多项式 $T_n(x)$ 在区间 $[-1, 1]$ 上正交，并对应于权重函数 $w(x)=\frac 1 {\sqrt {1-x^2}}$。这些多项式在 $x=\cos \theta$ 时可由下面的三角形式定义：

$$T_n(x)=\cos (n \cdot arc\cos x) \tag{21}$$

对于非负整数 $n$，第 $n$ 阶的第一类切比雪夫多项式可通过下面的罗德里格斯式给出：

$$T_n(x)=\frac n 2 \sum _ {k=0} ^ {\lfloor n/2 \rfloor} (-1)^k \frac {(n-k-1)!}{k! (n-2k)!} (2x) ^{n-2k} \tag{22}$$

正交性质：

$$\int _ {-1} ^ 1 \frac {T_m(x) T_n(x)}{\sqrt {1-x^2}} dx = \begin{cases} 0 & m \ne n \\\\ \pi & m=n=0 \\\\ \frac {\pi} 2 & m=n\ne 0 \end{cases} \tag{23}$$

递归公式：

$$T_{n+1}(x)=2x T_n(x) - T _ {n-1}(x) \tag{24}$$

## 4.2 第二类切比雪夫不等式

第二类切比雪夫多项式 $U_n(x)$ 在区间 $[-1, 1]$ 上正交，权重函数为 $w(x)=\sqrt {1 - x^2}$，可由下面的三角函数定义，

$$U_n(x)=\frac {\sin((n+1) arc\cos x)}{\sqrt {1 - x^2}} \tag{25}$$

对于非负整数 $n$，第 $n$ 阶的第二类切比雪夫多项式可通过下面的公式给出：

$$U_n(x)=\frac n 2 \sum _ {k=0} ^ {\lfloor n/2 \rfloor} (-1)^k \frac {(n-k)!}{k! (n-2k)!} (2x) ^{n-2k} \tag{26}$$

正交性质：

$$\int _ {-1} ^ 1 U_m(x) U_n(x)\sqrt {1-x^2} dx = \begin{cases} 0 & m \ne n  \\\\ \frac {\pi} 2 & m=n \end{cases} \tag{27}$$

递归公式：

$$U_{n+1}(x)=2x U_n(x) - U _ {n-1}(x) \tag{28}$$

# 5. 赫米特多项式

对于非负整数 $n$，第 $n$ 阶赫米特多项式定义为

$$H_n(x)=(-1)^n e ^{x^2} \frac {d^n}{d x^n}(e ^ {-x^2}) \tag{29}$$

**正交性**

赫米特多项式在带有权重函数 $w(x)=e^{-x^2}$ 的整个实数域上正交。

$$\int _ {-\infty} ^{\infty} e^{-x^2} H_m(x) H_n(x) dx = \begin{cases} 0 & m\ne n \\\\ 2^n n! \sqrt {\pi} & m=n \end{cases} \tag{30}$$

**递归公式**

$$H_{n+1}(x)=2x H_n(x) - 2n H_{n-1}(x)
\\\\ \frac d {dx} H_n(x) = 2n H_{n-1}(x) \tag{31}$$