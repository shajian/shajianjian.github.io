---
title: Denoising diffusion probabilistic models
date: 2022-06-27 15:26:21
tags: diffusion model
mathjax: true
img: ''
summary: 降噪的扩散模型
---

论文：[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

# 1. 原理
作者使用扩散模型实现了一种高质量图像的合成方法。扩散模型是一种参数化的马尔科夫链，马尔科夫链的每一步转移均表示在当前数据中增加噪声，如图 1 的 $q(\mathbf x_t|\mathbf x_{t-1})$，这称为前向过程，也称作扩散过程，训练模型以学习反向过程 $p_{\theta}(\mathbf x_{t-1}|\mathbf x_t)$，即从噪声中逐步恢复数据。

![](/images/diffusion_model/denoising_diffusion_model_1.png)

<center>图 1</center>

$\mathbf x_0$ 是图像数据，$\mathbf x_0 \sim q(\mathbf x_0)$

$\mathbf x_T$ 是高斯噪声，$p(\mathbf x_T)= \mathcal N(\mathbf x_T;\mathbf 0, I)$


马尔科夫链上每次转移均采用高斯分布（高斯转移），于是

__反向过程：__

$$p_{\theta}(\mathbf x_{0:T})=p(\mathbf x_T) \prod_{t=1}^T p_{\theta}(\mathbf x_{t-1}|\mathbf x_t), \quad p_{\theta}(\mathbf x_{t-1}|\mathbf x_t)=\mathcal N(\mathbf x_{t-1}; \mu_{\theta}(\mathbf x_t,t), \Sigma_{\theta}(\mathbf x_t, t)) \tag{1}$$


$p(\mathbf x_T)$ 使用一个简单的分布如高斯分布。

__前向过程：__

前向过程中，引入方差 $\beta_1, \ldots, \beta_T$ 以控制噪声，

$$q(\mathbf x_{1:T}|\mathbf x_0)=\prod_{t=1}^T q(\mathbf x_t|\mathbf x_{t-1}), \quad  q(\mathbf x_t|\mathbf x_{t-1})=\mathcal N(\mathbf x_t;\sqrt{1-\beta_t} \mathbf x_{t-1}, \beta_t I) \tag{2}$$

步数 $T$ 足够大，使得 $\mathbf x_T$ 为高斯噪声。

**方差 $\beta_t$ 既可以通过学习得到，也可以预设为一组固定的值**。

## 1.1 前向过程

前向过程的一个显著特性是可以得到 $\mathbf x_t$ 的分布的解析形式，根据 (2) 式，

$$q(\mathbf x_1|\mathbf x_0)=\mathcal N(\mathbf x_1;\sqrt {1-\beta_1} \mathbf x_0, \beta_1 I) \tag{3-1}$$

$$q(\mathbf x_2|\mathbf x_1)=\mathcal N(\mathbf x_2; \sqrt{1-\beta_2}\mathbf x_1, \beta_2 I)$$

对于两个随机向量 $\mathbf x_1, \mathbf x_2$，记联合概率密度分布的参数如下，

$$\mu = (\mu_1, \mu_2)$$

$$\Sigma=\begin{bmatrix} \sigma_1^2 & \rho \sigma_1 \sigma_2 \\\\ \rho \sigma_1 \sigma_2 & \sigma_2^2 \end{bmatrix}$$

那么条件分布为（这部分知识可参考相关教材），

$$p(x_1|x_2) = \mathcal N(\mu_1 + \rho \sigma_1 / \sigma_2 (x_2 - \mu_2), (1 - \rho^2) \sigma_1 ^2)$$

$$p(x_2|x_1) = \mathcal N(\mu_2 + \rho \sigma_2 / \sigma_1(x_1 - \mu_1), (1 - \rho^2) \sigma_2^2)$$

于是得到如下等式：

$$\sigma_1^2=\beta_1
\\\\ (1-\rho^2) \sigma_2^2=\beta_2
\\\\ \mu_1=\sqrt {1-\beta_1} \mathbf x_0
\\\\ \mu_2+ \rho \frac {\sigma_2}{\sigma_1} (\mathbf x_1-\sqrt{1-\beta_1}\mathbf x_0)=\sqrt {1-\beta_2}\mathbf x_1$$

其中，$\mathbf x_0$ 看作常量，所以 $q(\mathbf x_1|\mathbf x_0)$ 是 $\mathbf x_1$ 的 PDF，得到上述方程组的第一和第三个等式；$q(\mathbf x_2|\mathbf x_1)$ 是 $\mathbf x_2$ 的条件 PDF。

解上述方程组得，

$$\rho^2=\frac {\beta_1-\beta_1\beta_2}{\beta_1+\beta_2-\beta_1\beta_2}
\\\\ \sigma_2^2=\beta_1+\beta_2-\beta_1\beta_2=1-(1-\beta_1)(1-\beta_2)
\\\\ \mu_2=\rho \frac {\sigma_2}{\sigma_1} \sqrt{1-\beta_1} \mathbf x_0=\sqrt {(1-\beta_1)(1-\beta_2)}\mathbf x_0$$

记 $\alpha_t :=1 -\beta_t$，${\overline \alpha} _ t := \prod _ {s=1}^t \alpha_s$，于是 $\mathbf x_2$ 的分布可写为

$$q(\mathbf x_2|\mathbf x_0)=\mathcal N(\mathbf x_2;\sqrt{\overline \alpha_2}\mathbf x_0, (1-\overline \alpha_2)I) \tag{3-2}$$

根据归纳法，可得

$$q(\mathbf x_t|\mathbf x_0)=\mathcal N(\mathbf x_t; \sqrt {\overline \alpha_t}\mathbf x_0, (1-\overline \alpha_t)I) \tag{3}$$

注： $\mathbf x_0$ 在图像空间中也是一个随机变量，但是在选定一个图像 $\mathbf x_0$ 后，在前向过程和反向过程中的每个 step 中，$\mathbf x_0$ 均保持不变，而 $\mathbf x_t, \ t>0$ 是随机变量。

## 1.2 训练目标

学习的 **目标** 是使得通过模型预测得到的 $p_{\theta}(\mathbf x_0)$ 尽可能的大，这里 $\mathbf x_0$ 可看作是训练 target，来自训练集中的真实图片。

负对数似然为

$$\mathbb E_{q(\mathbf x_0)}[-\log p_{\theta}(\mathbf x_0)] \tag{4}$$

于是训练的目标变成 **最小化负对数似然**。

这里是 **针对真实图片的分布 $q(\mathbf x_0)$ 求期望**，当然图片的真实分布不可得，只有基于训练集的经验分布。

为了计算负对数似然，首先要计算 $p_{\theta}(\mathbf x_0)$ ，下面的推导过程省略了模型参数 $\theta$，目的是为了看起来简洁，

$$\begin{aligned}p(\mathbf x_0)&=\int p(\mathbf x_{0:T}) d \mathbf x_{1:T}
\\\\ &=\int p(\mathbf x_{0:T}) \frac {q(\mathbf x_{1:T}|\mathbf x_0)}{q(\mathbf x_{1:T}|\mathbf x_0)} d \mathbf x_{1:T}
\end{aligned} \tag{5}$$

$$\begin{aligned}\mathbb E_q[-\log p(\mathbf x_0)]&=-\int q(\mathbf x_0) \log p(\mathbf x_0) d\mathbf x_0
\\\\ &= -\int d\mathbf x_0 q(\mathbf x_0) \log \left[\int q(\mathbf x_{1:T}|\mathbf x_0) d \mathbf x_{1:T} \frac {p(\mathbf x_{0:T})}{q(\mathbf x_{1:T}|\mathbf x_0)}\right]
\end{aligned} \tag{6}$$

根据 Jensen 不等式，上式 $\log [\cdot]$ 部分存在不等式关系

$$\log [\cdot] \ge \int q(\mathbf x_{1:T}|\mathbf x_0) d \mathbf x_{1:T} \cdot \log \frac {p(\mathbf x_{0:T})}{q(\mathbf x_{1:T}|\mathbf x_0)} \tag{7}$$

将 (7) 式代入 (6) 式，得

$$\mathbb E_q[-\log p(\mathbf x_0)] \le -\int d \mathbf x_{0:T} q(\mathbf x_{0:T}) \cdot \log \frac {p(\mathbf x_{0:T})}{q(\mathbf x_{1:T}|\mathbf x_0)}=\mathbb E_q \left[-\log \frac {p(\mathbf x_{0:T})}{q(\mathbf x_{1:T}|\mathbf x_0)}\right]:=L \tag{8}$$

(8) 式就是训练目标，即 **最小化 $L$** 。(8) 式中 $L$ 是 **针对联合概率密度 $q(\mathbf x_{0:T})$ 求期望** 。 

下面分析目标 $L$ 。

$$\begin{aligned}L&=\mathbb E_q \left[-\log \frac {p_{\theta}(\mathbf x_{0:T})}{q(\mathbf x_{1:T}|\mathbf x_0)}\right]
\\\\ &=\mathbb E_q \left[-\log p(\mathbf x_T)-\sum_{t\ge 1} \log \frac {p_{\theta}(\mathbf x_{t-1}|\mathbf x_t)}{q(\mathbf x_t|\mathbf x_{t-1})}\right]
\\\\ &=\mathbb E_q \left[-\log p(\mathbf x_T)-\sum_{t> 1} \log \frac {p_{\theta}(\mathbf x_{t-1}|\mathbf x_t)}{q(\mathbf x_t|\mathbf x_{t-1})}-\log \frac {p_{\theta}(\mathbf x_0|\mathbf x_1)} {q(\mathbf x_1|\mathbf x_0)}\right]
\\\\ &=\mathbb E_q \left[-\log p(\mathbf x_T)-\sum_{t> 1} \log \frac {p_{\theta}(\mathbf x_{t-1}|\mathbf x_t)}{q(\mathbf x_{t-1}|\mathbf x_t, \mathbf x_0)} \cdot \frac {q(\mathbf x_{t-1}|\mathbf x_0)}{q(\mathbf x_t|\mathbf x_0)}-\log \frac {p_{\theta}(\mathbf x_0|\mathbf x_1)} {q(\mathbf x_1|\mathbf x_0)}\right]
\end{aligned} \tag{9}$$

上式最后一步推导利用了贝叶斯定理，

$$q(\mathbf x_t|\mathbf x_{t-1})=\frac {q(\mathbf x_{t-1}, \mathbf x_t|\mathbf x_0)}{q(\mathbf x_{t-1}|\mathbf x_0)}=\frac {q(\mathbf x_{t-1}|\mathbf x_t, \mathbf x_0)q(\mathbf x_t|\mathbf x_0)}{q(\mathbf x_{t-1}|\mathbf x_0)}$$

继续分析 $L$，求和项中的 

$$\sum_{t>1} \log \frac {q(\mathbf x_{t-1}|\mathbf x_0)}{q(\mathbf x_t|\mathbf x_0)}=\log q(\mathbf x_1|\mathbf x_0)-\log q(\mathbf x_T|\mathbf x_0)$$

于是 (9) 进一步转为

$$\begin{aligned}L=\mathbb E_q \left[-\log \frac {p(\mathbf x_T)}{q(\mathbf x_T|\mathbf x_0)}-\sum_{t>1} \log \frac {p_{\theta}(\mathbf x_{t-1}|\mathbf x_t)}{q(\mathbf x_{t-1}|\mathbf x_t, \mathbf x_0)}-\log p_{\theta}(\mathbf x_0|\mathbf x_1) \right]
\end{aligned} \tag{9}$$

一方面我们已知 KL 散度为

$$D_{KL}(p_1(x)||p_2(x))=\int p_1(x) \log \frac {p_1(x)}{p_2(x)}dx = E_{p_1} \left[\log \frac {p_1(x)}{p_2(x)} \right]= E_{p_1} \left[-\log \frac {p_2(x)}{p_1(x)} \right]$$

另一方面，

$$\begin{aligned}\mathbb E_q \left[-\log \frac {p(\mathbf x_T)}{q(\mathbf x_T|\mathbf x_0)}\right]&=\int q(\mathbf x_{0:T}) d \mathbf x_{0:T}\left[-\log \frac {p(\mathbf x_T)}{q(\mathbf x_T|\mathbf x_0)}\right]
\\\\ &=\int q(\mathbf x_0,\mathbf x_T) d\mathbf x_0 d\mathbf x_T \left[-\log \frac {p(\mathbf x_T)}{q(\mathbf x_T|\mathbf x_0)}\right]
\\\\ &=D_{KL}(q(\mathbf x_T|\mathbf x_0)||p(\mathbf x_T))
\end{aligned}$$

注意条件熵表达式为 $H(Y|X)=-\int p(x,y) \log p(y|x)$ 。


于是 (9) 式最终可转变为

$$L = \underbrace {D_{KL}(q(\mathbf x_T | \mathbf x_0) | p(\mathbf x_T))}_ {L_T} + \sum_{t>1}\underbrace {D_{KL}(q(\mathbf x_{t-1} | \mathbf x_t, \mathbf x_0) || p_{\theta}(\mathbf x_{t-1} | \mathbf x_t))}_ {L_{t-1}} - \underbrace {\mathbb E_q[\log p_{\theta}(\mathbf x_0 | \mathbf x_1)]}_{L_0} \tag{10}$$

上式使用 KL 距离来比较 $p_{\theta}(\mathbf x_{t-1}|\mathbf x_t)$ 与 前向后验分布 $q(\mathbf x_{t-1}|\mathbf x_t, \mathbf x_0)$ 之间的差距，这个前向后验分布是可以计算的，

$$q(\mathbf x_{t-1}|\mathbf x_t, \mathbf x_0)=\mathcal N(\mathbf x_{t-1}; \tilde {\mu}_t(\mathbf x_t, \mathbf x_0), \tilde {\beta}_t I) \tag {11}$$

其中 

$$\tilde \mu_t (\mathbf x_t, \mathbf x_0) = \frac {\sqrt {\overline \alpha_{t-1}} \beta_t} {1 - \overline \alpha_t} \mathbf x_0 + \frac {\sqrt {\alpha_t} (1 - \overline \alpha_{t-1})} {1 - \overline \alpha_t} \mathbf x_t, \quad \tilde \beta_t = \frac {1 - \overline \alpha_{t-1}} {1 - \overline \alpha_t} \beta_t \tag{12}$$

解：

根据 (2) 和 (3) 式知，

$$q(\mathbf x_t|\mathbf x_{t-1}) = \mathcal N(\mathbf x_t; \sqrt {1-\beta_t}\mathbf x_{t-1}, \beta_t I)
\\\\ q(\mathbf x_{t-1}|\mathbf x_0) = \mathcal N(\mathbf x_{t-1}; \sqrt {\overline \alpha_{t-1}} \mathbf x_0, (1 -\overline \alpha_{t-1})I)
\\\\ q(\mathbf x_t|\mathbf x_0) = \mathcal N(\mathbf x_t; \sqrt {\overline \alpha_t} \mathbf x_0, (1 -\overline \alpha_t)I)$$

以及解 (3-2) 过程中用到的等式关系 

$$p(x_1 |x_2)=\mathcal N(\mu_1 +\rho \sigma_1 /\sigma_2 (x_2 -\mu_2), (1 -\rho^2) \sigma_1^2)
\\\\ p(x_2 |x_1)=\mathcal N(\mu_2 +\rho \sigma_2 /\sigma_1(x_1 -\mu_1), (1 -\rho^2) \sigma_2^2)$$

可得

$$(1 - \rho^2 ) (1 -\overline \alpha_t) = \beta_t
\\\\ \tilde \mu_t (\mathbf x_t, \mathbf x_0) = \rho \sqrt{ \frac {1 - \overline \alpha_{t-1} }{ 1 - \overline \alpha_t } }(\mathbf x_t -\sqrt {\overline \alpha_t}\mathbf x_0)
\\\\ \tilde \beta_t = (1 - \rho^2)  (1 - \overline \alpha_{t-1})$$

解上述方程组可得最终结果。



### 1.2.1 前向过程和 $L_T$

我们固定方差 $\beta_t$，通过一个事先已知的规则来确定所有 $\beta_t$ 值，那么 $\overline {\alpha}_t$ 也全部确定，根据 (3) 式，$q(\mathbf x_t|\mathbf x_0)$ 是一个确定的概率分布，而 $p(\mathbf x_T)=\mathcal N(\mathbf x_T; \mathbf 0, I)$ 是高斯噪声，于是 $L_T$ 则与所要学习的参数 $\theta$ 无关，可将 $L_T$ 看作常量。

我们需要确保 $\beta_t$ 的值足够小，以便前向过程和反向过程有相同的函数形式，这样反向过程也可以使用高斯分布。

$\beta_t$ 选择方案如：

1. $\beta_1=10^{-4}$ 线性增加到 $\beta_T=0.02$

### 1.2.2 反向过程和 $L_{1:T-1}$

反向过程

$$p_{\theta}(\mathbf x_{t-1}|\mathbf x_t)=\mathcal N(\mathbf x_{t-1}; \mu_{\theta}(\mathbf x_t, t), \Sigma_{\theta}(\mathbf x_t, t)), \quad 1 < t \le T$$

注意上式指明了 $1<t \le T$，这点与 (1) 式不同，实际上 $t=1$ 时，$\mathbf x_0$ 是离散变量（因为像素值是整型），具体见下文，(1) 式为了简单起见，没有对这个离散区别处理。 

为了简单起见，令高斯分布各向同性，于是

$$\Sigma_{\theta}(\mathbf x_t, t)=\sigma_t^2 I$$

$$p_{\theta}(\mathbf x_{t-1}|\mathbf x_t)=\mathcal N(\mathbf x_{t-1}; \mu_{\theta}(\mathbf x_t, t), \sigma_t^2I) \tag{13}$$

实验中，可以设置 $\sigma_t^2=\beta_t$ 或 $\sigma_t^2=\tilde \beta_t$ ，这两种方法有着相似的实验结果。

于是 $\Sigma_{\theta}(\mathbf x_t, t)$ 也不需要学习。我们接着看如何得到期望 $\mu_{\theta}(\mathbf x_t, t)$ 。

根据 (9) 式可知，

$$L_{t-1}=\mathbb E_q\left[-\log \frac {p_{\theta}(\mathbf x_{t-1}|\mathbf x_t)}{q(\mathbf x_{t-1}|\mathbf x_t, \mathbf x_0)}\right]$$

将 (13) 式和 (11) 式代入到上式中有，

$$\begin{aligned}L_{t-1}&=\mathbb E_q \left[\frac 1 {2\sigma_t^2} ( \| \mathbf x_{t-1} -\mu_{\theta}(\mathbf x_t, t) \| ^2 - \| \mathbf x_{t-1} -\tilde \mu_t(\mathbf x_t, \mathbf x_0) \| ^2)\right]+C
\\\\ &=\mathbb E_q \left[\frac 1 {2\sigma_t ^2} (\mu_{\theta}^{\top} \mu_{\theta} - 2\mu_{\theta}^{\top} \mathbf x_{t-1} - \tilde \mu_t^{\top}\tilde \mu_t +2\tilde \mu_t^{\top} \mathbf x_{t-1})\right]+C
\end{aligned}$$

这里使用了等式 $\sigma_t^2=\tilde \beta_t$，且 $\tilde \beta_t$ 与学习参数 $\theta$ 无关，训练过程中看作常数。

由于 $\tilde \mu_t$ 是 $\mathbf x_0, \mathbf x_t$ 的函数，$\mu_{\theta}$ 是 $\mathbf x_t, t$ 的函数，反向过程中，$\mathbf x_t$ 是模型的一个输入，$\mathbf x_0$ 是真实图像数据， $\mathbf x_{t-1}$ 是生成模型最终得到的输出，所以将 $\mathbf x_{t-1}$ 的期望提取到内部并计算，

$$\begin{aligned}L_{t-1}&=\mathbb E_{q(\mathbf x_0, \mathbf x_t)} \left[\mathbb E_{q(\mathbf x_{t-1}|\mathbf x_t, \mathbf x_0)} [\frac 1 {2\sigma_t ^2 }(\mu_{\theta}^{\top} \mu_{\theta} - 2\mu_{\theta}^{\top} \mathbf x_{t-1} -\tilde \mu_t^{\top}\tilde \mu_t+2\tilde \mu_t^{\top} \mathbf x_{t-1})]\right] +C
\\\\ &=\mathbb E_{q(\mathbf x_0, \mathbf x_t)} \left[\frac 1 {2\sigma_t ^2 }(\mu_{\theta}^{\top} \mu_{\theta} - 2\mu_{\theta}^{\top} \tilde \mu_t -\tilde \mu_t^{\top}\tilde \mu_t+2\tilde \mu_t^{\top} \tilde \mu_t)\right] +C
\\\\ &=\mathbb E_{q(\mathbf x_0, \mathbf x_t)} \left[\frac 1 {2 \sigma_t ^2 } \|\tilde \mu_t(\mathbf x_t, \mathbf x_0) - \mu_{\theta}(\mathbf x_t, t)\|^2\right]+C
\end{aligned}$$

上式中， $C$ 是常量，根据 (11,12) 式可知 $\mathbb E_{q(\mathbf x_{t-1}|\mathbf x_t, \mathbf x_0)}[\mathbf x_{t-1}]=\tilde \mu_t$ 。

根据上式可知，要使得 $L_{t-1}$ 最小，那么对于任意的一组训练数据 $(\mathbf x_0, \mathbf x_t)$，损失函数$\|\tilde \mu_t(\mathbf x_t, \mathbf x_0) - \mu_{\theta}(\mathbf x_t, t)\|^2$ 都需要最小。

总结一下:

1. 我们的初衷是使得生成模型生成的图像数据的似然最大，或负对数似然最小。
2. 要使得负对数似然最小，那就需要令 $L$ 最小，从而 $L_{t-1}$ 也需要最小。
3. 生成模型对应马尔科夫链的反向过程（的一次转移过程）。

训练时，模型输入为 $\mathbf x_t$（通过 (3) 式采样得到），target 为 $\epsilon$，即训练 data pair 为 $(\mathbf x_t, \epsilon)$，注意 $\mathbf x_t$ 是与 $\epsilon$ 成对的，而不是 $\mathbf x_0$。同一个 $\mathbf x_0$，不同的 $t$ 会生成不同的 $\mathbf x_t$ 。模型最终得到 $\mathbf x_{t-1}$ 的概率为 $p_{\theta}(\mathbf x_{t-1}|\mathbf x_t)$，要使得 $L_{t-1}$ 最小，那么需要使得 $\|\tilde \mu_t(\mathbf x_t, \mathbf x_0) - \mu_{\theta}(\mathbf x_t, t)\|^2$ 最小 。

**计算 $\tilde \mu_t(\mathbf x_t, \mathbf x_0)$**

根据 (3) 式采样得到 $\mathbf x_t=\sqrt {\overline \alpha_t} \mathbf x_0+\sqrt {1-\overline \alpha_t} \epsilon, \quad \epsilon \sim \mathcal N(\mathbf 0, I)$ ，

变换得 $\mathbf x_0=1/\sqrt {\overline \alpha_t} (\mathbf x_t-\sqrt {1-\overline \alpha_t} \epsilon)$，代入 (12) 式得，

$$\tilde \mu_t(\mathbf x_t, \mathbf x_0)=\frac 1 {\sqrt {\alpha_t}} (\mathbf x_t-\frac {\beta_t}{\sqrt {1-\overline \alpha_t}} \epsilon) \tag{14}$$

由于模型并不直接输出 $\mu_{\theta}$，使用重参数技巧，

$$\mu_{\theta}(\mathbf x_t, t)=\frac 1 {\sqrt {\alpha_t}} (\mathbf x_t-\frac {\beta_t}{\sqrt {1-\overline \alpha_t}} \epsilon_{\theta}(\mathbf x_t, t)) \tag{15}$$

这里 **$\epsilon$ 是 target，模型的输出记为 $\epsilon_{\theta}(\mathbf x_t, t)$** 。

$\mathbf x_{t-1}$ 不是模型的直接输出，而是根据模型的直接输出 $\epsilon_{\theta}(\mathbf x_t, t)$ 间接采样得到。

于是，只要 $\|\epsilon - \epsilon_{\theta}(\mathbf x_t, t)\|^2$ 最小，就能使得 $\|\tilde \mu_t(\mathbf x_t, \mathbf x_0) - \mu_{\theta}(\mathbf x_t, t)\|^2$ 最小，从而 $L_{t-1}$ 最小，将 (14) (15) 两式代入 $L_{t-1}$ 表达式得，

$$L_{t-1}=\mathbb E_{\mathbf x_0, \epsilon} \left[\frac {\beta_t^2}{2 \sigma_t^2 \alpha_t (1-\overline \alpha_t)}\|\epsilon-\epsilon_{\theta}(\mathbf x_t, t)\|^2\right]$$

上式 $L_{t-1}$ 是 variational bound，即变分上边界，这可以从 (8) 式中知道。然而作者发现使用这个 variational bound 的一个变体形式更简单且训练有效，如下

$$L_{simple}(\theta)=\mathbb E_{t,\mathbf x_0, \epsilon} \left[\|\epsilon-\epsilon_{\theta}(\mathbf x_t, t)\|^2\right]$$

采样 $\mathbf x_{t-1} \sim p_{\theta}(\mathbf x_{t-1}|\mathbf x_t)$，可得 

$$\mathbf x_{t-1}=\mu_{\theta}(\mathbf x_t, t)+\sigma_t \mathbf z=\frac 1 {\sqrt {\alpha_t}} (\mathbf x_t-\frac {\beta_t}{\sqrt {1-\overline \alpha_t}} \epsilon_{\theta}(\mathbf x_t, t))+\sigma_t \mathbf z \tag{16}$$

其中 $\mathbf z \sim \mathcal N(\mathbf 0, I)$ ，$1<t\le T$。

下面给出算法，

![](/images/diffusion_model/denoising_diffusion_model_2.png)

<center>图 2</center>

**模型总结**

训练阶段，模型的输入是 $\mathbf x_t=\sqrt {\overline \alpha_t} \mathbf x_0+\sqrt {1-\overline \alpha_t} \epsilon$，其中 $\epsilon$ 取样自 $\mathcal N(\mathbf 0, I)$ 且作为 target，$t$ 从 $1,\ldots, T$ 中均匀随机选取，模型的输出为 $\epsilon_{\theta}(\mathbf x_t, t)$ ，模型输入是 $\mathbf x_t$，训练目标是最小化 $\|\epsilon-\epsilon_{\theta}\|^2$ ，使得反向过程的转移分布 $p_{\theta}(\mathbf x_{t-1}|\mathbf x_t)$ 尽量逼近前向过程的后验分布 $q(\mathbf x_{t-1}|\mathbf x_t)$ 。这样训练好后，就可以根据 $\mathbf x_t$ 以及模型输出 $\epsilon_{\theta}(\mathbf x_t, t)$ 采样得到 $\mathbf x_{t-1}$ （即 (16) 式）。

采样阶段，初始时从高斯噪声中采样得到 $\mathbf x_T$，然后模型的输入是上一轮计算的 $\mathbf x_t$（初始为 $\mathbf x_T$），输出是 $\boldsymbol \epsilon_{\theta}(\mathbf x_t, t)$，根据模型的输出可以计算出本轮的 $\mathbf x_{t-1}$ 。注意 $t=1$ 时，$\mathbf z=\mathbf 0$，原因见下方关于 $L_0$ 的说明。

由于 $\epsilon_{\theta}$ ，$\epsilon$ 与 $\mathbf x_0$ shape 相同，所以模型可以选择 segmentation 任务中的模型，例如 UNet （参考 文章[<sup>1</sup>](#refer-anchor-1) 中的 UNet 结构图）。

**$\sigma_t^2$ 的选择**

论文中有这么一句话：

> The first choice is optimal for $x_0 ∼ \mathcal N (0, I)$, and the second is optimal for $x_0$ deterministically set to one point. These are the two extreme choices corresponding to upper and lower bounds on reverse process entropy for data with coordinatewise unit variance. 

如何理解？

根据论文 《Deep Unsupervised Learning using Nonequilibrium Thermodynamics》[<sup>2</sup>](#refer-anchor-2) 中的推导结论，反向路径的条件熵 $H_q ( X^{(t-1)} | X^{(t)} )$ 满足

$$H_q( X^{(t)} | X^{(t-1)}) \ge H_q( X^{(t-1)} | X^{(t)}) \ge H_q( X^{(t)} | X^{(t-1)}) + H_q( X^{(t-1)} | X^{(0)}) - H_q( X ^{(t)} | X^{(0)})$$

1. 条件熵上限为 

    $$H_q(X^{(t-1)} | X^{(t)}) = H_q( X ^{(t)} | X^{(t-1)})$$

    前向过程转为概率为 

    $$q(\mathbf x_t|\mathbf x_{t-1})=\mathcal N(\sqrt {1-\beta_t} \mathbf x_{t-1}, \beta_t I)$$

    反向过程的转换概率为

    $$p_{\theta}(\mathbf x_{t-1}|\mathbf x_t)=\mathcal N(\mu_{\theta}(\mathbf x_t, t), \Sigma_{\theta}(\mathbf x_t, t))=\mathcal N(\mu_{\theta}(\mathbf x_t, t), \sigma_t^2I)$$

    再根据高斯分布的熵

    $$H[\mathcal N(\mu, \sigma^2I)]=\frac D 2 (\log 2\pi +1)+\frac 1 2\log \sigma^2$$

    也就是说熵与方差的对数成正线性关系，综上四个等式可得，

    $$\sigma_t^2=\beta_t$$

    根据论文 [2](#refer-anchor-2) 的推导，条件熵取上限的条件是 $H_q(X^{(t)}) = H_q(X^{(t-1)})$ ，根据 $\mathbf x_t=\sqrt {\overline \alpha_t} \mathbf x_0+\sqrt {1-\overline \alpha_t} \epsilon$，可知方差 

    $$V(X^{(t)})=\overline \alpha_t V(X^{(0)})+1-\overline \alpha_t=\overline \alpha_{t-1} V(X^{(0)})+1-\overline \alpha_{t-1} \Rightarrow V(X^{(0)})=1$$

2. 条件熵下限为 

    $$H_q(X^{(t-1)} | X^{(t)}) = H_q(X^{(t)} | X^{(t-1)}) +H_q(X^{(t-1)} |X ^{(0)} ) - H_q(X^{(t)} |X^{(0)})$$

    前向过程转为概率为 

    $$q(\mathbf x_t |\mathbf x_{t-1})=\mathcal N(\sqrt {1-\beta_t} \mathbf x_{t-1}, \beta_t I)
    \\\\ q(\mathbf x_t |\mathbf x_0)=\mathcal N(\sqrt {\overline \alpha_t}\mathbf x_0, (1-\overline \alpha_t)I)$$

    反向过程的转换概率为

    $$p_{\theta}(\mathbf x_{t-1} |\mathbf x_t)=\mathcal N(\mu_{\theta}(\mathbf x_t, t), \Sigma_{\theta}(\mathbf x_t, t))=\mathcal N(\mu_{\theta}(\mathbf x_t, t), \sigma_t^2I)$$

    再根据高斯分布的熵

    $$H[\mathcal N(\mu, \sigma^2I)]=\frac D 2 (\log 2\pi +1)+\frac 1 2\log \sigma^2$$

    也就是说熵与方差的对数成正线性关系，综上五个等式可得，

    $$\log \sigma_t^2=\log \beta_t+ \log (1-\overline \alpha_{t-1}) - \log(1-\overline \alpha_t)$$

    即 
    $$\sigma_t^2=\frac {1-\overline \alpha_{t-1}}{1-\overline \alpha_{t}} \beta_t$$

    根据论文 [2](#refer-anchor-2) 的推导，条件熵取下限的条件是 $H_q(X ^{(0)} | X^{(t)}) = H_q(X ^{(0)} | X^{(t-1)})$ ，即

    $$H_q(X ^{(t-1)}) - H_q(X ^{(t)})=H_q(X ^{(t-1)} | X ^{(0)})-H_q(X ^{(t)} | X ^{(0)})$$

    于是，

    $$\log \frac {\overline \alpha_{t-1}V(X^{(0)})+1-\overline \alpha_{t-1}}{\overline \alpha_t V(X^{(0)})+1-\overline \alpha_t}=\log \frac {1-\alpha_{t-1}} {1-\overline \alpha_t}$$

    解得 $V(X^{(0)})=0$，即 $x_0$ is deterministically set to one point。


### 1.2.3 $L_0$

要最小化 $L$，还需要考虑最小化 $L_0$ 。

**数据缩放**

将图片数据归一化到 $[-1,1]$ 范围内，这样处理与反向过程中 $\mathbf x_T$ 使用标准正态分布 $p(\mathbf x_T)$ 在数据尺度上保持一致。

由于图像像素值为整数 $\{0, 1, \ldots, 255\}$，通过 

$$\mathbf x_0:=\frac 2 {255} \mathbf x_0 - 1 \tag{17}$$

缩放到 $[-1, 1]$ 范围内仍是离散值，我们从高斯分布 $\mathcal N(\mathbf x_0; \mu_{\theta}(\mathbf x_1, 1), \sigma_1^2 I)$ 中推导出一个各像素独立的离散分布，

$$p_{\theta}(\mathbf x_0|\mathbf x_1)=\prod_{i=1}^D \int_{\sigma_-(x_0 ^i )} ^{\sigma_+(x_0 ^i )} \mathcal N(x; \mu_{\theta}^i(\mathbf x_1, 1), \sigma_1^2) dx \tag{18}$$

其中 $D$ 为图像数据的维度，$i$ 表示图中像素 index，且

$$\sigma_+(x)=\begin{cases}\infty & x=1 \\\\ x+1/255 & x < 1\end{cases}, \quad \sigma_-(x)=\begin{cases}-\infty & x=-1 \\\\ x-1/255 & x >- 1\end{cases}$$


映射到 $[-1,1]$ 范围后，像素值 $x_0$ 的取值范围为 $\{-1, -\frac {253}{255}, \ldots, \frac {253}{255}, 1\}$，相邻值相差 $2/255$，故以 $x_0$ 为中心，$x \in [x_0-\frac 1 {255}, x_0+\frac 1 {255}]$，对于边界值 $x_0=-1, 1$ ，则将 $x$ 的范围向外延伸到使得能覆盖整个实数范围，就得到 $\sigma_+(x), \ \sigma_-(x)$ 的表达式，展开如下表所示，

$$\begin{array} {c|cc}
像素值 \ x_0 & 归一化像素值 \ x_0 & 积分变量 \ x \\\\
\hline
0 & -1 & (-\infty, -\frac {254}{255}] \\\\
1 & -\frac {253}{255} & [-\frac {254}{255}, - \frac {252}{255}] \\\\
\vdots & \vdots & \vdots \\\\
254 & \frac {253}{255} & [\frac {252}{255}, \frac {254}{255}] \\\\
255 & 1 & [\frac {254}{255}, +\infty)
\end{array}$$

根据 (18) 式可以计算 $L_0$ 如下，

$$L_0=\mathbb E_q [-\log p_{\theta}(\mathbf x_0|\mathbf x_1)]=\mathbb E_{q(\mathbf x_1)} [\mathbb E_{q(\mathbf x_0|\mathbf x_1)} [-\log p_{\theta}(\mathbf x_0|\mathbf x_1)]]$$

注意 $\mathbb E_q$ 是针对 $q(\mathbf x_{0:T})$ 求期望，这在上文推导中已经说明。去除不相关的 $\mathbf x_{2:T}$ 得到上式最后的表达式。

这里 $\mathbf x_1$ 作为模型输入，使用经验分布（即训练样本集，根据 (3) 式和 $\mathbf x_0$ 采样得到 $\mathbf x_1$）作为真实分布 $q(\mathbf x_1)$ 来计算 $\mathbb E_{q(\mathbf x_1)}[\cdot]$ 。

$\mathbb E_{q(\mathbf x_0|\mathbf x_1)} [-\log p_{\theta}(\mathbf x_0|\mathbf x_1)]$ 是交叉熵 $H(q(\mathbf x_0|\mathbf x_1),p_{\theta}(\mathbf x_0|\mathbf x_1))$，故 $L_0$ 其实就是以 $\mathbf x_1$ 作为输入时模型输出的交叉熵损失，当 $p_{\theta}(\mathbf x_0|\mathbf x_1)=q(\mathbf x_0|\mathbf x_1)$ 时交叉熵损失取得最小值。

前向过程的后验分布 $q(\mathbf x_0|\mathbf x_1)$ 其实也应该是一个离散分布，离散化处理过程与 (18) 式完全一致，故根据 $p_{\theta}(\mathbf x_0|\mathbf x_1)=q(\mathbf x_0|\mathbf x_1)$，同样可以得到

$$\mu_{\theta}(\mathbf x_1, 1)=\frac 1 {\sqrt {\alpha_1}} (\mathbf x_1-\frac {\beta_1}{\sqrt {1-\overline \alpha_1}} \epsilon_{\theta}(\mathbf x_1, 1))$$

与 (15) 式保持一致，即 (15) 式的适用范围是 $1 \le t \le T$ 。

采样时，根据 $\mathcal N(\mathbf x_0; \mu_{\theta}(\mathbf x_1,1), \sigma_1^2I)$ 分布进行采样，这是最后一步采样，所以不增加噪声，即采样过程为

$$\mathbf x_0=\mu_{\theta}(\mathbf x_1, 1)=\frac 1 {\sqrt {\alpha_1}} (\mathbf x_1-\frac {\beta_1}{\sqrt {1-\overline \alpha_1}} \epsilon_{\theta}(\mathbf x_1, 1))$$

还需要对 $\mathbf x_0$ 进行离散化处理，

$$(-\infty, -\frac {254} {255}) \rightarrow 0
\\\\
[-\frac {254} {255}, -\frac {252} {255} ) \rightarrow 1
\\\\
\vdots
\\\\
[\frac {252} {255}, \frac {254} {255} ) \rightarrow 254
\\\\
[\frac {254} {255}, \infty) \rightarrow 255$$

这种离散化处理使得离散概率遵循 (18) 式 。

这里有个小技巧。直接查找 $x_0^i$ 值落于哪个区间效率太低，可以使用另一种等价高效的离散化方法，根据 (17) 式的逆过程得

$$\mathbf x_0 := \frac {255} 2 (\mathbf x_0 + 1)$$

经过上式变换后，像素值 $x$ 对应的区间为 $[x-\frac 1 2, x+\frac 1 2)$，$x=0, 1, \ldots 255$

然后做以下变换

$$\mathbf x_0:= \lfloor \mathbf x_0+\frac 1 2 \rfloor$$

这样时间复杂度就从 $O(n)$ 降为 $O(1)$ 。最后需要注意对 $[0,255]$ 区间之外的值进行截断，

$$\mathbf x_0^i:=\begin{cases} 0 & x_0^i < 0 \\\\ 255 & x_0^i > 255 \\\\ x_0^i & \text{o.w.}\end{cases}$$

## 1.3 算法总结

---
**算法 1** 训练

**repeat**

&emsp; $\mathbf x_0 \sim q(\mathbf x_0)$

&emsp; $t \sim \text{Uniform}(\{1,\ldots, T\})$

&emsp; $\epsilon \sim \mathcal N(\mathbf 0, I)$

&emsp; 求梯度
    $$\nabla_{\theta} \|\epsilon - \epsilon_{\theta}(\sqrt {\overline \alpha_t} \mathbf x_0+\sqrt {1-\overline \alpha_t} \epsilon, t)\|^2$$

**until converged**

---

---
**算法 2** 采样

$\mathbf x_T \sim \mathcal N(\mathbf 0, I)$

**for** $t=T,\ldots, 1$

&emsp; $\mathbf z \sim \mathcal N(\mathbf 0, I)$ if $t > 0$ else $\mathbf z = \mathbf 0$

$\mathbf x_{t-1}=\frac 1 {\sqrt \alpha_t} \left(\mathbf x_t - \frac {1-\alpha_t}{\sqrt {1-\overline \alpha_t}} \epsilon_{\theta}(\mathbf x_t, t) \right) + \sigma_t \mathbf z$

**end for**

**return** $\mathbf x_0$

---

# 2. 参考

<div id="refer-anchor-1"></div>

- [1] [diffusion-models-for-machine-learning-introduction](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction)

<div id="refer-anchor-2"></div>

- [2] [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)