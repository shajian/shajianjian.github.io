---
title: Auto-Encoding Variational Bayes
date: 2022-05-20 17:04:14
tags: 
    - generative model
    - vae
mathjax: true
summary: 变分编码器
---

论文：[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

# 1. VAE 基本概念

编码器解码器的基本结构如图 1，

![](/images/generative_model/vae_1_1.png)

<center>图 1. 编码器和解码器结构图。图中红色表示隐变量空间</center>

编码器用于将输入进行编码，编码后的向量保留原输入（input）的特征，称为 隐变量（code），解码器则以这个隐变量作为输入，解码得到 output，这个 output 尽量能还原原输入 input。故 VAE 通常可作为一种压缩方式。

> 两个相似的原输入，编码后在隐空间中的向量也应该靠近，否则应该互相远离。

## 1.1 编/解码器

编码器网络使用下式表达

$$Z = g(\theta X+b)$$

其中 $X$ 是原输入，$\theta, \ b$ 分别为网络权重和偏置，$Z$ 为编码即隐向量。

解码器的数学表达式为

$$X'=g'(\theta' Z + b')$$

使用 MSE 作为损失函数，

$$\mathcal L=||X-X'||^2=||X-g'[\theta' (g(\theta X+b))+b]||^2$$

设计网络时需要注意平衡编码器的编码能力和解码器的解码能力。

上面这个损失函数（向量之差的范数）设计的不够好，所以实际上生成器生成的图像不理想。

# 2. 概率视角下的 VAE

## 2.1 问题抛出

考虑数据集 $X=\\{\mathbf x ^ {(i)}\\}_{i=1} ^ N$，其中数据 $\mathbf x$ 是离散型或连续型，数据由某个随机过程生成，且涉及不可观察的连续型隐变量 $\mathbf z$。数据 $\mathbf x$ 生成过程分为两步：

1. 从某个先验 $p_{\theta}(\mathbf z)$ 生成 $\mathbf z$
2. 从条件分布 $p_{\theta}(\mathbf x|\mathbf z)$ 生成 $\mathbf x$

三个经典问题：

1. 对参数 $\theta$ 高效地进行 generative_model 或者 MAP 估计
2. 选择了参数 $\theta$ 后，给定观察值 $\mathbf x$，高效地对隐变量 $\mathbf z$ 求后验分布
3. 高效地求 $\mathbf x$ 的边缘分布。通过这种方式，可以实现图像降噪，超分辨率和图像修复等任务。

为了解决以上三个问题，作者介绍了一个识别模型 $q _ {\phi}(\mathbf z|\mathbf x)$，作为后验分布 $p _ {\theta}(\mathbf z|\mathbf x)$ 的近似，根据贝叶斯定理，由于边缘分布含有积分，后验分布无法计算，故使用近似的思想来解决，学习识别模型参数 $\phi$ 和生成模型参数 $\theta$ 。

通常将 $q _ {\phi}(\mathbf z|\mathbf x)$ 称作编码器 encoder，将 $p _ {\theta}(\mathbf x|\mathbf z)$ 称作解码器 decoder。

如图 2，

![](/images/generative_model/vae_1_2.png)

图 2. 实线表示生成模型 $p _ {\theta}(\mathbf z) p_{\theta}(\mathbf x|\mathbf z)$，虚线表示变分近似 $q _ {\phi}(\mathbf z|\mathbf x)$，这个分布用于近似后验分布 $p_{\theta}(\mathbf z|\mathbf x)$，由于

$$p_{\theta}(\mathbf z|\mathbf x)=\frac {p _ {\theta}(\mathbf x, \mathbf z)}{\int  p_{\theta}(\mathbf x, \mathbf z) d \mathbf z}$$

分母中积分不可解析处理，故使用变分近似  $q_{\phi}(\mathbf z|\mathbf x)$ 。

我们的目的是根据一组观测值 $\mathbf X=\{\mathbf x _ i\} _ {i=1} ^ n$，求得后验分布 $p_{\theta}(\mathbf z|\mathbf x)$，然后根据这个后验分布，随机取 $\mathbf z$ 的一个值，就可以通过解码器生成一个新的观测值 $\mathbf x$，这在图像生成等任务有用。

假设 $\mathbf z$ 是一个 n 维高斯分布，那么 $\mathbf z$ 的分布由期望向量和协方差矩阵确定，为了简单期间，假设协方差是一个对角矩阵，那么一共有 2n 个参数，根据一组观测值，训练 encoder 和 decoder，encoder 的输出就是这 2n 个参数，decoder 的输出为 $\mathbf x$ 的一个取值。当训练完成后，我们可以直接根据这 2n 个参数，生成一个 $\mathbf z$，然后通过 decoder 得到一个新的 $\mathbf x$ 。

## 2.2 变分边界

数据集对数似然为 $\log p_{\theta}(X)=\sum_{i} \log p_{\theta}(\mathbf x^{(i)})$，其中

$$\log p_{\theta}(\mathbf x ^ {(i)})=D_{KL}(q_{\phi}(\mathbf z|\mathbf x ^ {(i)})|| p_{\theta}(\mathbf z|\mathbf x^{(i)})) + \mathcal L(\theta, \phi; \mathbf x ^ {(i)}) \tag{1}$$

(1) 式中，想要让似然函数最大，由于 $D_{KL}(\cdot || \cdot) \ge 0$，所以需要令 $\mathcal L(\theta, \phi; \mathbf x ^ {(i)})$ 尽可能地大，$\mathcal L(\theta, \phi; \mathbf x ^ {(i)})$ 称为变分下界。

**证明：**

前面提到，使用 $q_{\phi}(\mathbf z|\mathbf x)$ 来近似 $p_{\theta}(\mathbf z|\mathbf x)$，常使用 KL 散度来衡量两个分布的距离，

$$\begin{aligned} D_{KL}(q_{\phi}(\mathbf z|\mathbf x)||p_{\theta}(\mathbf z|\mathbf x))&=\mathbb E_{q_{\phi}(\mathbf z|\mathbf x)} [\log q_{\phi}(\mathbf z|\mathbf x)]-\mathbb E_{q_{\phi}(\mathbf z|\mathbf x)} [\log p_{\theta}(\mathbf z|\mathbf x)]
\\\\ &=\mathbb E_{q_{\phi}(\mathbf z|\mathbf x)} [\log q_{\phi}(\mathbf z|\mathbf x)]-\mathbb E_{q_{\phi}(\mathbf z|\mathbf x)} [\log p_{\theta}(\mathbf z,\mathbf x)] + \mathbb E_{q_{\phi}(\mathbf z|\mathbf x)} [\log p_{\theta}(\mathbf x)]
\\\\ &=\mathbb E_{q_{\phi}(\mathbf z|\mathbf x)} [\log q_{\phi}(\mathbf z|\mathbf x)]-\mathbb E_{q_{\phi}(\mathbf z|\mathbf x)} [\log p_{\theta}(\mathbf z,\mathbf x)] + \log p_{\theta}(\mathbf x)
\end{aligned}$$

故有

$$\log p_{\theta}(\mathbf x) = D_{KL}(q_{\phi}(\mathbf z|\mathbf x)||p_{\theta}(\mathbf z|\mathbf x)) + \mathcal L(\theta, \phi; \mathbf x) \tag{2}$$

$$ELBO=\mathcal L(\theta, \phi; \mathbf x)=-\mathbb E_{q_{\phi}(\mathbf z|\mathbf x)} [\log q_{\phi}(\mathbf z|\mathbf x)]+\mathbb E_{q_{\phi}(\mathbf z|\mathbf x)} [\log p_{\theta}(\mathbf z,\mathbf x)] \tag{3}$$

如果展开 $D_{KL}(q_{\phi}(\mathbf z|\mathbf x)||p_{\theta}(\mathbf z))$，

$$\begin{aligned}D_{KL}(q_{\phi}(\mathbf z|\mathbf x)||p_{\theta}(\mathbf z))&=\mathbb E_{q_{\phi}(\mathbf z|\mathbf x)} [\log q_{\phi}(\mathbf z|\mathbf x)]-\mathbb E_{q_{\phi}(\mathbf z|\mathbf x)} [\log p_{\theta}(\mathbf z)]
\\\\ &=-\mathcal L(\theta, \phi; \mathbf x)+\mathbb E_{q_{\phi}(\mathbf z|\mathbf x)} [\log p_{\theta}(\mathbf z,\mathbf x)]-\mathbb E_{q_{\phi}(\mathbf z|\mathbf x)} [\log p_{\theta}(\mathbf z)]
\\\\ &=-\mathcal L(\theta, \phi; \mathbf x)+\mathbb E_{q_{\phi}(\mathbf z|\mathbf x)} [\log p_{\theta}(\mathbf x|\mathbf z)]
\end{aligned}$$

那么可以得到

$$\mathcal L(\theta, \phi; \mathbf x)=-D_{KL}(q_{\phi}(\mathbf z|\mathbf x)||p_{\theta}(\mathbf z))+\mathbb E_{q_{\phi}(\mathbf z|\mathbf x)} [\log p_{\theta}(\mathbf x|\mathbf z)] \tag{3-1}$$

(3-1) 式和 (3) 式是 $ELBO$ 的两种表达形式。

继续研究 $\mathcal L$。

$$\begin{aligned}D_{KL}(q(x,z)||p(x,z))&=\mathbb E_{q(x,z)} \left[\log \frac {q(x,z)}{p(x,z)}\right]
\\\\ &=\mathbb E_{q(x)}\left[\mathbb E_{q(z|x)} \left[ \log \frac {1}{p(x|z)}+\log \frac {q(z|x)}{p(z)}+ \log q(x)\right]\right]
\\\\ &=\mathbb E_{q(x)}\left[-\mathbb E_{q(z|x)}[\log p(x|z)]+ D_{KL}(q(z|x)||p(z)) + \log  q(x)\right]
\\\\ &=\mathbb E_{q(x)} [-\mathcal L(\theta, \phi; x)+ \log q(x)]
\end{aligned}$$

上式中，$q(x)$ 是 $x$ 的经验分布，即模型的训练集分布，$q(z|x)$ 是推断模型的输出分布。

记 $\mathcal L(\theta,\phi)= \mathbb E_{q(x)} [\mathcal L(\theta,\phi; x)]$，那么上式变为

$$\mathcal L(\theta,\phi) = -D _ {KL}(q(x,z)||p(x,z)) + \mathbb E _ {q(x)} [\log q(x)] \tag{3-2}$$

(3-2) 式中，$\mathbb E _ {q(x)} [\log q(x)]$ 是基于训练集分布的未知常数，$-D_{KL}(q(x,z)||p(x,z)) \le 0$，所以训练目标为最大化 $\mathcal L(\theta,\phi)$，最优解为 $D_{KL}(q(x,z)||p(x,z))=0$，即 $q(x,z)$ 尽可能逼近 $p(x,z)$ 。

由于 $D _ {KL}(\cdot) \ge 0$，所以根据 (1) 式可知 $\log p _ {\theta}(\mathbf x) \ge \mathcal L(\theta, \phi; \mathbf x)$ ，即 **对数似然的下限是 $\mathcal L(\theta, \phi; \mathbf x)$** ，所以我们的 **目标是最大化 $\mathcal L(\theta, \phi; \mathbf x)$ ，从而达到最大化对数似然的目的** 。

如果能求得 $\mathcal L(\theta, \phi; \mathbf x)$ ，自然能计算出 $\mathcal L(\theta,\phi)$，方法是：对训练集中每个样本计算 $\mathcal L(\theta, \phi; \mathbf x)$，然后求平均，就是 $\mathbb E _ {q(x)}[\mathcal L(\theta, \phi; \mathbf x)]$ 。

但是求最大化目标 $\mathcal L(\theta, \phi; \mathbf x)$ 存在问题，因为表达式中含有期望计算 $\mathbb E_{q_{\phi}(\mathbf z|\mathbf x)}[\cdot]$（见 (3) 式），这使得导数计算不好处理。一种常见的解决方法是使用 Monte Carlo 梯度估计方法：


$$\mathcal F(\theta,\phi)=\mathbb E _ {q _ {\phi}(\mathbf z)}[f(\mathbf z;\theta,\phi)] \approx \frac 1 L \sum_{l=1} ^ L f(\mathbf z ^ {(l)};\theta,\phi) \tag{4}$$

其中 $\mathbf z ^ {(l)} \sim q _ {\phi}(\mathbf z|\mathbf x)$。

通常目标函数（是一个随机函数）的方差较大，故根据 (4) 求梯度（Monte Carlo 梯度估计）结果可能会不准确，故不实用。

## 2.3 SGVB 估计和 AEVB 算法

使用重参数技巧。如图 3 所示，

![](/images/generative_model/vae_1_3.png)

<center>图 3.</center>

$\mathbf z$ 是一个随机变量，参数 $\phi$ 包含在节点 $\mathbf z$ 内部，重参数化 $\mathbf z \sim q_{\phi}(\mathbf z|\mathbf x)$ 为

$$\mathbf z = g_{\phi}(\epsilon, \mathbf x) \tag{5}$$

其中 $\epsilon \sim p(\epsilon)$ 是一个随机噪声向量，$g_{\phi}(\cdot)$ 是某个参数为 $\phi$ 的向量函数。由于 (5) 式是确定性函数，可求导，那么

$$q_{\phi}(\mathbf z|\mathbf x) d \mathbf z=p(\epsilon) d\epsilon
\\\\ \int q_{\phi}(\mathbf z|\mathbf x) f(\mathbf z) d\mathbf z=\int p(\epsilon) f(\mathbf z) d\epsilon=\int p(\epsilon) f(g_{\phi}(\epsilon, \mathbf x)) d\epsilon$$

（注：为了简洁，$f(\mathbf z)$ 中未注明参数）。

那么 Monte Carlo 估计为

$$\mathbb E_{q_{\phi}(\mathbf z|\mathbf x)}[f(\mathbf z)]=\int q_{\phi}(\mathbf z|\mathbf x) f(\mathbf z) d\mathbf z \approx \frac 1 L \sum_{l=1}^L f(g_{\phi}(\mathbf x, \epsilon^{(l)})) \tag{6}$$

其中 $\epsilon^{(l)} \sim p(\epsilon)$ ，一共 L 次采样。

以一维高斯分布为例， $z \sim p(z|x)=\mathcal N(\mu, \sigma^2)$，那么一种重参数表达为 $z = \mu + \sigma \epsilon$，其中 $\epsilon \sim \mathcal N(0, 1)$，那么目标函数为

$$\mathbb E_{\mathcal N(z;\mu, \sigma^2)}[f(z)]=\mathbb E_{\mathcal N(\epsilon;0,1)}[f(\mu+\sigma \epsilon)] \approx \frac 1 L \sum_{l=1}^L f(\mu+\sigma \epsilon^{(l)})$$

使用这种重参数技巧，并采用 Monte Carlo 估计 ELBO （3）式，

$$\tilde {\mathcal L}(\theta, \phi;\mathbf x)=\frac 1 L \sum_{l=1}^L \log p_{\theta}(\mathbf x, \mathbf z^{(l)})-\log q_{\phi}(\mathbf z^{(l)}|\mathbf x) \tag{7}$$

其中 $\mathbf z ^ {(l)}=g_{\phi}(\epsilon ^ {(l)}, \mathbf x)$，$\epsilon ^ {(l)} \sim p(\epsilon)$ 。



---
算法 1 Auto-Encoding VB (AEVB)

输入：mini batch size $M=100$。Monte Carlo 近似的采样次数 $L=1$
输出：encoder/decoder 参数 $\phi, \theta$
初始化：$\phi, \theta$

**repeat**
&emsp; $X^M \leftarrow$ 从数据集中随机选择批数据，批大小为 $M$
&emsp; $\epsilon \leftarrow$ 从噪声分布 $p(\epsilon)$ 中随机采样，每个数据 $\mathbf x^{(i)}$ 使用独立的采样（不共享）
&emsp; $\mathbf g \leftarrow \nabla_{\theta, \phi} \mathcal L^M(\theta, \phi;X^M, \epsilon)$ ，计算梯度，使用 (7) 或 (7-1) 式
&emsp; 使用梯度和 learning rate 更新 $\theta, \phi$ （梯度上升）
**util $(\theta, \phi)$ 收敛**

___

我们也可以采用 (3-1) 式计算 $\mathcal L$，由于 $D_{KL}(q_{\phi}(\mathbf z|\mathbf x)||p_{\theta}(\mathbf z))$ 可以计算闭式解（close form）（见下方折叠内容），(3-1) 式中只有 $\mathbb E_{q_{\phi}(\mathbf z|\mathbf x)} [\log p_{\theta}(\mathbf x|\mathbf z)]$ 需要通过采样计算，前者可以理解为对 $\phi$ 的正则化，即近似后验 $q_{\phi}(\mathbf z|\mathbf x)$ 尽可能地逼近先验 $p_{\theta}(\mathbf z)$ 。这样就得到第二种目标函数的计算方式，

$$\tilde {\mathcal L}(\theta,\phi;\mathbf x)=-D_{KL}(q_{\phi}(\mathbf z|\mathbf x)||p_{\theta}(\mathbf z)) + \frac 1 L \sum_{l=1}^L \log p_{\theta}(\mathbf x|\mathbf z^{(l)}) \tag{7-1}$$

作者发现只要 minibatch size $M$ 足够大，采样数量 $L$ 可设置为 1 。

观察 (7-1) 式，KL 散度用作正则，另外一项则是负重建损失（似然）。

**计算 $D_{KL}(q_{\phi}(\mathbf z|\mathbf x)||p_{\theta}(\mathbf z))$**

<details>
<summary>点此展开</summary>

假设先验为 $p_{\theta}(\mathbf z)=\mathcal N(0,I)$，$\mathbf z$ 向量维度记为 $J$，后验近似 $q_{\phi}(\mathbf z|\mathbf x)$ 为高斯型。

为了表述简洁，直接使用 $q_{\phi}(\mathbf z)$ 表示 $q_{\phi}(\mathbf z|\mathbf x)$，由于是高斯型分布，记变分期望和变分标准差为 $\mu, \ \sigma$ （在数据点 $\mathbf x$ 的条件下），那么计算以下几个部分：

$$\begin{aligned}\int q_{\phi}(\mathbf z)\log p_{\theta}(\mathbf z) d\mathbf z&=\int q_{\phi}(\mathbf z) \log \mathcal N(\mathbf z; 0, I) d\mathbf z
\\\\ &=\int q_{\phi}(\mathbf z) (-\frac J 2 \log 2\pi - \frac 1 2 \mathbf z^{\top} \mathbf z) d\mathbf z
\\\\ &=-\frac J 2 \log 2\pi - \frac 1 2 \mathbb E_{q_{\phi}(\mathbf z)} \left[\sum_{j=1}^J z_j^2 \right]
\\\\ &=-\frac J 2 \log 2\pi - \frac 1 2 \sum_{j=1}^J (\mu_j^2 + \sigma_j^2)
\end{aligned}$$

最后一步推导用到了 $\mathbb E[z_j^2]=\mathbb V[z_j]+ \mathbb E^2[z_j]$ 。上式计算负交叉熵，显然两个高斯分布的交叉熵可以有闭式解，高斯分布的负熵也有闭式解，如下

$$\begin{aligned} \int q_{\phi}(\mathbf z) \log q_{\phi}(\mathbf z) d\mathbf z &= -H(q_{\phi}(\mathbf z))
\\\\ &=-\frac J 2 (\log 2\pi + 1)-\frac 1 2 \log |\Sigma|
\\\\ &=-\frac J 2 (\log 2\pi + 1)-\frac 1 2 \sum_{j=1}^J \log \sigma_j^2
\end{aligned}$$

第二个等式推导用到了[多维高斯分布的熵公式](/2022/06/29/math/gaussian)，最后一步是因为协方差矩阵是 $\Sigma=\text{diag} (\sigma_1^2, \ldots, \sigma_J^2)$ ，其行列式为对角线元素乘积。

综合上面两式得，

$$\begin{aligned}-D_{KL}(q_{\phi}(\mathbf z|\mathbf x)||p_{\theta}(\mathbf z))&=\int q_{\phi}(\mathbf z)[\log p_{\theta}(\mathbf z)-\log q_{\phi}(\mathbf x)] d\mathbf z
\\\\ &=\frac 1 2 \sum_{j=1}^J (1+ \log \sigma_j^2 - \mu_j^2 -\sigma_j^2)
\end{aligned}$$

</details>

<br/>

**# 重参数技巧**

什么样的 $q_{\phi}(\mathbf z|\mathbf x)$ 使得我们可以选择合适的转换函数 $g_{\phi}(\cdot)$ 和辅助随机变量 $\epsilon \sim p(\epsilon)$ ？

1. 易处理的 反 CDF 函数。

    令 $\epsilon \sim \mathcal U (0, I)$ （多元 0-1 均匀分布），根据 encoder 网络输出得到 $(\mu, \sigma^2)=encoder(\mathbf x)$。令 $g_{\phi}(\epsilon, \mathbf x)$ 为 $q_{\phi}(\mathbf z|\mathbf x)$ 的反 CDF 函数，即 
    
    $$\mathbf z= g_{\phi}(\epsilon, \mathbf x)= F^{-1}(\epsilon; \mu, \sigma^2)$$

    例如：指数分布，柯西分布，Logistic，Rayleigh，Pareto，Weibull，Reciprocal，Gompertz，Gumbel 和 Erlang 分布。

2. 类似高斯分布那样，对于 `location-scale` 族的分布，可以选择标准分布 `(location=0, scale=1)` 作为辅助分布 $p(\epsilon)$，然后 $\mathbf z= g_{\phi}(\cdot)=loc + scale \cdot \epsilon$

    例如：Laplace，Elliptical，学生 t 分布，Logistic，Uniform，Trangular 和高斯分布
3. 组合：可以使用辅助变量的变换来表示随机变量。例如： Log-Normal，Gamma，Dirichlet，Beta，F，卡方分布等。


# 3. VAE 例子

本节进行实例讲解。使用一个神经网络作为 encoder $q_{\phi}(\mathbf z|\mathbf x)$ （也是生成模型 $p_{\theta}(\mathbf x,\mathbf z)$ 的后验近似），其中模型参数 $\phi, \theta$ 由 AEVB （算法 1）优化。

## 3.1 网络模型输出作为分布的参数

令隐变量 $\mathbf z$ 的先验为各向同性的多维高斯分布 $p_{\theta}(\mathbf z)=\mathcal N(\mathbf z;0, I)$ ，此先验分布不包含参数。

似然分布 $p_{\theta}(\mathbf x|\mathbf z)$ 是一个多维高斯分布（连续型数据）或者 Bernoulli 分布（二分类数据），对应 decoder 模型，似然分布的参数则由 decoder 模型输出确定。decoder 模型采用 MLP 结构，

真实后验 $p_{\theta}(\mathbf z|\mathbf x)$ 使用高斯型分布，其协方差矩阵使用一个对角矩阵作为近似（这里这么假设是为了使问题简化），那么变分后验近似则为

$$q_{\phi}(\mathbf z|\mathbf x^{(i)})=\mathcal N(\mathbf z; \mu^{(i)}, \sigma^{2(i)}) \tag{8}$$


**模型参数 $\mu^{(i)}, \ \sigma^{(i)}$ 由 encoder 网络输出** 。 这两个参数均为 $J$ 维向量。这里特意标注了数据 index $i$，前面的数据 $\mathbf x$ 没有特意标出 index，但实际上相关的参数 $\mu, \sigma^2$ 也都是跟具体观测数据相关。

encoder 网络为 MLP，即输入为 $\mathbf x^{(i)}$ 的非线性函数，参数为 $\phi$ 。

### 3.1.1 Bernoulli MLP (decoder)

MLP 包含单个 hidden layer，输入为 $\mathbf z$，多元 Bernoulli 概率计算如下，

$$\log p(\mathbf x|\mathbf z)=\sum_{i=1}^D x_i [\log y_i + (1-x_i) \cdot \log (1-y_i)] \tag{9}$$

其中 $\mathbf y= f_{\sigma} (W_2 \tanh (W_1 \mathbf z+ \mathbf b_1)+\mathbf b_2)$，$\mathbf z \in \mathbb R^J, \ \mathbf x \in \mathbb R^D$ ，这是一个二层 layers 的网络 MLP，(9) 式中的 $\mathbf x$ 就是输入。

decoder 输出为 $p_{\theta}(\mathbf x|\mathbf z)$ 的分布参数，这里是多元 Bernoulli 分布，那么输出 $\mathbf y \in \mathbb R^D$ 表示 $\mathbf x=\mathbf 1$ 的概率向量。 

$f_{\sigma}(\cdot)$ 表示 sigmoid 激活函数，$(W_1,W_2, \mathbf b_1, \mathbf b_2)$ 为 MLP 的参数。

### 3.1.2 Gaussian MLP (encoder/decoder)

高斯 MLP 既可以做 encoder，也可以做 decoder，即我们的例子中隐变量 $\mathbf z$ 是连续型的，$\mathbf x$ 可以是连续型（高斯MLP）也可以是离散型（Bernoulli MLP）。

**# decoder**

我们假设了高斯分布的协方差矩阵是对角型，表示 decoder 时，MLP 网络表达为

$$\begin{aligned}\mathbf h &= \tanh (W_3 \mathbf z + \mathbf b_3)
\\\\ \mu &=W_4 \mathbf h +  \mathbf b_4
\\\\ \log \sigma^2 &= W_5 \mathbf h + \mathbf b_5
\end{aligned} \tag{10}$$

根据 (10) 式，可以计算出对数似然分布为 

$$\log p(\mathbf x|\mathbf z)= \log \mathcal N(\mathbf x;\mu, \sigma^2) \tag{10-1}$$

**# encoder**

当此 MLP 作为 encoder 时，对数变分分布为 $\log q_{\phi}(\mathbf z|\mathbf x)= \log \mathcal N(\mathbf z;\mu, \sigma^2)$ ，将 (10) 式中 $\mathbf z$ 改为 $\mathbf x$ 就得到此 encoder 的网络结构。

在我们的例子中，从变分分布 $\mathbf z^{(i,l)} \sim q_{\phi}(\mathbf z|\mathbf x ^ {(i)})$ 中采样，根据等式 $\mathbf z ^ {(i,l)}=g _ {\phi}(\mathbf x ^ {(i)}, \epsilon ^ {l})=\mu ^ {(i)} + \sigma ^ {(i)} \odot \epsilon ^ {(l)}$，其中 $\epsilon ^ {(l)} \sim \mathcal N(0,I)$，$\mu ^ {(i)}, \sigma ^ {(i)}$ 是 $encoder(\mathbf x ^ {(i)})$ 的输出。

$p _ {\theta}(\mathbf z)$ 和 $q _ {\phi}(\mathbf z|\mathbf x)$ 均为高斯分布，使用 (7-1) 式计算目标函数，其中 KL 散度可以计算出闭式解（close form），根据前面的 KL 散度的计算结果可知，

$$\mathcal L(\theta,\phi;\mathbf x)=\frac 1 2 \sum_{j=1} ^ J (1+ \log \sigma_j ^ 2 - \mu_j ^ 2 -\sigma_j ^ 2) + \frac 1 L \sum_{l=1} ^ L \log p_{\theta}(\mathbf x|\mathbf z ^ {(l)}) \tag{11}$$

其中 $\mathbf z ^ {(i,l)}=\mu ^ {(i)} + \sigma ^ {(i)} \odot \epsilon ^ {(l)}$，$\epsilon ^ {(l)} \sim \mathcal N(0,I)$。

计算 (11) 式时，$\epsilon ^ {(l)} \sim \mathcal N(0,I)$，$\mu ^ {(i)}, \sigma ^ {(i)}$ 是 $encoder(\mathbf x ^ {(i)})$ 的输出，(11) 式第一项容易计算，(11) 式第二项则可以根据 (9) 式或者 (10-1) 式计算。

例如使用 mnist 数据集训练，

|Layer|Output|Shape|所属|
|--|--|--|--|
|Input| x(归一化到 [0,1])| (batch_size, 784)|-|
|FC+relu|h(隐藏层输出)|(batch_size, hidden_dim)|encoder|
|FC|o(输出层)|(batch_size, 2*latent_dim)|encoder|
|Sampling|z(采样)|(batch_size, latent_dim)|-|
|FC+relu|h(隐藏层输出)|(batch_size, hidden_dim)|decoder|
|FC+sigmoid|u(解码，x 的期望)|(batch_size, 784)|decoder|

表 1.

说明：`hidden_dim` 可以取 256，`latent_dim` 可以取 2（或者其他更大的数），`2*latent_dim` 表示两个输出，分别对应 $q_{\phi}(\mathbf z|\mathbf x)$ 的期望和协方差对角线向量。

decoder 如果是高斯 MLP，那么 (11) 式中最后一项为（L=1）

$$\log p_{\theta}(\mathbf x|\mathbf z)=\log \mathcal N(\mathbf x;\mu, \sigma ^ 2)=-\sum_{i=1} ^ D \frac 1 {2\sigma_i ^ 2}(x_i - \mu_i) ^ 2+C \tag{12}$$

其中 $C$ 是与模型参数无关的常量，$D=784$（mnist 数据集）。

或者使用交叉熵作为损失函数，

$$-p\log q=-[x_i \log \mu_i + (1-x_i) \log (1-\mu_i)] \tag{13}$$

这里 $x_i, \mu_i \in [0,1]$ 。

对于 (13) 式，针对 $\mu_i$ 求导，

$$\frac {\partial l}{\partial \mu_i}=-x_i/\mu_i + (1-x_i)/(1-\mu_i)=0$$

解上式得最优解  $\mu_i^{\star}=x_i$ ，这与 (12) 式的最优解相同，所以 (12) 和 (13) 两式都可以用作损失项。

实际上，使用 (13) 式作为损失项，没有考虑 $p _ {\theta} (\mathbf x|\mathbf z)$ 的方差，那么通过 decoder 输出的就只是 $p _ {\theta} (\mathbf x|\mathbf z)$ 的期望，这个期望就是通过采样 $\mathbf z$ 解码得到最终生成的数据，见表 1。

代码（截取自 ref 2）

```python
# (13) 式交叉熵
xent_loss = K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=-1)
# (11) 式右端第一项的负值
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)
```


## 3.2 图像生成

采样随机变量 $\mathbf z \sim \mathcal N(0, I)$，$\mathbf z$ 的 shape 为 $(1,2)$，其中 1 表示只生成一个图像，2 就是 latent_dim 的值。然后按照 表 1 中 decoder 的 layer 变换，得到像素值位于 $(0,1)$ 内的图像，乘以 255 后取整，然后 clip 到 $[0,255]$。


# 4. 代码

给一个 pytorch 实现的 vae 代码。

```python
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # encoder
        self.fc1 = nn.Linear(784, 200)  
        self.fc2_mu = nn.Linear(200, 10)    # mu of z
        self.fc2_log_std = nn.Linear(200, 10)   # log std of z
        # decoder
        self.fc3 = nn.Linear(10, 200)
        self.fc4 = nn.Linear(200, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc2_mu(h1)
        log_std = self.fc2_log_std(h1)
        return mu, log_std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        recon = torch.sigmoid(self.fc4(h3)) # 重建
        return recon

    def reparametrize(self, mu, log_std):
        '''
        sample to get a value of z
        '''
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        z = mu + eps * std

    def forward(self, x):
        mu, log_std = self.encode(x)
        z = self.reparametrize(mu, log_std)
        recon = self.decode(z)
        return recon, mu, log_std

    def loss_function(self, recon, x, mu, log_std):
        # 对应 (11) 式的第二项
        recon_loss = F.mse_loss(recon, x, reduction='sum')
        # 对应 (11) 式的第一项
        kl_loss = -0.5 * (1 + 2 * log_std - mu.pow(2) - torch.exp(2 * log_std))
        kl_loss = torch.sum(kl_loss)
        loss = recon_loss + kl_loss
        return loss
```

# ref

https://sassafras13.github.io/VAE/

https://github.com/bojone/vae/blob/master/vae_keras.py

