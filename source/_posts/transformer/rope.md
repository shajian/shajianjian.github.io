---
title: 旋转位置编码RoPE
date: 2024-12-26 15:28:58
tags: transformer
mathjax: true
---

论文：[Roformer: Enhanced Transformer With Rotray Position Embedding](https://arxiv.org/abs/2104.09864)

[huggingface](https://huggingface.co/docs/transformers/model_doc/roformer)

[源码](https://github.com/ZhuiyiTechnology/roformer)




# 1. 背景

记 N 个单词的句子为 $\{w_i\}_{i=1}^N$，单词 $w_i$ 的词向量为 $\mathbf x_i \in \mathbb R^d$，考虑位置信息后，自注意力机制使用三个线性变换，得到 

$$\mathbf q_m = f_q(\mathbf x_m, m), \mathbf k _n =f_k(\mathbf x_n, n), \mathbf v_n = f_v(\mathbf x_n, n) \tag{1}$$

其中 $m, n \in [1,N]$ 表示位置，然后注意力为

$$a_{m,n}=\frac {\exp(\mathbf q_m^{\top} \mathbf k_n/\sqrt d)}{\sum _{j=1}^N \exp(\mathbf q_m^{\top} \mathbf k_j/\sqrt d)} \tag{2}$$

最后输出为

$$\mathbf o _m = \sum _{n=1}^N a_{m,n} \mathbf v_n$$

## 1.1 绝对位置编码

如何将位置信息考虑进去呢，一种方法是对每个位置编码一个位置向量 $\mathbf p$，然后加到词向量 $\mathbf x$ 上，

$$f_{t}(\mathbf x_i,i)=W _ t (\mathbf x _ i + \mathbf p_i) , \ t \in \{q,k,v\}\tag{3}$$

其中 $i \in [1,N]$ 表示位置。

### 1.1.1 基于正弦和余弦的PE

PE 维度为 $d$，计算位置 $i$ 处的 PE 如下，

$$\begin{cases}\mathbf p_{i, 2t}=\sin \left(\frac i {10000^{2t/d}}\right) \\\\ \mathbf p_{i,2t+1}=\cos \left(\frac i {10000^{2t/d}}\right)
\end{cases} \tag{4}$$

其中 $i \in [1,N], \ t \in [0, d/2)$

所有位置的 PE 维度为 $(N, d)$

```python
d = 512                             # model dimension，也是 PE 维度
N = 100                             # seq_len <= N
position = torch.arange(0, L).unsqueeze(1)  # (N, 1)
# e^{2t*-log(10000)/d}=(e^{-log10000})^{2t/d}=(1/10000)^{2t/d}
den = torch.exp(torch.arange(0, d, 2) * -(math.log(1e4) / d))#(d/2,)
a = position * den
pe = torch.empty(N, d)
pe[:,::2] = torch.sin(a)       # interleave
PE[:,1:d:2] = torch.cos(a)     # interleave

# get the position embedding for current mini-batch
pos = torch.arange(0, seq_len)  # L 是最大句子长度，seq_len 是当前句子长度
pe = PE[pos]    # (seq_len, d)
pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)
```

此方法虽然是基于绝对位置，但是也能一定程度的捕捉相对位置信息。自注意力机制能够利用这些位置编码之间的差异来推断 token 之间的相对距离。

我们记 $k_t=1/10000^{2t/d}$，$i$ 位置处 PE 记为

$$\mathbf p_i=[\sin(k_0i), \cos(k_0i), \ldots, \sin(k_{d/2-1}i), \cos(k_{d/2-1}i)]^{\top}$$

那么

$$\begin{aligned}P_i ^{\top}  P_{i+j} &=\sum _ {t=0}^{d/2-1} \sin(k_t i) \sin(k_t (i+j)) + \cos(k_t i) \cos(k_t(i+j))
\\\\ &= \sum _ {t=0}^{d/2-1} cos(k_t (i - (i+j)))
\\\\ &= \sum _ {t=0}^{d/2-1} cos(k_i j)
\end{aligned}$$

两个 PE 向量内积只跟相对距离 $j$ 有关，只是这种关系不是显式的，模型不容易直接学习到。

### 1.1.2 可学习PE

```python
pe = nn.Embedding(L, hidden_dim)
```

下面给出旋转位置编码（Rotary Position Embedding）。

## 1.2 相对位置编码

### 1.2.1 other work
一种相对 PE 为，

$$\begin{aligned}f_q(\mathbf x_m)&=W_q \mathbf x_m
\\\\ f_k(\mathbf x_n, n) &= W_k(\mathbf x_n+\tilde {\mathbf p} _ r ^ k)
\\\\ f_v(\mathbf x_n, n) &= W_v(\mathbf x_n + \tilde {\mathbf p} _ r ^ v)
\end{aligned} \tag{5}$$

其中 $\tilde {\mathbf p} _ r ^ k, \tilde {\mathbf p} _ r ^ v \in \mathbb R ^ d$ 是可学习相对位置编码，$r=\text{clip}(m-n, r_{min}, r_{max})$ 表示位置 $m, n$ 之间的相对距离（故可学习相对位置编码维度为 $(r_{max}-r_{min}, d)$，编者注）。

这种方法认为，仅在某个范围内需要准确的相对位置信息。

另一种相对 PE，则是根据 (3) 式，得到

$$\mathbf q_m ^{\top} \mathbf k_n = \mathbf x _ m ^{\top}W_q^{\top} W_k \mathbf x_n +\mathbf x _ m ^{\top}W_q^{\top} W_k \mathbf p_n + \mathbf p _ m ^{\top}W_q^{\top} W_k \mathbf x_n + \mathbf p _ m ^{\top}W_q^{\top} W_k \mathbf p_n \tag{6}$$

然后将 $\mathbf p_n$ 替换为基于正弦的PE $\tilde {\mathbf p} _{m-n}$，而第 3、4 项中的 $\mathbf p_m$ 替换为两个不同的可学习向量 $\mathbf u, \mathbf v$，这两个向量与 query 位置（即 $m$）无关。进一步地，将 content-based 向量 $\mathbf x_n$ 和 location-based 向量 $\mathbf p_n$ 关联的 $W_k$ 进行区分，分为 $W_k$ 和 $\tilde W_k$，于是 (6) 式变为

$$\mathbf q_m ^{\top} \mathbf k_n = \mathbf x _ m ^{\top}W_q^{\top} W_k \mathbf x_n +\mathbf x _ m ^{\top}W_q^{\top} \tilde W_k \tilde {\mathbf p} _{m-n} + \mathbf p _ m ^{\top}W_q^{\top} W_k \mathbf x_n + \mathbf p _ m ^{\top}W_q^{\top} \tilde W_k \tilde {\mathbf p} _{m-n} \tag{7}$$

最后将 value 项中的位置信息去掉，即 $f_v(\mathbf x_j)=W_v \mathbf x_j$。

还有一些相对位置编码，这里不再介绍了。

### 1.2.2 PoPE

根据 (1) 和 (2) 式，要计算注意力就要计算 $\mathbf q _m ^{\top} \mathbf k _ n = \langle f_q(\mathbf x_m, m), f_k (\mathbf x_n, n) \rangle$，使用一个新的函数 $g$ 表示

$$\langle f_q(\mathbf x_m, m), f_k (\mathbf x_n, n) \rangle=g(\mathbf x_m, \mathbf x_n, m-n) \tag{11}$$

其中 $g$ 参数为两个位置处的内容 $\mathbf x_m, \mathbf x_n$ 和相对位置信息 $m-n$，这样就显示地将位置信息用相对位置考虑进去，如果我们能找到一个合适的函数 $g$，并从中得到相对位置信息编码，那问题不就解决了嘛。

__2D 情况__

先考虑维度 $d=2$ 的情况。作者找到满足 (11) 式的如下一组解，

$$\begin{aligned}f_q(\mathbf x_m, m)&=(W_q \mathbf x_m) e^{im\theta}
\\\\ f_k(\mathbf x_k, n)&=(W_k \mathbf x_n) e^{in\theta}
\\\\ g(\mathbf x_m, \mathbf x_n, m-n) & = Re[(W_q \mathbf x_m)(W_k \mathbf x_n)^* e^{i(m-n)\theta}]
\end{aligned} \tag{12}$$

其中，$e^{im\theta}$ 表示将 2d 向量 $W_q \mathbf x_m$ 旋转 $m\theta$ 角度， Re 表示求复数的实部，$(W_k \mathbf x_n)^*$ 中的星号表示求共轭，即虚部符号取反。

由于 $e^{i\theta}$ 表示将向量旋转 $\theta$ 角度，例如著名的欧拉公式 $e^{i\pi}=-1$，表示复平面上 $(1, 0)$ 旋转 $\pi$ 角度后变成 $(-1, 0)$，将 $e^{i\theta}$ 表示为矩阵形式，

$$e^{i\theta}=\begin{bmatrix} \cos \theta & -\sin \theta \\\\ \sin \theta & \cos \theta\end{bmatrix}$$

容易验证 $e^{i\pi}=-1$ 的矩阵形式为

$$\begin{bmatrix} -1 & 0 \\\\ 0 & -1\end{bmatrix} \begin{bmatrix} 1 \\\\ 0\end{bmatrix}=\begin{bmatrix} -1 \\\\ 0\end{bmatrix}$$

于是，将 $f_{\{q,k\}}$ 写成矩阵形式为

$$f_{\{q,k\}}(\mathbf x _ m, m) = \begin{pmatrix} \cos m \theta & -\sin m \theta \\\\ \sin m\theta & \cos m\theta\end{pmatrix} \begin{pmatrix} W_{\{q,k\}}^{(11)} & W_{\{q,k\}}^{(12)} \\\\ W_{\{q,k\}}^{(21)} & W_{\{q,k\}}^{(22)} \end{pmatrix} \begin{pmatrix} x _ m ^{(1)} \\\\ x _ m ^{(2)} \end{pmatrix}$$

<details>

<summary>证明 (12) 式的解满足 (11) 式。</summary>

首先，知道以下两个知识点：

1. 令 $z_1=a+ib, z_2=c+id$，那么 $z_1 \cdot z_2^*=ac+bd + i(bc-ad)$ 那么 $\langle z_1, z_2 \rangle=ac+bd=Re[z_1 \cdot z_2^*]$

2. $(z e^{i\theta})^*=z^* e^{i(-\theta)}$（可以使用矩阵形式进行证明）

3. 令 $z_1=a+ib, z_2=c+id$，那么 $z_1 e^{i\theta} \cdot z_2 e^{i\phi}= z_1 z_2 e^{i(\theta+\phi)}$

这里证明一下第 3 点，然后再证满足 (11) 式。

$$z_1 \cdot z_2 =\begin{bmatrix} c & -d \\\\ d & c \end{bmatrix}\begin{bmatrix} a \\\\ b\end{bmatrix}=\begin{bmatrix} ac-bd \\\\ ad+bc\end{bmatrix}$$

$$z_1 e^{i\theta}=\begin{bmatrix}a \cos \theta -b\sin \theta \\\\ a\sin \theta + b\cos \theta \end{bmatrix}, \quad z_2 e^{i\phi}=\begin{bmatrix}c \cos \phi -d\sin \phi \\\\ c\sin \phi + d\cos \phi \end{bmatrix}$$

$$\begin{aligned} z_1 e^{i\theta} \cdot z_2 e^{i\phi} &=\begin{bmatrix}c \cos \phi -d\sin \phi & -c\sin \phi - d\cos \phi \\\\ c\sin \phi + d\cos \phi & c \cos \phi -d\sin \phi\end{bmatrix} \begin{bmatrix} \cos \theta & -\sin \theta \\\\ \sin \theta &\cos \theta\end{bmatrix}\begin{bmatrix}a \\\\ b\end{bmatrix}
\\\\ &= \begin{bmatrix}c \cos(\theta + \phi) -d \sin(\theta+\phi) & -c \sin (\theta+\phi) -d \cos (\theta+\phi) \\\\ c \sin(\theta+\phi) + d \cos(\theta + \phi) & c \cos(\theta+\phi) -d \sin(\theta+\phi)\end{bmatrix}\begin{bmatrix}a \\\\ b\end{bmatrix}
\\\\ &=\begin{bmatrix} \cos(\theta + \phi) & -\sin(\theta + \phi) \\\\ \sin(\theta + \phi) & \cos(\theta + \phi)\end{bmatrix}\begin{bmatrix}c & -d \\\\ d & c\end{bmatrix}\begin{bmatrix}a \\\\ b\end{bmatrix}
\\\\ &= z_1 z_2 e^{i(\theta + \phi)}
\end{aligned}\tag{13}$$

那么 
$$\begin{aligned}\langle f_q(\mathbf x_m, m), f_k (\mathbf x_n, n) \rangle &=Re[(W_q \mathbf x_m) e^{im\theta} \cdot ((W_k \mathbf x_n) e^{in\theta})^*]
\\\\ &= Re[(W_q \mathbf x_m) e^{im\theta} \cdot (W_k \mathbf x_n)^* e^{i(-n\theta)}]
\\\\ &= Re[(W_q \mathbf x_m) (W_k \mathbf x_n)^* e^{i(m-n)\theta}]
\end{aligned}$$

证毕。

</details>

根据 (12) 式中的解可知，要包含相对位置编码，只需要将仿射变换后的词向量旋转一个角度，这个角度与词的位置下标线性相关，这样计算出来的注意力权重就包含了内容词向量和相对位置编码向量。

__通用形式__

为了将 2D 泛化到更一般的维度 $\mathbf x_i \in \mathbb R^d$，这里 $d$ 为偶数，将 $d$ 维度空间划分为 $d/2$ 个子空间，于是

$$f_{\{q,k\}}(\mathbf x_m, m)=R _ {\Theta, m}^d W_{\{q,k\}} \mathbf x_m \tag{14}$$

其中

$$R_{\Theta, m}^d=\begin{pmatrix} 
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\\\ 
\sin m\theta_1 & \cos m \theta_1 & 0 & 0 & \cdots & 0 & 0 \\\\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0 \\\\
0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots & 0 & 0 \\\\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\\\
0 & 0 & 0 & 0 & \cdots &  \cos m\theta_{d/2} & -\sin m\theta_{d/2}\\\\
0 & 0& 0 & 0  & \cdots & \sin m\theta_{d/2} & \cos m\theta_{d/2}
\end{pmatrix}
$$

$\Theta = \{\theta_i = 10000 ^ {-2(i-1)/d}|i=1,2,\ldots, d/2\}$。

$R_{\Theta, m}^d$ 写成上面这种形式，那么 $\langle f_q(\mathbf x_m, m), f_k(\mathbf x_n, n)\rangle=\sum_{i=1}^{d/2} g(\mathbf x_m^{(i)}, \mathbf x_n^{(i)}, m-n)$，满足 __注意力仅与词向量和相对位置信息有关__ 这一需求。这里 $\mathbf x_m^{(i)}$ 表示 $\mathbf x_m$ 的第 i 个子空间对应的 2D 向量。

RoPE 的过程如图 1 所示，将 query/key 分成 $d/2$ 份，然后依次将每份 2D 子向量旋转 $m\theta_i$ 角度，最后 concatenate 得到新的 $d$ 维向量 position encoded query/key。

![]()

<center>图 1</center>

应用 RoPE 到自注意力，得到

$$\mathbf q _ m ^{\top} \mathbf k _ n = (R _ {\Theta, m}^d W_q \mathbf x_m)^{\top} (R _ {\Theta, n}^d W_k \mathbf x_n)=\mathbf x _ m ^{\top} W _ q R _ {\Theta, m-n} ^ d W _ k \mathbf x _ n \tag{16}$$

其中 $R _ {\Theta, m-n} ^ d=(R _ {\Theta, m}^d) ^{\top} R _ {\Theta, n} ^ d$。$R_{\Theta}^d$ 是正交矩阵，这确保了位置信息编码的稳定性。另外，$R_{\Theta}^d$ 是稀疏矩阵，直接使用会导致计算效率不高，作者提供了一个高效的实现方法如下，

$$R_{\Theta, m}^d \mathbf x=\begin{pmatrix} x_1 \\\\ x_2 \\\\ x_3 \\\\ x_4 \\\\ \vdots \\\\ x_{d-1} \\\\ x_d\end{pmatrix} \otimes \begin{pmatrix} \cos m \theta_1 \\\\ \cos m \theta_1 \\\\ \cos m\theta_2 \\\\ \cos m\theta_2 \\\\ \vdots \\\\ \cos m \theta_{d/2} \\\\ \cos m \theta_{d/2}\end{pmatrix} +
\begin{pmatrix} -x_2 \\\\ x_1 \\\\ -x_4 \\\\ x_3 \\\\ \vdots \\\\  -x_d \\\\ x_{d-1}\end{pmatrix} \otimes \begin{pmatrix} \sin m \theta_1 \\\\ \sin m \theta_1 \\\\ \sin m\theta_2 \\\\ \sin m\theta_2 \\\\ \vdots \\\\ \sin m \theta_{d/2} \\\\ \sin m \theta_{d/2}\end{pmatrix}\tag{17}
$$

其中 $\otimes$ 表示 element-wise 相乘。

相关代码如下，

```python
def sinusoidal_position_embedding(batch_size, nums_head, max_len, dim, device):
    pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1) # (L, 1)
    ids = torch.arange(0, dim // 2, dtype=torch.float)  # (d/2,)
    theta = torch.pow(10000, -2 * ids / dim)
    embeddings = pos * theta    # (L, d/2), 角度
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)    # (L, d/2, 2)
    # (batch_size, nums_head, L, d/2, 2)
    embeddings = embeddings.repeat((batch_size, nums_head, *([1]*len(embeddings.shape))))
    embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, dim))
    embeddings = embeddings.to(device)
    return embeddings

def RoPE(q, k):
    '''
    q,k: (batch_size, nums_head, max_len, dim)
    '''
    batch_size, nums_head, max_len, dim = q.shape
    # (batch_size, nums_head, max_len, dim) 最后一维度，正余弦交替
    pos_emb = sinusoidal_position_embedding(batch_size, nums_head, max_len, dim, q.device)
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)    # (17) 式中的正弦向量
    cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)   # (17) 式中的余弦向量

    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
    q2 = q2.reshape(q.shape)    # (17) 式中的第三列向量
    # (17) 式仅仅考虑位置 m 处，这里所有 max_len 个位置的 q 向量全部进行了 RePE
    q = q * cos_pos + q2 * sin_pos

    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    k = k * cos_pos + k2 * sin_pos
    return q, k

def attention(q, k, v, mask=None, dropout=None, use_RoPE=True):
    if use_RoPE:
        q, k = RoPE(q, k)
    d_k = k.size()[-1]  # dim

    # (max_len, dim) x (dim, max_len)
    att_logits = torch.matmul(q, k.transpose(-2, -1))
    att_logits /= math.sqrt(d_k)

    if mask is not None:
        att_logits = att_logits.masked_fill(mask == 0, -1e-9)

    att_scores = F.softmax(att_logits, dim=-1)
    if dropout is not None:
        att_scores = dropout(att_scores)
    return torch.matmul(att_scores, v), att_scores
```