---
title: FlashAttention
date: 2025-01-17 09:26:24
tags: transformer
mathjax: true
summary: 通过切片减少 HBM 访问次数，提高 attention 的处理速度
---

论文：[FlashAttention: Fast and Memory-Efficient Exact Attention
with IO-Awareness](https://arxiv.org/abs/2205.14135)

GPU 内存层级包括不同 size 和访问速度的内存，更小的内存其速度更快。例如 A100 GPU 有 40-80 GB 的 HBM，带宽为 1.5-2.0TB/s，以及 108 个流处理器，每个流处理器上有 192KB 的 SRAM，其带宽约为 19TB/s。

GPU 执行计算过程：GPU 有大量的线程来进行一个 kernel 的计算。每个 kernel 从 HBM 上加载数据到寄存器（register）和 SRAM 上，经过计算，将结果写回 HBM。

性能特征：操作可以分为 compute-bound 和 memory-bound 两种。

1. compute-bound: 操作时间主要由算术计算量决定，而 HBM 的访问时间则非常小。例如：inner dimension 非常大的矩阵相乘，大 channel 的卷积

2. memory-bound: 操作时间由内存访问量决定，而计算时间则非常小。例如，elementwise 操作（activation，dropout），reduction 操作（sum，softmax，batch norm，layer norm）。

__kernel 融合__：加快 memory-bound 操作的方法通常为 kernel fusion，例如对同一个 input 有多个操作，那么 input 仅需从 HBM 上加载一次，而不需要为每个操作分别加载。注意，在训练阶段，有一些中间值需要写回到 HBM，这些中间值用于反向传播。


# 1. 标准 Attention

给定输入 $Q,K,V \in \mathbb R^{N \times d}$，attention 输出 $O \in \mathbb R^{N \times d}$ 计算如下

$$S=QK^{\top} \in \mathbb R^{N \times N}, \quad P = \text{softmax}(S) \in \mathbb R^{N \times N}, \quad O =PV \in \mathbb R^{N \times d}$$

其中 softmax 按行进行。

标准 attention 实现中，矩阵 S 和 P 写到 HBM，需要 $O(N^2)$ 的内存，通常 $N \gg d$（例如 GPT-2 中，$N=1024, d=64$）。算法 0 中描述了标准 attention 的实现过程，其中大多数操作为 memory-bound（例如 softmax），大量的内存访问导致较慢的墙钟时间。

其他一些有关 attention 矩阵的 elementwise 操作则进一步恶化了这个问题，例如对 $S$ 使用 mask，或者对 $P$ 进行 dropout。

---
算法 0：标准 Attention 实现

---
输入：$Q,K,V \in  \mathbb R^{N \times d}$，位于 HBM

1. 从 HBM 上按块（block）加载 $Q,K$，计算 $S=QK^{\top}$，将 $S$ 写回 HBM
2. 从 HBM 上读取 $S$，计算 $P=\text{softmax}(S)$，将 $P$ 写回 HBM
3. 从 HBM 上按块加载 $P,V$，计算 $O=PV$，将 $O$ 写回 HBM
---


# 2. FlashAttention

本文给出如何使用较少的 HBM 读写计算 attention，并且不将中间值存储到 HBM，得到一个 memory 高效的 attention 算法。

## 2.1 高效 Attention

将 $Q,K,V \in \mathbb R^{N \times d}$ 划分为 blocks，按 block 从 HBM 加载到 SRAM，计算这些 blocks 的 attention 输出。

__Tiling:__ 按块计算 attention，由于 softmax 需要用到 $K$ 的所有列，所以将大的 softmax 分解。

首先，为了数值计算稳定，向量 $x \in \mathbb R^B$ 的 softmax 计算如下（这里 $B$ 可能对应 block），

$m(x)=\max_i x_i$

$f(x)=[e^{x_1-m(x)}, \ldots, e^{x_B-m(x)}]$

$l(x)=\sum _ i f(x)_i$

$\text{softmax}(x)=\frac {f(x)}{l(x)}$

然后给定两个向量 $x^{(1)}, x^{(2)} \in \mathbb R ^ B$，我们将 $x=[x^{(1)}\quad x^{(2)}] \in \mathbb R^{2B}$ 的 softmax 操作分解为，

$m(x)=m([x^{(1)} \ x^{(2)}])=\max(m(x^{(1)}, x^{(2)}))$

$f(x)=[e^{m(x^{(1)})-m(x)}f(x^{(1)}) \quad e^{m(x^{(2)})-m(x)}f(x^{(2)})]$

$l(x)=e^{m(x^{(1)})-m(x)}l(x^{(1)}) + e^{m(x^{(2)})-m(x)}l(x^{(2)})$

$\text{softmax}(x)=\frac {f(x)}{l(x)}$

根据上面这个分解过程，我们只要存储一些统计量（$m(x), l(x)$），我们就可以一次计算出一个 block 的 softmax。于是，将 $Q,K,V$ 划分为 blocks，计算出 softmax 值并保存统计量，那么在反向传播时，我们可以直接根据统计量计算出相关中间值。

__Recomputation:__ 我们的目标是不存储反向传播所用的数量为 $O(N^2)$ 的中间值 $S, P \in \mathbb R^{N\times N}$，仅存储统计量 $m,l$，就可以再次计算出 $S, P$。这可看作是一种选择性梯度检查点（selective gradient checkpointing）。梯度检查点可以降低最大内存要求，以速度换内存空间，空间复杂度降低。虽然 FLOPs 更大，但是 HBM 访问减少，反向传播速度反而变快。

__kernel 融合：__ tiling 使得算法实现可以在一个 CUDA kernel 中完成，从 HBM 中加载数据，执行计算步骤（矩阵乘法，softmax，masking和dropout，矩阵乘法），然后将结果写回 HBM，这避免了反复从 HBM 读取数据和将数据写回 HBM。

---
算法 1： FlashAttention

---
输入： $Q,K,V\in \mathbb R^{N\times d}$ 位于 HBM，on-chip SRAM 的 size $M$

1. 设置 block size $B_c=\lceil \frac M {4d} \rceil, \ B_r =\min(B_c,d)$
2. 在 HBM 中初始化 $O=(0)_{N\times d} \in \mathbb R ^{N \times d}, \ l=(0)_N \in \mathbb R^N, \ m=(-\infty)_N \in \mathbb R^N$
3. 将 $Q$ 划分为 $T_r=\lceil N/B_r \rceil$ 个 blocks $Q_1,\ldots, Q_{T_r} \in \mathbb R^{B_r \times d}$，将 $K,V$ 划分为 $T_c=\lceil N/B_c \rceil$ 个 blocks $K_1,\ldots, K_{T_c}, V_1,\ldots, V_{T_c} \in \mathbb R^{B_c \times d}$
4. 将 $O$ 划分为 $T_r$ 个 blocks $O_1,\ldots, O_{T_r} \in \mathbb R^{B_r \times d}$， 将 $l,m$ 分别划分为 $T_r$ 个 blocks $l_1,\ldots, l_{T_r}, m_1,\ldots, m_{T_r} \in \mathbb R^{B_r}$

5. __for__ $1 \le j \le T_c$ __do__

    - 从 HBM 上加载 $K_j, V_j$ 到 SRAM

    - __for__ $1 \le i \le T_r$ __do__

        - 从 HBM 加载 $Q_i, O_i, l_i, m_i$ 到 SRAM
        - 计算 $S_{ij}=Q_i K_j^{\top} \in \mathbb R^{B_r \times B_c}$
        - 计算 $\tilde m _{ij}=\text{rowmax}(S_{ij}) \in \mathbb R^{B_r}, \ \tilde P_{ij}=\exp(S_{ij}-\tilde m_{ij})\in \mathbb R^{B_r \times B_c}, \ \tilde l_{ij}=\text{rowsum}(\tilde P_{ij}) \in \mathbb R^{B_r}$
        - 计算 $m_i^{new}=\max(m_i, \tilde m_{ij})\in \mathbb R ^{B_r}, \ l_i^{new}=e^{m_i-m_i^{new}}l_i + e^{\tilde m_{ij}-m_i^{new}} \tilde l_{ij} \in \mathbb R^{B_r}$
        - 计算 $O_i\leftarrow \text{diag}(l_i^{new})^{-1}(\text{diag}(l_i) e^{m_i-m_i^{new}}O_i + e^{\tilde m_{ij} - m_i^{new}} \tilde P_{ij} V_j)$，然后将 $O_i$ 写回 HBM
        - 将 $l_i \leftarrow l_i^{new}, \ m_i \leftarrow m_i^{new}$ 写回 HBM

        __end for__
    
    __end for__

6. __Return__ $O$
---

为什么 $B_c, B_r$ 值如此设置？

为了最大化利用 SRAM 的内存 size，减少划分的 blocks 数量，从而减少 HBM 的访问次数。假设 $M$ 是 $4d$ 的整数倍，即 $\lceil \frac M {4d} \rceil=\frac M {4d}$，那么

外层循环中，每次加载 $K_j,V_j$，其 size 均为 $B_c \times d = \frac M 4$，那么 SRAM 剩余内存 size 为 $M-2\times \frac M 4=\frac M 2$

内存训练中，每次加载的 $Q_i, O_i$ size 均为 $B_r \times d$，例如 A100，$M=192*1024$，当 $d=64$（多头注意力中的一个 head 的维度）时，$Q_i, O_i$ size 均为 $d^2=\frac M {48}$，$S_{ij}$ 的 size 均为 $B_r \times B_c = d \lceil \frac M {4d} \rceil=\frac M 4$，计算 $\tilde m_{ij}$ 这需要 $B_r=d$ 的内存 size，然后计算 $\tilde P_{ij}$，可以对 $S_{ij}$ 采用 in-place 计算，即 $\tilde P_{ij}$ 与 $S_{ij}$ 共用 $\frac M 4$ 的内存 size。可见总的内存使用在不超过 $M$ 的情况下尽量接近 $M$ 。

内存循环是行分块，因为没办法一次性加载全部行到 SRAM。

外层训练是列分块，每次将新的块 concatenate，得到更长的向量，然后重新按行计算 softmax，以及更新 attention 输出。

__定理 1：__ 算法 1 计算 $O=\text{softmax}(QK^{\top})V$，FLOPs 为 $O(N^2d)$，除了输入输出还需要额外的内存为 $O(N)$。

FLOPs 与标准 attention 相同，额外的内存使用其实就是 $m_i, l_i \in \mathbb R^{B_r}$ 等，总共 $T_r$ 个，而 $T_r \times B_r=N$。

## 2.2 IO 复杂度

__定理 2__ 记 $N$ 为序列长度，$d$ 为注意力 head 维度，SRAM size 满足 $d \le M \le Nd$。标准 attention（算法 0）需要 $\Theta(Nd+N^2)$ HBM 访问量，而 FlashAttention 仅需 $\Theta(N^2d^2M^{-1})$ HBM 访问量，注意 IO 复杂度是 IO 读写的数据量而非次数。

对于标准 attention 实现：

1. 从 HBM 上分别加载 $Q$ 和 $K$，IO 复杂度为 $\Theta(Nd)$，计算 $S$ 矩阵并写回 HBM，这一步 IO 复杂度为 $\Theta(N^2)$
2. 从 HBM 加载 $S$，计算 $P$ 并写回 HBM，这一步 IO 复杂度为 $\Theta(N^2)$
3. 从 HBM 加载 $P$ 和 $V$，IO 复杂度为 $\Theta(N^2+Nd)$，计算 $O$ 写回 HBM 的 IO 复杂度为 $\Theta(Nd)$

FlashAttention 实现：

1. 从 HBM 上加载 $K,V$ 的 IO 复杂度为 $\Theta(Nd)$
2. 从 HBM 上加载 $Q,O$ 的 IO 复杂度为 $\Theta(NdT_c)=\Theta(Nd \cdot \frac N {M/4d})=\Theta(M^2d^2M^{-1})$
3. 将 $O$ 写回 HBM 的 IO 复杂度为 $\Theta(Nd T_c)$

