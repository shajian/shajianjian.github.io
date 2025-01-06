---
title: Deep Spatial Gradient and Temporal Depth Learning for Face Anti-spoofing
date: 2024-05-18 16:53:06
tags: face anti-spoofing
mathjax: true
---

# 1. 简介

人脸识别会面临人脸欺骗（spoofing）攻击，包括：

1. 打印攻击（将人脸打印到纸上）
2. 回放攻击（将包含人脸的视频显示在电子屏上）
3. 3D 面具攻击

也成为展示攻击（presentation attack, PA）。

已有的基于深度学习的方法将人脸反欺骗（anti-spoofing）看作一个二值分类问题，即活体人脸和欺骗人脸两个分类，记为 living 和 spoofing。但是之前的这些方法面临着难以挖掘 spoofing 中的模式属性，例如缺失一些皮肤细节，颜色失真，moire 条纹等。

为了解决这个问题，一些方法使用深度监督进行辅助。例如， living 人脸包含深度信息，而 spoofing 人脸照片则没有深度信息，所以利用深度监督训练能提高 PAD（展示攻击检测）的准确性。

根据调研发现存在两个问题：

1. 传统方法设计局部描述子来解决 PAD，而深度学习方法则通过学习提取相对高级语义特征。尽管这些特征有效，但是低级特征包含更多的细节，也是有用的，例如，空间梯度幅值，如图 1 所示，由于 spoofing 人脸图片更加光滑（细节更少），所以空间梯度幅值更小。

    ![](/images/face/sgtd_1.jpg)

2. 近期深度监督方法根据单帧图像估计人脸深度，并直接将深度进行密集像素级别的监督。作者认为，living 和 spoofing 之间的实质性的区别信息，可以从多帧图像中挖掘出更多。

    图 2 是一个明显的例子，

    ![](/images/face/sgtd_1.jpg)
    
    图 2. 三个 frame 表示从做到右的微小移动。(a) living，鼻子与右耳的夹角随着移动变小 $\alpha > \beta_2$，而左耳与鼻子的夹角随着移动变大$\beta_1 < \gamma$；(b) spoofing 则相反，$\alpha' < \beta_2'$，$\beta_1' >\gamma'$。


为了解决这两个问题，作者提出新的深度监督方法：spatio-temporal network，此网络具有
1. Residual Spatial Gradient Block(RSGB)
2. Spatio-Temporal Propagation Module(STPM)

# 2. 方法

## 2.1 网络结构

网络输入为 $N_f$ 帧人脸图像，端到端监督训练，网络输出为 depth map，如图 3 所示，

![](/images/face/sgtd_3.jpg)

骨干网络由级联的多个 RSGB 和池化层组成，提取 low-level、mid-level 和 high-level 三组特征，然后 concate 这三组特征，最后得到每帧的粗 depth map 的预测。

为了捕获更加丰富的动态信息，在帧间加入 STPM，其中，短期空时模块（STSTB）用于提取相邻帧之间的空时特征，ConvGRU 将这些短期特征转换为多帧长期特征，得到时序 depth maps，这些时序 depth maps 用于对 backbone 输出的粗 depth maps 进行精细化调整。

### 2.1.1 RSGB

在区分 living 和 spoofing 时，细粒度的空间信息至关重要，如图 1，living 和 spoofing 人脸图像的梯度幅值不同，我们设计 RSGB 来捕获这一具有判别性的特征。使用 sobel 算子计算梯度幅值，水平梯度和竖直梯度计算如下，

$$F_h(x)=\begin{bmatrix}-1 & 0 & +1\\\\-2 & 0 & +2 \\\\ -1 & 0 & +1\end{bmatrix} \odot x,\quad F _ v(x)=\begin{bmatrix}-1 & -2 & -1 \\\\ 0 & 0 & 0 \\\\ +1 & +2 & +1\end{bmatrix}\odot x \tag{1}$$

其中 $\odot$ 表示按深度卷积，即每个 channel 分别独立进行卷积。

使用残差结构聚合卷积特征和梯度幅值信息，如图 4 所示，梯度幅值需要经过归一化，

![](/images/face/sgtd_4.jpg)

使用下式表示这一聚合过程，

$$y=\phi(\mathcal N(F(x,\{W_i\}) + \mathcal N(F_h(x')^2 + F _ v(x')^2 ))) \tag{2}$$

其中 $x$ 表示卷积层的输入，$x'$ 表示 $x$ 经过一个 `1x1` conv 的输出，$\mathcal N, \ \phi$ 分别表示归一化和 ReLU。

### 2.1.2 STPM

living 和 spoofing 人脸图像的 depth 之间的区别可以从多帧图像中提取，故设计 STPM 从多帧中提取空时特征用于 depth 估计。STPM 包含 STSTB 和 ConvGRU。

**STSTB**

STSTB 通过融合以下五种特征来提取通用短期空时信息：

1. 当前压缩过的特征 $F_l(t)$
2. 当前空间梯度特征 $F_l^S(t)$
3. 未来空间梯度特征 $F_l^S(t+\Delta t)$
4. 时间梯度特征 $F_l^T(t)$
5. 上一个 STSTB 特征 $STSTB_{l-1}(t)$

本文中，空间梯度使用 Sobel 算子按深度卷积，时间梯度通过对时序特征按元素相减计算得到。使用 `1x1` 卷积压缩 channel 提高计算效率。

**ConvGRU**

两个相邻帧的 STSTB 的输出为 short term information 的表示能力有限，可以使用循环神经网络捕获 long range 空时上下文信息。LSTM 和 GRU 忽略了网络中隐藏单元的空间信息，故使用 ConvGRU 来传递 long range 空时信息。ConvGRU 表示如下，

$$\begin{aligned}R_t &= \sigma(K_r \otimes [H_{t-1}, X_t], \ U_t =\sigma(K_u \otimes [H_{t-1}, X_t])
\\\\ \hat H_t &=\tanh (K _ {\hat h} \otimes [R_t * H_{t-1}, X_t])
\\\\ H_t &=(1-U_t)*H_{t-1} + U_t * \hat H_t
\end{aligned} \tag{3}$$

其中 $X_t, H_t, U_t, R_t$ 分别表示输入、输出、更新门和重置门的矩阵，$K_r, K_u, K_{\hat h}$ 为卷积层的 kernels，$\otimes$ 表示卷积操作，$*$ 表示按元素相乘，$\sigma$ 为 sigmoid 激活函数。

### 2.1.3 改善 Depth Map

基于 RSGB 的 backbone 和 STPM，可以获得对应的粗 depth maps $D_{single}^t$ 和时序 depth maps $D _{multi}^t$，其中 $t \in [1, N_f -1]$ 表示帧的序号。使用下式对粗 depth maps 进行改善，

$$D_{refined}^t = (1-\alpha)\cdot D_{single}^t + \alpha \cdot D_{multi}^t , \quad \alpha \in [0,1] \tag{4}$$

## 2.2 损失函数

### 2.2.1 对比深度损失

在经典的基于 depth 的人脸反欺骗方法中，经常使用欧氏距离损失（EDL）来进行 pixel 级别的监督，如下

$$L_{EDL}=||D_G - D_G||_2^2 \tag{5}$$

其中 $D_P, \ D_G$ 分别为预测的 depth map 和 gt depth map。EDL 监督了像素级的 depth，但是忽略了相邻像素之间的 depth 之差。

本文提出 Contrastive Depth Loss（CDL）以进行更强的监督，如图 5，

$$L_{CDL}=\sum _ i ||K_i ^{CDL} \odot D_P - K_i ^{CDL}\odot D_G||_2^2 \tag{6}$$

其中 $K_i^{CDL}$ 是第 i 个 对比卷积核，$i \in [0, 7]$，图 5 显示了这些卷积核的细节。

![](/images/face/sgtd_4.jpg)

### 2.2.2 总损失

得到预测的 depth map，还要知道如何判断 living 和 spoofing 二值中的哪个类别，我们考虑如下二值损失和总损失，

$$L_{binary}=-B_G * \log(fcs(D_{avg})) \tag{7}$$

$$L_{overall}=\beta \cdot L_{binary} + (1-\beta) \cdot (L_{EDL}+L_{CDL}) \tag{8}$$

其中 $B_G$ 是二值 gt label，$D_{avg}$ 是 $\{D_{refined}^t\} _ {t=1}^{N_f-1}$ 的均值池化，$fcs$ 表示两个全连接层和一个 softmax 层，其输出表示两个分类的 logits，根据这个输出结果来预测是二分类中的哪个分类。

# 3. 双模态反欺骗数据集

作者收集了一个真实的双模态数据集（RGB 和 depth），有三种回放攻击的显示材料：AMOLED 屏、OLED 屏和 IPS/TFT 屏，有三种打印攻击的纸质材料：高质量 A4 纸，涂料纸和海报纸。使用 RealSense SR300 作为相机，可以提供 RGB 和 Depth 信息。包含 300 个主题，总共有 2700 个样本（4~12秒的视频）。此数据集简称为 DMAD。

> 涂料纸：涂上一层涂料(coating color)，使纸张具有良好的光学性质及印刷性能等。其主要用途有：印刷杂志、书籍等出版用纸和商标、包装、商品目录等印刷用纸。

# 4. 实验

## 4.1 数据集和指标

**数据集**

5 个数据集：OULU-NPU，SiW，CASIA-MFSD，Replay-Attack，DMAD

OULU-NPU 是一个高分辨率数据集，包含 4950 个视频。SiW 包含更多的 live 主题。CASIA-MFSD 和 Replay-Attack 包含低分辨率视频。

**性能指标**

在 OULU-NPU 和 SiW 数据集中，作者遵循对应的原始协议和指标，以便公平比较。

OULU-NPU，SiW 和 DMAD 使用 Attack Presentation 分类错误率 APCER 和 Bona Fide Presentation 分类错误率 BPCER，前者计算所有 Presentation Attack 中的最高错误率，后者计算真实 access 视频样本的错误率，最后计算两种错误率的平均，

$$ACER=\frac {APCER+BPCER} 2 \tag{9}$$

CASIA-MFSD 和 Replay-Attack 两个数据集使用 HTER，计算错误拒绝率（FRR）和错误接受率（FAR）的平均，

$$HTER=\frac {FRR+FAR}2 \tag{10}$$

## 4.2 实现细节

使用密集人脸对齐方法 PRNet 估计 living 人脸的 3D 形状，然后生成人脸 depth map $D_G \in \mathbb R^{32 \times 32}$ 。一个典型的例子见图 6，

![](/images/face/sgtd_6.jpg)

为了区分 living 和 spoofing，在训练阶段，将 living depth maps 进行归一化到 $[0, 1]$，而 spoofing depth map 全部置 0。


**训练策略**

使用一个二阶段策略：

1. 使用 $L_{EDL}, \ L _ {CDL}$ 训练级联 RSGB 构成的 backbone，以便学到基本的表征，从而预测出粗 depth maps

2. 固定 backbone 的参数，使用 $L_{overall}$ 训练 STPM，以便能 refine depth maps

网络输入为 $N_f$ 个帧，从视频中间隔 3 帧采样得到，这个采样间隔使得样本帧在有限的 GPU 内存中维持足够的时序信息。

**测试策略**

为了得到最终的分类得分，将序列帧喂给模型获得 depth maps $\{D_{refined} ^ t\} _ {t=1}^{N_f - 1}$，$fcs(D_{avg})$ 中的 living 分类的 logit 记为 $\hat b$，最终的 living 分类得分为

$$score = \beta \cdot \hat b + (1-\beta) \cdot \frac {\sum _ {t=1} ^{N_f-1} ||D_{refined}^t * M^t||_1}{N_f - 1} \tag{11}$$

其中 $\beta$ 与 (8) 式中的相同，$M^t$ 是帧 t 的 mask，可以通过密集人脸地标 PRNet 生成。

**超参设置**

学习率对单帧部分（backbone）的训练为 `1e-4`，对多帧部分（STMP）的训练为 `1e-2`。batch size 对单帧部分为 48，对多帧部分为 2。

$N_f=5$ 。

使用 Adadelta 优化器，其中参数 $\rho=0.95$，$\epsilon=1e-8$。

设置平衡因子 $\alpha=0.6, \ \beta=0.8$ 。

# 5. 其他

## 5.1 OULU-NPU 数据集介绍

参考论文：《OULU-NPU: A mobile face presentation attack database with real-world variations》

下载链接： https://sites.google.com/site/oulunpudatabase/

Oulu-NPU人脸活体检测数据库由4950个真实和攻击视频组成。这些视频是用6台移动设备（Samsung Galaxy S6 edge, HTC Desire EYE, MEIZU X5, ASUS Zenfone Selfie, Sony XPERIA C5 Ultra Dual 和 OPPO N3）的前置摄像头录制的。共有三种不同的光照条件和背景场景 (Session 1, Session 2 and Session 3)。为了模拟真实的移动认证场景，视频长度被限制为5秒，要求受试者像被认证一样握住移动设备，但在正常使用设备时不要偏离自然姿势太多。在Oulu-NPU数据库中考虑的呈现攻击类型是打印和视频重放。这些攻击是使用两台打印机(Printer 1 和 Printer 2)和两台显示设备 (Display 1 and Display 2)创建的

训练集：360个真实人脸视频，1440个攻击人脸视频，20个主题。

验证集：270个真实人脸视频，1080个攻击人脸视频，15个主题。

测试集：360个真实人脸视频，1440个攻击人脸视频，20个主题。

同时提供了眼部位置信息。（不细写了）

视频命名：手机型号_场景_使用人_文件.avi（Phone_Session_User_File.avi）

手机型号：1-6

场景：1-3

使用人：1-55（其中1-20是用来训练的，21-35是用来验证development的，36到55是用来测试的。

文件：1-5（1：真实人脸；2：打印人脸1；3：打印人脸2；4：视频攻击1；5：视频攻击2）

协议：
协议1：不同场景
协议2：不同的攻击方式
协议3：不同手机
协议4：场景+攻击方式+手机

其中P是打印攻击，D是视频重放

![](/images/face/sgtd_7.jpg)
<center>图 7. 表中可能一部分数据错误，对照图 8</center>

![](/images/face/sgtd_8.jpg)
<center>图 8. </center>

对于每个协议，文件都会在’train.txt’、'Dev.txt’和’Test.txt’中存放训练、验证、测试建议方法的视频文件列表。这些文件的组织如下：
+1, filename_1
-1, filename_2
…
+1, filename_i
其中+1代表真实人脸，-1代表攻击。


（在与评估集对应的列表中，-1表示对应的视频文件是打印攻击，-2表示对应的视频文件是重放攻击。 ）
（请注意，在协议3和协议4中，使用了更换摄像头的场景。因此，有六个训练和开发子集（Train_i.txt、Dev_i.txt、Test.txt:i=1…6）。这些子集将用于训练、验证和测试六种不同的模型。 ）

以上关于 OULU-NPU 数据集的介绍来自 [博文](
https://blog.csdn.net/m0_45682738/article/details/123373370)。