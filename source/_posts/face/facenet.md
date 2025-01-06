---
title: FaceNet 论文解读
date: 2024-04-01 16:02:04
tags: face recognition
mathjax: true
---

论文：[FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)

源码：[davidsandberg/facenet](https://github.com/davidsandberg/facenet)

# 1. 简介
人脸验证（是否是同一人）
人脸识别（这是哪个人）

训练网络学习人脸的欧氏嵌入向量表示。使用 L2 距离训练网络：同一个人的人脸向量之间距离较小，不同人的人脸向量距离较大。

本文提出 FaceNet，使用 三元损失函数学习得到 128D 的向量表示。如图 1 所示，其中 L2 层用于归一化向量，

![](/images/face/facenet_1.png)
<center>图 1. </center>

通过三元损失的学习，让正例向量之间的距离更近，负例向量之间的距离更远。

# 2. 方法

## 2.1 triplet loss

人脸图像 $x$ 的向量表示为 $f(x) \in \mathbb R ^ d$，我们限制向量范数为 1，即 $||f(x)|| _ 2 = 1$ 。

一个人的人脸 anchor $x _ i ^ a$，应该接近这个人的其他人脸图像 $x _ i ^ p$（正例），远离其他人的人脸图像 $x _ i ^ n$（负例），

$$||f(x _ i ^ a) - f(x _ i ^ p)|| _ 2 ^ 2 + \alpha < ||f(x _ i ^ a) - f(x _ i ^ n)|| _ 2 ^ 2 \tag{1}$$

其中 $\alpha$ 是一个 margin，$\alpha$ 越大，区分性能越好，根据 (1) 式可得三元损失为

$$L=\sum _ i ^ N \left[||f(x _ i ^ a) - f(x _ i ^ p)|| _ 2 ^ 2 - ||f(x _ i ^ a) - f(x _ i ^ n) || _ 2 ^ 2+ \alpha \right] _ + \tag{2}$$

给定一个数据集，可以很容易构建这样的三元组（anchor，正例，负例），但是其中大量的三元组都满足 (1) 式，也就是三元损失 $L=0$，这样的三元组样本对训练没有贡献，我们需要挖掘困难样本。

## 2.2 三元组选择

给定 $x _ i ^ a$，选择难正例 $x _ i ^ p$ 满足 $\arg \max _ {x _ i ^ p} ||f(x _ i ^ a) - f(x _ i ^ p)|| _ 2 ^ 2$，类似地，选择难负例满足 $\arg \min _ {x _ i ^ n} ||f(x _ i ^ a) - f(x _ i ^ n)|| _ 2 ^ 2$ 。

这里遍历整个数据集， $x _ i ^ a$ 为当前人脸图像，$x _ i ^ p, x _ i ^ n$ 为数据集中除 $x _ i ^ a$ 之外的图像，分别求极大值和极小值得到难正例和难负例。

在整个训练集中求 argmax 和 argmin 实现起来计算量太大，训练太慢，另外，由于一些错误标注以及较差质量的图片的存在，导致在整个训练集中求 argmax 和 argmin 必然总是会得到这些错误标注和较差质量的图片，有两种方法可以解决这个问题：

1. 离线生成三元组。使用最近训练好的网络模型在数据的一个子集上求 argmax 和 argmin
2. 在线生成三元组。minibatch 前向传播后，得到相应的向量表示，然后在 minibatch 中求 argmin 和 argmax

本文作者使用在线生成策略，使用一个较大 size 的 minibatch，这个 size 大约是几千。minibatch 的 size 不能太小，否则求得的 argmin 在整个训练集中还是较大的，同样 argmax 在整个训练集中还是较小的，导致生成的并非真正的难正例和难负例。

实验中，minibatch 中每个人大约有 40 个人脸图片，每个人脸 ID 选择 45 个图片，一共 1800 个图片，没有选择困难正例，使用了所有的 anchor-positive pairs（$40C_{45}^2$ 个正例 pair？），但是选择了困难负例。作者发现选择所有的 anchor-positive pairs 比选择困难正例，在训练开始解读更加的稳定并且收敛稍快。

选择难负例在实际训练过程中会导致训练初期收敛到错误的局部最小值点，特殊情况下会导致进入一个坍缩模式即 $f(x)=0$，所有的向量均为 0 。为了解决这个问题，选择负例满足

$$||f(x _ i ^ a) - f(x _ i ^ p)|| _ 2 ^ 2 < ||f(x _ i ^ a) - f(x _ i ^ n)|| _ 2 ^ 2 \tag{3}$$

我们称满足 (3) 式的为半难负例 (semi-hard)：负例比正例更加远离 anchor，但是仍在 margin $\alpha$ 之内，即负例距离靠近正例距离，而前面的三元损失是要让负例位于 $\alpha$ 之外。


## 2.3 网络

初始学习率设置为 $0.05$，margin $\alpha=0.2$ 。

使用的第一种网络结构如图 2，第二种网络则是Inception 模型。

![](/images/face/facenet_2.png)
<center>图 2 Zeiler&Fergus 模型，使用 ReLU 作为非线性层</center>

# 3. 数据集

**# CASIA-WebFace**

CASIA-WebFace 包含 494414 张图片，来自 10575 个人。

CASIA-WebFace 数据集文件夹结构：

1. CASIA-WebFace：根目录
2. 000001-000200：每个身份对应一个子目录，子目录名为身份编号
3. 000001_0.jpg：每个图像文件的命名规则为 身份编号_索引.jpg

**# LFW**

LFW 共有 13233 张人脸图像，每张图像均给出对应的人名，共有 5749 人。图像尺寸均为 `250x250` 。

## 3.1 评估

给定两个人脸图片的 pair，距离 $D(x _ i, x _ j)$ 是否大于阈值决定是同一个人，还是不同的人。相同 id 的人脸图片 pairs $(i, j)$ 记为 $\mathcal P _ {same}$，不同 id 的所有人脸图片 pairs 记为 $\mathcal P _ {diff}$ 。

定义 true accept 集合为

$$TA(d) = \lbrace (i,j) \in \mathcal P _ {same}| D(x _ i, x _ j) \le d \rbrace \tag{4}$$

(4) 式表示来自相同人脸图像 pair 集合的 pair (i, j)，经过模型计算其特征向量距离 $D(x_i, x_j) < d$，表示预测这个 pair 是同一个人。

类似于 true positive，(4) 式表示正确的分类为 same 的 pairs。类似地，定义 false accept 为

$$FA(d)=\lbrace (i,j) \in \mathcal P _ {diff} | D(x _ i, x _ j) \le d \rbrace \tag{5}$$

这表示错误分类为 same 的 pairs，类似于 false positive。(5) 是表示来自不同人 pair 集合的一个 pair (i, j)，经过模型计算其特征向量距离 $D(x_i, x_j) < d$，表示预测这个 pair 是同一个人。

验证率 VAL 和 false accept rate FAR 定义为

$$VAL(d) = \frac {|TA(d)|}{|\mathcal P _ {same}|}, \quad FAR(d) = \frac {|FA(d)|}{|\mathcal P _ {diff}|} \tag{7}$$

(7) 式表明，VAL 越大，模型越好；FAR 越小，模型越好。

# 4. 源码分析

## 4.1 train softmax
facenet 源码中提供了 `train_softmax.py`，也就是将模型看作是一个分类器进行训练，没有使用论文中的 triplet loss，而是使用 softmax loss，这是因为训练一个分类器比较简单且速度较快。

**# face alignment**

在进行人脸识别之前，需要先做人脸对齐，使用 MTCNN 进行人脸对齐，项目中也提供了 MTCNN 的源码，使用如下命令，

```sh
python src/align/align_dataset_mtcnn.py \
~/datasets/casia/CASIA-maxpy-clean/ \
~/datasets/casia/casia-maxpy_mtcnn_182  \
--image_size 182    \
--margin    44
```

但是我看了源码，发现并没有进行人脸对齐，所做的工作包括：

1. 检测人脸 box
2. 在 box 外围留 margin/2 的边距，然后 crop
3. 将 crop 得到的图像 resize 到 (182, 182)

以上处理完毕之后，将每个 crop+resize 到 (182, 182) 的人脸图像按照原先的人的身份编号保存到文件，每个身份编号作为一个子目录，图像文件保存在这个子目录下，并将标注数据写入到标注文件中，标注数据格式为，

```sh
crop+resize图像文件名   在原图上 crop box 的 x1y1x2y2 坐标
```

人脸对齐之后，我们再看 `train_softmax.py` 训练代码：

加载数据集，就是获取上述保存的 crop+resize 的 (182, 182) 人脸图像文件路径，每个身份编号对应一个路径列表。

人脸图像经过：1. 旋转；2. 随机 crop 一个 (160, 160) 的 patch；3. 左右翻转；4. 归一化 等的预处理。

以 inception_resnet_v1 网络为例，输入 size 为 (n, 160, 160, 3)，经过卷积网络下采样后，再 flatten，然后使用 fc 层，输出 size 为 (n, 128)，得到人脸图像的特征表示（128维的向量）。



```python
# train_softmax.py
with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)
    # image_list: 所有图片的文件路径
    # label_list: 所有图片对应的身份 id（从 0 开始），一个人名对应一个 id
    image_list, label_list = facenet.get_image_paths_and_labels(train_set)
    val_image_list, val_label_list = facenet.get_image_paths_and_labels(val_set)

    labels = ops.convert_to_tensor(label_list, dtype=tf.int32)  # (N, )
    range_size = array_ops.shape(labels)[0] # N，数据集大小
    ...
    # embedding_size: 输出人脸向量的维度
    # prelogits: 非归一化人脸向量表征   (B, 128)
    prelogits, _ = network.inference(image_batch, args.keep_probability,
        phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
        weight_decay=args.weight_decay)
    # 根据特征向量，使用一个 fc 层，得到分类预测得分，非归一化预测得分, (B, C)
    logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None,
        weights_initializer=slim.initializers.xavier_initializer(),
        weights_regularizer=slim.l2_regularizer(args.weight_decay),
        scope='Logits', reuse=False)
    # 归一化的人脸向量
    embedings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
    # 人脸特征向量的L1 范数均值
    prelogits_norm = tf.reduce_mean(
        tf.norm(tf.abs(prelogits) + eps, ord=args.prelogits_norm_p, axis=1)
    )
    # 添加第一种正则损失，L1 范数惩罚项，为 L1 范围 * 正则化损失因子
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * args.prelogits_norm_loss_factor)
    # 计算第二种正则损失，中心损失，特征与所在类别特征中心的差的平方，再求均值
    prelogits_center_loss, _ = facenet.center_loss(prelogits, label_batch, args.center_loss_alfa, nrof_classes)
    # 添加中心损失
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.center_loss_factor)

    # 计算分类的交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_batch, logits=logits, name='cross_entropy_per_example'
    )
    # 交叉熵损失
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # 交叉熵损失和正则损失之和
    total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')
```

上述代码中，中心损失参考 "A Discriminative Feature Learning Approach for Deep Face Recognition"，中心损失（center loss），用以有效增强神经网络中深度学习特征的判别能力。

- 为每个类别的深度特征学习一个中心（一个与特征维度相同的向量）。在训练过程中，我们同时更新中心并最小化深度特征与其对应类别中心之间的距离。
- CNNs 在 softmax 损失和中心损失的联合监督下进行训练，有一个超参数来平衡这两个监督信号。直观地说，softmax 损失强制不同类别的深度特征保持分离。中心损失有效地将同一类别的深度特征拉向它们的中心。
- 在联合监督下，不仅类间特征差异被放大，而且类内特征变化也被减少。因此，深度学习特征的判别能力可以得到高度增强。

下面给出计算中心损失的代码，

```python
nrof_features = features.get_shape()[1]     # 特征维度  D=128
# 每个类别的特征中心    （C, D)，初始化为 0，后面每次则使用 diff 进行更新
centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
    initializer=tf.constant_initializer(0), trainable=False)
# 类别 (B, )
label = tf.reshape(label, [-1])
# 样本的类别特征中心，  (B, D)
centers_batch = tf.gather(centers, label)
# 样本类型对应的特征中心与样本特征之差，(c-x) * 0.05
diff = (1 - alfa) * (centers_batch - features)
# 更新特征中心：c=c-(c-x)*0.05=0.95c+0.05x
centers = tf.scatter_sub(centers, label, diff)  # (1-alpha)*mean(f_i)，batch 中，类内特征的均值乘以 1-alpha
with tf.control_dependencies([centers]):
    loss = tf.reduce_mean(tf.square(features - centers_batch))  # center loss， 差的平方
```

更新特征中心的公式为

$$\mathbf c := \mathbf c - (\mathbf c - \mathbf x) * 0.05=0.95*\mathbf c + 0.05 \mathbf x$$

可见这是采用了移动平均。中心损失为特征与特征中心的 MSE。

使用 `train_softmax.py` 训练，不需要准备 三元组 训练数据，更多训练细节，参见 [Classifier training of inception resnet v1](https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1)。下面介绍另一种训练方法。



## 4.2 train triple loss

训练入口为 `train_tripleloss.py` 文件

```python
# train_tripleloss.py
with tf.Graph().as_default():
    # 创建 batch
    ...
    # 人脸向量 (n, D)
    prelogits, _ = network.inference(image_batch, args.keep_probability,
        phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
        weight_decay=args.weight_decay)
    # L2 归一化人脸向量
    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
    # 将人脸向量分成 anchor，positive，negative，然后计算 tripleloss
    # 输入是 anchor positive 和 negative 三个部分的 stack
    # 故输出向量直接 unstack 即可
    anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1, 3, args.embedding_size]), 3, 1)
    triple_loss = facenet.triple_loss(anchor, positive, negative, args.alpha)

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([triple_loss] + regularization_losses, name='total_loss')

    train_op = facenet.train(total_loss, global_step, args.optimizer, learning_rate,
        args.moving_average_decay, tf.global_variable())
```

其中计算梯度和更新模型参数的代码为，

```python
# facenet.py
def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay,
    update_gradient_vars, log_histograms=True):
    loss_averages_op = _add_loss_summaries(total_loss)  # 计算移动平均
    with tf.control_dependecies([loss_averages_op]):
        if optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
        ...
    
        grads = opt.compute_gradients(total_loss, update_gradient_vars)#计算梯度
    # 更新梯度
    applay_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    # 创建移动平均方法
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step
    )
    # 对训练参数进行 移动平均
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op
```

上面代码中，移动平均公式为

```
shadow_variable = decay * shadow_variable + (1 - decay) * variable
```

衰减值为

```
decay = min(decay, (1 + steps) / (10 + steps))
```

最后看下输入数据如何构造，

```python
# train_tripletloss.py
# train 方法

# 随机选择至少 45 个人，每个人随机选择最多 40 个图片，总图片数量为 45 * 40 = 1800
# image_paths: 图片路径list，共 1800 个
# num_per_class: 每个人使用的图片数量
image_paths, num_per_class = sample_people(dataset, args.people_per_batch, args.images_per_person)
nrof_examples = args.people_per_batch * args.images_per_person  # 1800
labels_array = np.reshape(np.arange(nrof_examples), (-1, 3))    # 三元组的 label
image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))   # 三元组
...
emb_array = np.zeros((nrof_examples, embedding_size))   # 保存 1800 个人脸向量

# 1800 个图片分成 (600, 3)，
nrof_batches = int(np.ceil(nrof_examples / args.batch_size))    # batch 数量
for i in range(nrof_batches):
    batch_size = min(nrof_examples - i*args.batch_size, args.batch_size)
    # 前向传播。emb: 特征向量(n,D), lab: gt label (n,)
    emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size, learning_rate_placeholder: lr, phase_train_placeholder: True})
    # 将前向传播计算的 人脸向量 更新到数组中
    emb_array[lab, :] = emb

# select triplets based on the embeddings
triplets, nrof_random_negs, nrof_triplets = select_triplets(
    emb_array, num_per_class, image_paths, args.people_per_batch, args.alpha
)
```

上面代码中，首先随机选择 `45 * 40` 个图片进行前向传播，计算出这些人脸向量之后，使用 `select_triplets` 选择三元组，

```python
def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    '''
    embeddings: (1800, 128) 1800 个 人脸向量。1800 个图片包含至少 45 个人
    nrof_images_per_class: 每个人的图片数量，这是一个 list
    image_paths: 1800 个 图片路径
    people_per_batch: 人数 45
    '''
    emb_start_idx = 0
    num_trips = 0
    triplets = []   # 三元组列表

    for i in range(people_per_batch):   # 遍历每个人
        nrof_images = int(nrof_images_per_class[i]) # 第 i 个人的图片数量
        for j in range(1, nrof_images): # 遍历这个人的所有图片
            a_idx = emb_start_idx + j - 1   # anchor 在 1800 图片中的 idx
            # 在 1800 个向量中寻找 argmin 困难负例
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in range(j, nrof_images):  # 所有正例，不使用困难正例
                p_idx = emb_start_idx + pair    # 正例 idx
                # 计算正例与 anchor 的距离平均
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
                # 屏蔽当前 人 对应的图片，这些图片不是负例，这个语句放在这里，其实是重复执行了
                neg_dists_sqr[emb_start_idx:emb_start_idx+nrof_images] = np.NaN
                # 选择负例，不满足以下条件的负例，实际上 tripletloss 为 0，对学习没有贡献
                all_neg = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs > 0:    # 存在这样的负例
                    rnd_idx = np.random.randint(nrof_random_negs)   # 随机选择一个
                    n_idx = all_neg[rnd_idx]    # 随机选择的负例 idx
                    triples.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                num_trips += 1 # 实际上是二元组 anchor positive 数量
        emb_start_idx += nrof_images
    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)
```

以上代码中，对于第 `i` 个人，假设其有 `n` 个图片，那么可得 (anchor, positive) 二元组数量为组合数 $C _ n ^ 2$，根据二元组来构造三元组，即，为每个二元组再构造一个负例。实际上对第 `i` 个人，有 `1800-n` 个负例，但是我们筛选不满足 (1) 式的负例，因为满足 (1) 式的负例损失为 0，对学习没有贡献，最后从筛选的负例中随机采样一个负例，构造出三元组。

