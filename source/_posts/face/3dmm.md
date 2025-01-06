---
title: A Morphable Model For The Synthesis Of 3D Faces
date: 2024-04-25 17:13:19
tags: 3D reconstruction
mathjax: true
---

论文：A Morphable Model For The Synthesis Of 3D Faces

本文提出一种人脸 3D 建模的方法。

# 1. 数据集

通过激光扫描 200 个年轻人（100 男性和 100 女性）的头部，得到头部结构数据的柱面表示，得到人脸表面点的径向距离 $r(h, \phi)$，均匀采样得到 512 个角度 $\phi$ 和 512 个高度 $h$，另外同时记录了每个点的颜色 $R(h, \phi), \ G(h, \phi), \ B(h, \phi)$ 。所有人脸均无化妆、挂饰和面部毛发，戴上浴帽进行扫描，后面再通过程序去掉浴帽。 预处理包括一个竖直切割去掉耳朵后面的部分，以及一个水平切割去掉肩部以下的部分，然后进行空间归一化，使得脸部朝向标准方向以及位于标准位置，经过处理之后，一个人脸大约由 70000 个点表示，每个点包含形状和纹理（即颜色）两部分。

# 2. 可变形 3D 人脸模型

使用一个形状向量表示人脸的几何特征，

$$S=(X _ 1, Y _ 1, Z _ 1, \ldots, X _ n, Y _ n, Z _ n) ^ {\top}$$

使用一个纹理向量表示人脸的纹理特征，

$$T=(R _ 1, G _ 1, B _ 1, \ldots, R _ n, G _ n, B _ n) ^ {\top}$$

使用大小为 $m$ 的数据集来建立可变性人脸模型。新人脸的形状和纹理可表示为

$$S _ {mod} = \sum _ {i=1} ^ m a _ i S _ i, \quad T _ {mod} = \sum _ {i=1} ^ m b _ i T _ i$$

其中 $\sum _ {i=1} ^ m a _ i = \sum _ {i=1} ^ m b _ i = 1$ 。

这意味我们使用一个线性模型建模人脸。

但是 $a _ i, \ b _ i$ 显然不能随便取值，否则生成的人脸 $S _ {mod}$ 可能不合理，所以要根据已有的数据集来估计 $a _ i, \ b _ i$ 的分布。实际上这里并不是估计 $a _ i$ 的分布，因为 $S _ i, \ i=1,\ldots, 3n$ 之间不是相互正交的，可以使用 PCA，各个主成分之间是相互正交的。

根据数据集计算人脸形状和纹理的均值向量，$\overline S$ 和 $\overline T$，然后计算 $\Delta S _ i = S _ i - \overline S, \ \Delta T _ i = T _ i - \overline T$，再计算协方差 $C _ S, \ C _ T$，

$$C _ S = \frac 1 m \sum _ {i = 1} ^ m \Delta S _ i \Delta S _ i ^ {\top}
\\C _ T = \frac 1 m \sum _ {i = 1} ^ m \Delta T _ i \Delta T _ i ^ {\top}$$

$\Delta S _ i, \ \Delta T _ i$ 维度均为 $3n$，故 $C _ S, \ C _ T$ 维度为 $3n \times 3n$，$m=200$，可以认为 $3n > m$，故矩阵的秩 $\text{rank}(C _ S) \le m-1, \ \text{rank}(C _ T) \le m-1$。

于是

$$\Delta S _ {model} = \sum _ {i = 1} ^ {m-1} \alpha _ i s _ i$$

而 $\Delta S _ {model} = S _ {model} - \overline S$，于是

$$S _ {model} =\overline S + \sum _ {i = 1} ^ {m-1} \alpha _ i s _ i, \quad T _ {model} =\overline T + \sum _ {i = 1} ^ {m-1} \beta _ i t _ i \tag{1}$$

其中 $s _ i, \ t _ i$ 为特征向量，按对应的特征值降序排列。

特征值是降序排列的，所以相当于使用了前面 $m-1$ 个主成分。

我们使用 **多维高斯分布模型** 作为参数 $\alpha _ i, \ \beta _ i$ 的分布，拟合这 200 个人脸数据集，概率分布为

$$p (\vec {\alpha}) \sim \exp \left[-\frac 1 2 \sum _ {i=1} ^ {m-1} (\alpha _ i / \sigma _ i) ^ 2\right] \tag{2}$$

其中 $\sigma _ i ^ 2$ 为 $C _ S$ 的特征值。$\vec \beta$ 的分布类似处理。

这样人脸形状和纹理均有 $m-1$ 个自由度，修改每个参数就相当于对每个特征向量子空间独立的修改，一个子空间可能对应眼睛、鼻子或者其他周围区域，从而修改人脸。

## 2.1 面部属性

人脸可变模型的参数 $\alpha _ i, \ \beta _ i$ 并不对应人类语言所描述的面部属性，例如面部的女性气质或者胖瘦程度。

这里我们为样本手动设计标签，这些标签描述了面部属性，然后找到一个方法与人脸可变模型的参数联系起来。在人脸空间中，定义形状和纹理向量使得当增加到人脸或者从人脸中减去，其实就是对某个特别的属性进行操作而其他属性保持不变。

使用面部表情，那么形状和纹理的表情模型为 

$$\Delta S = S _ {expression} - S _ {neutral}, \ \Delta T = T _ {expression} - T _ {neutral} \tag{3}$$

其中 expression 下标表示有表情，neutral 下标表示无表情。

这样就得到数据集中每个人脸样本的 $\Delta S$ 和 $\Delta T$ 值，使用与上面相同的方法（PCA）建模，得到人脸表情的三维模型。

面部表情是统一的，即每个个体的面部表情没有什么差异，面部属性则不同，每个个体的差异很明显，下面介绍如何建模来表示性别、面部丰满程度、眉毛的黑度、是否双下巴以及是钩鼻还是凹鼻。

数据集 $(S _ i, T _ i)$，手动打标签 $\mu _ i$ 表示属性的显著程度，然后计算加权和，

$$\Delta S = \sum _ {i=1} ^ m \mu _ i (S _ i - \overline S), \quad \Delta T = \sum _ {i=1} ^ m \mu _ i(T _ i - \overline T) \tag{4}$$

然后对每个个体的人脸，加上或者减去 $(\Delta S, \Delta T)$ 的数倍，可以对人脸进行面部属性的修改。对于二进制属性，例如性别，即属性类别为 A 和 B，对数量为 $m _ A$ 的样本类别 A 赋值为常量 $\mu _ A$，对数量为 $m _ B$ 的样本类别 B 赋值为常量 $\mu _ B$。

但是面部属性很多，对每一种属性，均人工打 $m$ 个标签，这是很困难的。我们使用一个函数 $\mu(S, T)$ 表示人脸 $(S, T)$ 的属性显著程度，我们假设 $\mu (S, T)$ 是一个线性函数，这样最优解在以下条件达到最小时满足，

$$||\Delta S|| _ M ^ 2 = \langle \Delta S, C _ S ^ {-1} \Delta S \rangle, \quad ||\Delta T|| _ M ^ 2 = \langle \Delta T, C _ T ^ {-1} \Delta T \rangle \tag{5}$$

# 3. 可变形模型与图像的匹配

本文提出的是一种自动将可变形人脸模型与一个或多个图像匹配的算法。3D 模型的系数根据一组渲染参数进行优化，使得生成的图像与输入图像的差距越来越小。

模型参数为形状与纹理的系数 $\alpha _ j, \ \beta _ j, j=1,\ldots, m-1$。渲染参数 $\vec \rho$ 表示相机参数，包括方位角，仰角，目标尺度，图像平面的旋转角度和偏移，背景光强度 $i _ {r,amb}, i _ {g,amb}, i _ {b,amb}$，直射光强度 $i _ {r,dir}, i _ {g,dir}, i _ {b,dir}$。为了处理在不同条件下拍摄的照片，$\vec \rho$ 还包含了颜色对比度，红绿蓝三通道的偏移和增益，其他参数如相机距离等由用户估计后确定。

根据参数 $\alpha, \beta, \vec \rho$，生成图像为

$$I _ {model} (x,y) =(I _ {r,mod}(x,y), I _ {g,mod}(x,y), I _ {b,mod}(x,y)) ^ {\top} \tag{6}$$

重建的图像应该要与输入图像尽可能接近，使用欧氏距离测量，

$$E _ I = \sum _ {x,y} ||I _ {input}(x,y) - I _ {model}(x,y)|| ^ 2 \tag{7}$$

由于与 2D 图像匹配的 3D 模型可以有无数个，所以这两者的匹配是一个病态问题，并且还会出现很多不像人脸的重建结果，因此需要在解集上增加限制。

在人脸向量空间中，根据匹配质量和先验概率进一步限制匹配模型的解。根据 (2) 式得到先验概率 $p(\vec \alpha), \ p(\vec \beta)$，对 $p(\vec \rho)$ 进行相应的估计。

根据贝叶斯理论，问题变成给定输入一个输入图像 $I _ {input}$，求参数 $(\vec \alpha, \vec \beta, \vec \rho)$ 的最大后验概率。观测到的图像 $I _ {input}$ 也可能因为噪声也改变，假设噪声符合高斯分布，标准差为 $\sigma _ N$，那么观察图像的似然值就是以 $E _ I$ 为中心，$\sigma _ N$ 为标准差的高斯分布，这其实类似于回归分布，预测值为 $y$，而实际观测值由于噪声影响为 $\hat y$，那么 $\hat y \sim N(y, \sigma _ N)$，所以观测图像的似然为

$$p(I _ {input}|\vec \alpha, \vec \beta, \vec \rho) \sim \exp\left(- \frac {E _ I} {2 \sigma _ N ^ 2}\right) \tag{8}$$

于是，

$$p(I _ {input}) = p(I _ {input}|\vec \alpha, \vec \beta, \vec \rho) \cdot p(\vec \alpha) \cdot p(\vec \beta) \cdot p(\vec \rho) \tag{9}$$

其中 $p(\vec \alpha)$ 的先验概率见 (2) 式，损失函数为负对数似然，

$$E = \frac {E _ I} {\sigma _ N ^ 2} + \sum _ {j=1} ^ {m-1} \frac {\alpha _ j ^ 2}{\sigma _ {S,j} ^ 2} + \sum _ {j=1} ^ {m-1} \frac {\beta _ j ^ 2}{\sigma _ {T,j} ^ 2} + \sum _ {j=1} ^ {m-1} \frac {(\rho _ j - \overline \rho _ j) ^ 2}{\sigma _ {\rho,j} ^ 2} \tag{10}$$



要从 3D 模型得到 2D 的图像预测 $I _ {model}$，使用 3D 数据中的三角格坐标（见 `4.1` 一节数据集的介绍）。三角形 k 的中心的纹理 $(\overline R _ k, \overline G _ k, \overline B _ k) ^ {\top}$ 和 3D 坐标 $(\overline X _ k, \overline Y _ k, \overline Z _ k)$，可以通过三角形三个角点的均值计算得到，透视变换将这些三角形中心点投影到图像位置 $(\overline p _ {x,k}, \overline p _ {y,k}) ^ {\top}$ 处。三角形 k 的表面法向量 $\mathbf n _ k$ 由 k 的 3D 坐标确定，根据冯氏光照模型（ Phong illumination），颜色 R 通道成分为

$$I _ {r, model, k}=(i _ {r, amb} + i _ {r, dir} \cdot (\mathbf n _ k \mathbf l)) \overline R _ k + i _ {r,dir} s \cdot (\mathbf r _ k \mathbf v _ k) ^ {\nu} \tag{11}$$

其中 $\mathbf l$ 是照明方向，$\mathbf v _ k$ 是相机位置与三角形中心的归一化距离，$\mathbf r _ k = 2(\mathbf {nl}) \mathbf {n - l}$ 表示反射光的方向，$s$ 表示表面光泽度（表面对光的镜面反射能力），$\nu$ 控制镜面反射的角度分布。

当三角形 k 被阴影覆盖时，(6) 式退化为 $I _ {r,model,k} = i _ {r,amb} \overline R _ k$ 。 

B G 颜色则与 R 类似。

对于一个高分辨率的 3D 网格，$I _ {model}$ 在每个三角形 $k \in \{1,\ldots, n _ t\}$ 之间的变化很小，于是 $E _ I$ 近似为

$$E _ I = \sum _ {k=1} ^ {n _ t} a _ k \cdot ||I _ {input} (\overline p _ {x,k} \overline p _ {y,k}) - I _ {model,k}|| ^ 2 \tag{12}$$

其中 $a _ k$ 表示三角形 k 对应到图像中的覆盖面积。

在实际的梯度下降中，不同的三角形对梯度下降的贡献有冗余，所以作者随机选择了一个子集 $\mathcal K \subset \{1,\ldots, n _ t\}$，包含 40 个三角形，那么误差为

$$E _ {\mathcal K} = ||I _ {input}(\overline p _ {x,k} \overline p _ {y,k}) - I _ {model,k}|| ^ 2 \tag{13}$$

随机选择三角形 k 的概率为 $p(k \in \mathcal K) \sim a _ k$。

在第一次迭代之前，以及每 1000 次迭代之后，均需要计算当前模型的 3D shape，以及所有顶点的 2D 坐标 $(p _ x, p _ y) ^ {\top}$，然后确定 $a _ k$。

根据梯度下降更新参数 

$$\alpha _ j := \alpha _ j - \lambda _ j \cdot \frac {\partial E}{\partial \alpha _ j}\tag{14}$$

根据 (10) 式可知

$$\frac {\partial E}{\partial \alpha _ j}=\frac 1 {\sigma _ N ^ 2} \frac {\partial E _ I} {\partial \alpha _ j} +2 \frac {\alpha _ j} {\sigma _ {S,j} ^ 2}\tag{15}$$

其他参数类似。

# 4. 代码讲解

如果之前不了解人脸 3D 重建，那么看完论文还是一头雾水，其实本文的主要思想就是根据 200 个人脸数据（形状和纹理），建立人脸模型 (1) 式，这是平均人脸的模型。然后对于给定一张图片，将人脸模型与这张图片进行匹配，这样就得到输入图片对应的人脸 3D 模型，最后调整参数 $\alpha _ i, \ \beta _ i$，相当于对这个人脸的性别、年龄等面部特征进行调整。

为了弄清楚，这里大概讲解一下[代码](https://github.com/icygurkirat/3DMM-Matlab)。

## 4.1 数据集

在讲源码之前，首先认识一下数据集。

[**Basel Face Model**](https://faces.dmi.unibas.ch/bfm/index.php?nav=1-2&id=downloads)

- 3D 可变形人脸模型（形状与纹理）
- 性别、身高、体重和年龄的属性向量
- ...

此数据集中单个人脸的顶点数量为 53490 个，也就是说，人脸形状和纹理向量维度均为 `53490*3=160470`。

加载数据

```python
from scipy.io import loadmat
#标准的bfm模型包含顶点个数为53490个
#表情系数(来自Exp_Pca.bin)只针对53215个顶点有参数
#不含脖子的模型顶点个数为35709个.
original_BFM = loadmat('01_MorphableModel.mat')
shapePC = original_BFM['shapePC']   # 形状主成分，(199, 160470) (1) 式中的 s_i
shapeEV = original_BFM['shapeEV']   # 形状特征值，(199,) (2) 式中的 \sigma_i
shapeMU = original_BFM['shapeMU']   # 平均人脸， (160470,) (1) 式中的 \overline S
... # 纹理与形状类似
```

BFM 中已经计算好了平均人脸，以及 PCA 特征值和特征向量，这样就已经得到平均人脸的 3D 模型。

小结：

1. `shapeMU` 是平均形状向量, 也就是相当于一个平均人脸的形状, 大小是160470*1, 按照S=(X1,Y1,Z1, X2,Y2,Z2…)的格式存储
2. `shapePC` 是形状PCA降维矩阵, 大小是160470*199, 可以理解成是200个人脸形状数据降维后减少了一维,当然主要目的是为了正交
3. `shapeEV` 是形状正交空间的系数, 大小是199*1
4. `texMU` 是平均纹理向量, 大小是160470*1, 按照T=(R1,G1,B1,R2,G2,B2,…)的格式存储
5. `texPC` 是纹理PCA降维矩阵, 大小是160470*199
6. `texEV` 是纹理正交空间的系数, 大小是199*1
7. `tri` 是三角形的点的索引, 大小是106466*3, 代表有106466个三角形。有的数据包中 `tri` 为 `tl`。
8. `tri_mouth` 是嘴唇部分三角格坐标, 大小为114*3
9. `kpt_ind` 是关键点索引, 大小是68

## 4.2 基于平均人脸的微调

`EditorApp.m` 是基于平均人脸的 3D 模型进行微调，得到微调后的人脸。GUI 操作界面如图 1，

![](/images/face/3dmm_1.jpg)
<center>图 1. EditorApp.fig</center>

其中，使用了四个 slider 控件，调节 4 个形状特征向量对应的系数，例如第二个 slider 控件（第一个 slider 控件实际上没有调节第一个形状特征的系数，因为其回调函数为空），

```matlab
% EditorApp.m 文件

handles.model = load('01_MorphableModel.mat');
handles.currentShape=handles.model.shapeMU; % 形状平均
handles.currentTexture=handles.model.texMU; % 纹理平均
handles.state.s2=0; % 第二个控件的默认值

% --- Executes on slider movement.
function slider2_Callback(hObject, eventdata, handles)
sliderValue = single(get(hObject, 'Value'));        % 控件移动后的值
x=sliderValue-handles.state.s2;     % 差值
handles.state.s2=sliderValue;       % 记录新值
handles.currentShape=handles.currentShape+x*handles.model.shapeEV(2)*handles.model.shapePC(:,2);
update_face(gcf,handles.currentShape,handles.currentTexture,handles.model.tl,handles.rp);
guidata(hObject,handles);
```

上述代码中，`currentShape` 本来是平均人脸形状 $\overline S$，现在根据 $s _ 2$ 特征向量调节，调节后的形状向量为

$$S _ {new} = \overline S + x \cdot \sigma _ 2 \cdot s _ 2 \tag{16}$$

其中 $x$ 是调节值，$\sigma _ 2$ 是 $s _ 2$ 对应的特征值。 形状向量和纹理向量的调节类似。

除了形状和纹理，还可以调节面部属性，如性别、年龄、身高和体重。以 slider9 调节 age 为例，

```matlab
handles.attrib = load('04_attributes.mat');

function slider9_Callback(hObject, eventdata, handles)
sliderValue = single(get(hObject, 'Value'));
x=sliderValue-handles.state.s9;
handles.state.s9=sliderValue;
% texPC (53490*3, 199)      纹理特征向量 199 个 vectors
% age_tex (199, )   年龄显著度 199 个 scalars  
% texEV (199, )     纹理特征值 199 个 scalars
% 括号内 x*handles.attrib.age_tex(1:199,:).*handles.model.texEV 得到 (199, 1)
% 的列向量，作为纹理向量的权重
% currentTexture    (53490*3, 1) 列向量
handles.currentTexture=handles.currentTexture+handles.model.texPC*(x*handles.attrib.age_tex(1:199,:).*handles.model.texEV);
handles.currentShape=handles.currentShape+handles.model.shapePC*(x*handles.attrib.age_shape(1:199,:).*handles.model.shapeEV);
update_face(gcf,handles.currentShape,handles.currentTexture,handles.model.tl,handles.rp);
guidata(hObject,handles);
```

面部属性的显著值向量维度为 `200`，例如 `age_shape`，表示每个人脸形状的年龄显著值，选择前 199 个人脸形状的年龄显著值，与 slider 控件差值 `x` 相乘，得到每个人脸形状的年龄显著值的变化量，再与人脸形状的特征值相乘，得到人脸形状特征向量的系数，最后将 199 个特征向量加权求和，就得到因年龄改变人脸形状向量的差值 $\Delta S$。

以上是基于平均人脸，然后根据某个形状向量或其他属性进行调节，得到调节之后的人脸。

## 4.3 可变性 3D 模型与输入图片的匹配

以文件 `demo.m` 中的例子进行说明。

```matlab
model = load('01_MorphableModel.mat');  % 加载 BFM 数据包
% 显示平均人脸 \overline S, \overline T
[I, GCA] = align(model);

% 初始化参数
alpha = zeros(199,1);   % 每个 Si 对应的权重
beta = zeros(199,1);    % 每个 Ti 对应的权重
m = 199;            % 每个点的向量维度
sigma = 10000;      % sigma_N

Eold = rand();  % 随机初始化一个 E 的旧值，E 表示输入人脸与模型输出人脸的距离，作为目标损失
alpha_old = alpha; % alpha 的旧值
beta_old = beta;    % beta 的旧值

lambda_shape = 100; % 形状学习率
lambda_tex = 2;     % 纹理学习率

T = imread('Capture.PNG');  % 输入人脸图像
[row,col,~]=size(T);    
min_E=100000000000;     % 记录最小误差
min_alpha=alpha;        % 最小 alpha，也就是迭代到最后，与输入人脸最匹配的 alpha 值
min_beta=beta;          % 最小 beta，与输入人脸最匹配的 beta 值
const=10;
baditr=0;               % 迭代连续失败次数
maxit = 200000;         % 迭代总次数

for i = 1:maxit % 迭代 200,000 次
    I = imresize(I,[row/const,col/const]);  % 调整当前更新后的模型输出人脸的 size
    [x,y,z]=size(I);
    T_ = imresize(T,[x,y]); % 缩放输入人脸图像
    T_ = double(reshape(T_,[],1,1));
    I = double(reshape(I,[],1,1));
    E = norm(I/255 - T_/255)^2; % 差的平方和
    E = (E/(x*y))*255   % 归一化到 0 ~ 255 之间
    ...
    delE =  (E - Eold)/ (sigma*sigma);  % \Delta E, 即 \partial E，(10) 式等号右侧第一项
    % update alpha， (15) 式，但是丢弃了等号右侧第二项
    alpha_temp = alpha - lambda_shape*(calcGrad(delE,(alpha - alpha_old)));% + 2*(alpha)./(model.shapeEV.*model.shapeEV));
  
    % update beta
    beta_temp = beta - lambda_tex*(calcGrad(delE,(beta - beta_old)));% + 2*(beta)./(model.texEV.*model.texEV));

    Eold = E;           % 更新旧值，以便下一迭代中计算 \partial E
    beta_old = beta;
    alpha_old = alpha;
    alpha = alpha_temp;
    beta = beta_temp;

    % updated noise
    if(i==1||i==2)  % alpha beta 向量的前 5 个元素，加噪声
        alpha = alpha + 0.0005*([rand(5,1); zeros(m - 5,1)]);
        beta = beta + 0.0005*([rand(5,1); zeros(m - 5,1)]);
    else if((rem(i,5)==0||rem(i,5)==1)&&i<=972)
        % 以后每经过 5 次迭代，更新 alpha beta 向量中的下一个元素，加噪声
            alpha(5 + fix(i/5)) = alpha(5 + fix(i/5)) + 0.001*rand();
            beta(5 + fix(i/5)) = beta(5 + fix(i/5)) + 0.001*rand();
        end
    end

    % update I using alpha and beta
    I = get_update(model,alpha,beta,GCA);
end % 迭代结束

% (16) 式 更新人脸形状向量
shp = model.shapeMU + model.shapePC*(min_alpha.*model.shapeEV);
tex = model.texMU + model.texPC*(min_beta.*model.texEV);
I = get_update(model,min_alpha,min_beta,GCA);
```

上述代码就是使用梯度下降算法更新模型的 $\alpha, \beta$ 参数，参见上文的 (15) 式，但是代码中去掉了 (15) 式右侧第二项。

上述代码将 3D 点坐标投影到 2D 图像，这一步使用了 matlab 的 figure 显示功能：在 2D 平面上显示 3D 点。下面介绍如何自行求解这种投影变换。

我们使用一个 `3x4` 的放射矩阵来表示将人脸模型的 3D 点映射到平面中 2D 坐标，即

$$\mathbf x _ {2d} = P \cdot \mathbf X _ {3d} \tag{16}$$

**# 黄金标准算法**

使用 Gold Standard Algorithm 求解，即给定一组世界点与图像点的对应 $\{\mathbf X _ i \leftrightarrow \mathbf x _ i\}$，求解矩阵 $P$。

目标：

给定至少 4 组 3D (world) 到 2D (image) 的对应点 $\{\mathbf X _ i \leftrightarrow \mathbf x _ i\}$，确定仿射相机矩阵的最大似然估计 $P _ A$，即相机参数 $P$ 使得 $\sum _ i d(\mathbf x _ i, P \mathbf X _ i)^2$ 最小，同时满足仿射限制 $P ^ {3\top} = (0, 0, 0, 1)$，（$P ^ {3\top}$ 是 $P$ 的第三行）

算法：

1. 归一化

    使用一个相似矩阵 $T$ 归一化图像点，使用另一个相似矩阵 $U$ 归一化 world 点，记为 $\overline {\mathbf x} _ i = T \mathbf x _ i$，$\overline {\mathbf X} _ i = U \mathbf X _ i$ 。

2. 对每个点均使用 (16) 式，得到 3 个方程。

    实际上其中只有两个方程是线性无关的，第三个方程可由另外两个方程表示，不妨对每个点均使用前两个方程，这样一共得到 8 个方程，用矩阵表示为 $A \mathbf x = \mathbf b$ 。

3. 求出 $A$ 的伪逆 $A ^ +$，然后解的 $\mathbf x = A ^ + \mathbf b$。这里 $\mathbf x$ 就是由 $\overline P$ 中的行构成。

4. 去归一化，得到矩阵 $P=T ^ {-1} \overline P U$。

关于黄金标准算法参考 [相机模型（三）](/2024/05/08/dip/camera_model3)。

其实得到变换矩阵 $P$ 之后，还需要根据光源计算每个顶点的着色（例如 Pure shading，Flat shading，Gouraud shading 或者 Phong shading），然后再进行渲染，具体可以参考 [face3d](https://github.com/YadiraF/face3d) 中的 `1_pipeline.py` 中的代码。


**# 68 个人脸地标点**

前面说到 BFM 数据集中每个人脸有 53490 个点，其中 68 个人脸地标点的 index 见文件 [Landmarks68_BFM.anl](https://gitcode.com/anilbas/BFMLandmarks/blob/master/Landmarks68_BFM.anl) 。

```matlab
shape = reshape(shapeMU, 3, 53490)  % 平均人脸形状向量，53490 个点，每个点有 XYZ 三个坐标
shape = shape.'
x = shape(:, 1)
y = shape(:, 2)
z = shape(:, 3)
scatter3(x,y,z, 1, 'filled');   % 3D 三点图表示人脸

% 将其中的Landmarks68_BFM.anl文件内的68个下标导入Matlab然后显示出来
% tmp存储了Landmarks68_BFM.anl中的68个下标
scatter3(x,y,z,2, 'filled');
hold on;
for i = 1:68
    scatter3(x(tmp(i)), y(tmp(i)), z(tmp(i)),10, 'r');% 68 个地标使用红色点标记出来
end
```