# 人工智能基础_第二次大作业实验报告
> 新雅62/CDIE6
> 2016013327 项雨桐

> Kaggle用户名：Leona Xiang
> Public榜：#152 （0.86133）
> Private榜：#143 （0.86571）

## 1. 任务描述
完成10类图⽚的分类问题，图像示例及类别如下:

<img src="https://i.loli.net/2019/12/24/JUxVWRwYgjEbmZG.png" width="60%">

本次作业的数据包含30000张图片组成的训练集以及5000张图⽚组成的测试集，每个``.npy``⽂件包含⼀个N$\times$ 784的矩阵，N为图⽚数量。矩阵每⾏对应⼀张28$\times$ 28的图⽚。``train.csv``⽂件包含训练集的标签，含``image_id``和``label``两列，共30000行，``image_id``对应矩阵中的⾏下标，``label``为该图⽚的类别标签。

在预测环节，需要利⽤训练好的模型对测试集中的5000张图片进行分类，预测结果应生成 ``submit.csv``文件，同样包含``image_id``和``label``两列，共5000⾏，每⾏对应⼀张图⽚的结果。

解决本问题可以通过构建用于图像分类的卷积神经网络实现。将28$\times$ 28的训练集图像作为输入，经过一系列卷积运算提取特征，再利用全连接层将输出结果转化为分类标签（概率）。通过利用反向传播算法最小化损失函数，来得到一组具有较高分类准确率的网络参数，实现对测试集图像的分类。

## 2. 模型设计与实现
### 2.1 监督学习分类任务的基本模式
完成监督学习任务神经网络的训练是通过最小化损失函数实现的，也即：
$$
{\argmin_\theta}  O(\mathcal{D} ; \theta)=\sum_{i=1}^{n} L(y_{i}, f(x_{i}) ; \theta)+\Omega(\theta)
$$其中 $y_i$ 是训练标签，$f(x_i)$ 是训练得到的拟合函数，$\theta$ 是网络中所有的参数，$\Omega(\theta)$ 是正则项。

### 2.2 baseline：LeNet-5与AlexNet
#### 2.2.1 卷积神经网络概述
卷积神经网络是一类包含卷积计算且具有深度结构的前馈神经网络。基本结构包括输入层、卷积层、池化层、全连接层以及输出层。与之前DNN每一层进行全连接不同的是，卷积神经网络利用卷积结构实现逐层连接。与输入数据进行卷积运算的张量（通常为矩阵或向量）称为卷积核，这种卷积运算通常会提取出图像的某些局部特征。
$$
S(i, j)=(I * K)(i, j)=\sum_{m} \sum_{n} I(m, n) K(i-m, j-n)
$$卷积核的大小通常远远小于输入数据的大小，从而实现了各层之间的稀疏连接；由于核的每一个元素都作用在输入的每一位置上，卷积运算实现了参数共享，保证了我们只需要学习一个参数集合，从而降低了模型的存储需求；由于卷积运算的平移等变性，使得卷积神经网络对图像也具有平移等变性。池化是基于本层学到的函数具有局部平移不变性这一先验来进行的。池化使用某一位置的相邻输出的总体统计特征来代替网络在该位置的输出，是一种降采样过程。 全连接层则实现的是传统的仿射变换，它不会丢失信息，主要起到分类的作用，通常放置在卷积神经网络的最后一层。

#### 2.2.2 LeNet-5与AlexNet
LeNet-5和AlexNet都是非常经典的图像识别卷积网络。它们都采用了 “输入层$\to$(卷积层$+$池化层)$\to$全连接层$\to$输出层” 的线性结构，如图所示：

![lenet5.png](https://i.loli.net/2019/12/25/G89fZ2qBTMtLEdF.png)
![Screen Shot 2019-12-24 at 23.45.33.png](https://i.loli.net/2019/12/25/KgmUsptqyC4TzHr.png)

### 2.3 网络结构设计
#### 2.3.1 参数与结构设计
根据问题的规模（$28\times 28$输入，30000条训练样本），可以推测3-5层卷积网络应当比较合适。根据[Shashank Ramesh的建议](https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-ii-hyper-parameter-42efca01e5d7)，选择了四层网络，通道数分别为32，32，64，64，并在第2和第4层应用池化。

在图像处理中，卷积核尺寸多数为$3\times 3$～$11\times 11$，且通常取奇数。事实证明，小卷积核的叠加与大卷积核的连通性相同，但可以大大降低参数的个数和计算复杂度，并且可以产生更多的非线性，从而增强特征的提取能力。卷积核的大小最好随层数增加而增大，从而获得更全局化、更高阶和更有代表性的特征。因此本网络在1-3层采用$3\times 3$卷积核，第4层采用$5\times 5$卷积核。

卷积神经网络的池化过滤器大小通常为$2\times 2$和$3\times 3$，鉴于输入的参数数目较少，本网络采用的过滤器都为$2\times 2$大小。

卷积网络下一层输入大小与各参数的关系满足：

$$
\mathrm{output\_size} = \mathrm{floor({\frac{input\_size-kernel\_size+2\times padding}{stride}+1})}
$$

为调节输出大小，增添适当的padding参数，stride均取默认值1，得到各卷积层输入-输出大小关系如下：
<br/>
<br/>
<br/>

层序号|输入通道数|输出通道数|卷积核大小|padding步长|输入大小|池化过滤器大小|输出尺寸|总参数量
--|--|--|--|--|--|--|--|--
1|1|32|3|1|28||$\mathrm{floor(\frac{28-3+2\times1}{1})+1}=28$|$32\times 28\times 28$
2|32|32|3|1|28|2|$(\mathrm{floor(\frac{28-3+2\times1}{1})+1})/2=14$|$32\times 14\times 14$
3|32|64|3|1|14||$\mathrm{floor(\frac{14-3+2\times1}{1})+1}=14$|$64\times 14\times 14$
4|64|64|5|0|14|2|$(\mathrm{floor(\frac{14-5+2\times0}{1})+1})/2=5$|$64\times 5\times 5$

为减小过拟合，原来的全连接层被替换为全局平均池化（Global Average Pooling，GAP）层（详见3.1.2节）。GAP层的输出大小设置为$3\times 3$，总参数为$64\times3\times3=576$个。再通过$576\rightarrow 10$的全连接映射实现10分类问题。

#### 2.3.2 网络设计图示

![nn.png](https://i.loli.net/2019/12/24/KU2n3pDSyhBQeW9.png)
### 2.4 激活函数：ReLU
网络中采用的激活函数为整流线性单元（Rectified Linear Unit，ReLU）：

$$g(z)=\max \{0, z\}$$

ReLU易于优化，因为它在正向的一阶导数保持不变，避免了传统的Sigmoid函数在远离零点处梯度消失的问题，但其在负数区间输出为零，使得神经元易死亡。

激活层应放在卷积层之后，但和池化层之间的位置关系对网络性能没有影响。

### 2.5 优化器选择：Adam
适应性矩估计 (Adaptive Moment Estimation，Adam) 算法是一种收敛速度很快的优化算法。和固定学习率的随机梯度下降不同， Adam 通过计算梯度的一阶矩估计和二阶矩估计而为不同的参数设计独立的自适应性学习率，同时获得了AdaGrad算法（自适应梯度，为每一个参数保留一个学习率） 和 RMSProp 算法（均方根传播，基于权重梯度最近量级的均值为每一个参数适应性地保留学习率）的优点。算法伪码如下所示：

<img src="https://i.loli.net/2019/12/26/T5SrXlw79fGY6Bp.png" width="60%">

这里的步长设置为0.0001，Weight Decay正则项（见 3.1.3）一般采用Lasso正则项或Ridge正则项。``PyTorch``优化器内部的正则项为Ridge正则项，即：

$$
\Omega(\theta)=\frac{\lambda}{2} \sum_{l=1}^{n_{l}-1} \sum_{i=1}^{s_{l}} \sum_{j=1}^{s_{l+1}}\left(\theta_{j i}^{(l)}\right)^{2}
$$

### 2.6 损失函数：Cross Entropy Loss
分类问题的损失函数常采用交叉熵损失函数（Cross Entropy Loss）。本任务里 $n=10$，$y_i\in\{0,1,2,...,9\}$：

$$
L(h(x_i),y_i)=\left\{
\begin{array}{cc}
{-\log \left[\frac{\exp \left(w_{0}^{T} x\right)}{\sum_{r=0}^{9} \exp \left(w_{r}^{T} x\right)}\right]} & {y_{i}=0} \\
{-\log \left[\frac{\exp \left(w_{1}^{T} x\right)}{\sum_{r=0}^{9} \exp \left(w_{r}^{T} x\right)}\right]} & {y_{i}=1} \\
\vdots&\vdots& \\
{-\log \left[\frac{\exp \left(w_{9}^{T} x\right)}{\sum_{r=0}^{9} \exp \left(w_{r}^{T} x\right)}\right]} & {y_{i}=9}
\end{array}\right.
$$

## 3 模型优化
### 3.1 正则化方法
#### 3.1.1 Batch Normalization
批归一化（Batch Normalization，BN）最早来自Google 的 Inception Net v2，被用于解决深层网络的协方差偏移问题。由于引入了归一化过程，故可提高模型的泛化能力，抑制过拟合现象，因此可用于替代AlexNet中用于正则化的`dropout`层和`LRN` 层，从而可以使用相对更高的学习率，提高网络的训练效率。

使用 $B$ 表示整个训练集的批次，大小为$m$ 。 $B$ 的均值和方差因此可以表示为

$$\mu_{B}=\frac{1}{m} \sum_{i=1}^{m} x_{i}\\ \sigma_{B}^{2}=\frac{1}{m} \sum_{i=1}^{m}\left(x_{i}-\mu_{B}\right)^{2}$$

对于具有$d$维输入的层，$x=\left(x^{(1)}, \ldots, x^{(d)}\right)$ ，输入的每个维分别归一化为：

$$\hat{x}_{i}^{(k)}=\frac{x_{i}^{(k)}-\mu_{B}^{(k)}}{\sqrt{{\sigma_{B}^{(k)}}^{2}+\epsilon}},$$

 这里 $k \in[1, d]$，$i \in[1, m]$，$\mu_{B}^{(k)}$ 和 $\sigma_{B}^{(k)}$ 分别是每个维度的均值和方差。

$\epsilon$ 被添加到分母中以防止方差过小导致的数值不稳定，是一个任意小的常数。如果不考虑 $\epsilon$ ，则得到的归一化激活 $\hat{x}^{(k)}$ 的均值为0，方差为1。

归一化的输出改变了本层特征分布，因此并不能直接作为其他层的输入。为了恢复网络的表示能力，接下来的转换步骤如下：

$$y_{i}^{(k)}=\gamma^{(k)} \hat{x}_{i}^{(k)}+\beta^{(k)}$$

参数 $\gamma^{(k)} =  \sigma{(x^{(k)})}$ 和 $\beta^{(k)} =\mathbb{E}[x^{(k)}]$ 通过随后的优化过程学习得到。整个BN过程的形式化表述为：

$$\mathrm{BN}_{\gamma^{(k)}, \beta^{(k)}}: x_{1, m}^{(k)} \rightarrow y_{1, m}^{(k)}$$

BN过程的输出 $y^{(k)}=\mathrm{BN}_{\gamma^{(k)}, \beta^{(k)}}\left(x^{(k)}\right)$ 被传递到网络的其他层，而被正则化的 $\hat{x}_{i}^{(k)}$ 则留在了当前层。

BN的操作是针对单个神经元的操作。在卷积神经网络中，每个特征图（也即每个通道）被视为一个神经元。在``PyTorch``中，``BatchNorm2d``的输入是一个四维矩阵，维数$(B,C,H,W)$分别代表batch的大小、通道数、输入矩阵的高和宽。对每个特征图应用一组 $\gamma^{(k)}$  和 $\beta^{(k)}$ 。

BN层一般添加在卷积层后、池化层和激活层之前，以及全连接层之后。这里，每一个卷积层之后都应用了BN层。

#### 3.1.2 Global Average Pooling
GAP最早被NUS 的Network In Network采用，被用于卷积神经网络中替代全连接层，以抑制神经网络的过拟合现象。与传统的``dropout``相比，GAP没有额外的超参数，只是通过简单的对特征图求平均来建立特征图和分类类别之间的联系。GAP可以看做是一种整个网络上的结构化正则，即对多个特征图的空间平均，因此剔除了全连接层中黑箱的特征。

#### 3.1.3 Weight Decay
深度学习优化器内部的权重衰减（weight decay）即为传统机器学习领域损失函数内部的正则项。使用正则项可以将模型的参数限制在一定范围内，从而降低模型的方差、避免过拟合。

### 3.2 其他优化手段
这些优化手段都是在Kaggle竞赛结束后引入的，因此对Kaggle的成绩没有影响。但事实证明，这些手段可以进一步提升模型的性能。

#### 3.2.1 Learning Rate Decay
学习率衰减主要被用于解决小批量、固定学习率的情景下在一定世代数后验证集损失振荡、不再下降的问题。学习率下降有多种不同的策略，这里选用的是阶梯下降法，即每隔一定的世代数（10），学习率缩减一定的比例（0.1）。

实践证明，应用了学习率衰减之后，验证集损失下降明显（0.01-0.03），准确率得到了 0.5%-1% 的提升。

#### 3.2.2 Ensemble Learning - Bagging
集成学习（ Ensemble Learning）是指训练多个模型并将其组合使用的过程，有并行模式和串行模式两种。对于具有过拟合倾向的模型来说，宜采用并行集成来减小模型的方差，也即Bagging方法。

<img src="https://i.loli.net/2019/12/26/DulSaHWzZGRpgCh.png" width="30%">

Bagging方法由Leo Breiman在1996年提出，其基础是Bootstrap方法。Bootstrap方法是一种对数据的重采样过程，特别适用于小型数据集的处理。其基本的策略是对数据集进行数次有放回的抽样，以生成多个新数据集。实践表明，这些对样本的抽样产生的数据集的统计性质非常接近对总体抽样产生的数据集。[理论表明](https://stats.stackexchange.com/questions/263710/why-should-boostrap-sample-size-equal-the-original-sample-size/275746)，当且仅当每个新数据集保持和原数据集的大小相同时，新数据集的标准差是总体标准差的无偏估计。

基于新生成的数据集的统计特性，我们可以用每个新数据集分别训练一个模型，再将这些模型的结果用投票的方式聚合起来，得到最终的预测结果。聚合结果的期望与单个模型相同，但方差是原来的$1/n$。

在本次作业中，我们将5个并行训练的模型得到的结果进行聚合。实践证明，在预测效果上得到了明显提升（0.5%-1%）。

## 4. 实验总结
### 4.1 测试结果
验证集采用留出法获得。训练集使用随机选取的25000条数据，每个子模型使用的训练集由原训练集Bootstrap得到，验证集使用5000条数据。分别计算在两个数据集上的准确率和损失。

Kaggle提交版选用的模型训练情况见附录二。

最后一次训练使用的世代数为40，在验证集上得到的准确率为87.6%，损失为0.385。具体过程件见附录三。

### 4.2 实验收获
#### 4.2.1 网络架构设计
网络的架构决定了模型的性能上限。由于深度学习神经网络参数数目庞大的特性，网络结构的设计与调试变成了一项高成本、高不确定度的工作。关于网络层数、通道数、卷积核大小的选取目前还没有严谨的理论支持，甚至前人的经验也只能供参考，因为这些都和问题的规模和具体任务类型强烈相关，因而难以借鉴。

但网络的设计仍有些通用的原则去遵循，这些在本次作业的实践中被验证有效。例如：

1. **选取合适的baseline**：baseline可以是论文中实现的经典网络或者经过实践检验的经典设计，实际上决定了模型性能的下限。本次作业选取的LeNet-5和AlexNet是CNN图像识别领域的奠基之作，LeNet-5也恰恰是用于解决28$\times$ 28输入的10分类问题的网络，因此选取它们做baseline是合理的；但LeNet-5面对的手写数据集问题相对简单，而AlexNet在2014年ImageNet大赛上实际的top-5准确率也只有80%左右。因此，如果想追求90%以上的准确率，以这两个模型作为baseline起点是有点低的。

2. **结构参数选取的原则**：与选取baseline的原则类似，网络参数的选取也尽量要遵循流行的做法。比如本次作业的 “卷积$\to$卷积$\to$池化$\to$卷积$\to$卷积$\to$池化” 以及 “32-32-64-64”的通道数设计都是经典设计，事实证明要比自己盲目设计的网络参数好很多；对于图片较小的问题，小的卷积核在深度上堆叠的效果要胜于大卷积核，卷积核尺寸通常为3$\times 3$和$5\times 5$；池化过滤器大小一般为$2\times 2$和$3\times 3$；网络的通道数要保持不变或逐层递增。

3. **各层的堆叠顺序**：这似乎是一个没有统一意见的问题，但可以确定的是池化层与激活层要在卷积层之后，池化层与激活层的位置理论上可以互换（所有的激活函数都是单调非减因而是保序的，但先池化后激活的计算量要少）；BN层的位置则存在争议，按照归一化的作用，BN层要加在卷积层和激活层之间，但实践中也有很多设计将BN层放在了激活层之后，个人支持第一种设计；droput层一般只用于全连接层之间，若用GAP层代替全连接层，则后面无需使用dropout或者BN层。

本次作业最大的难点在于控制网络的结构复杂度。最开始基于LeNet-5设计的三层模型性能很不稳定（方差大），表现出了强烈的过拟合倾向，但准确率有限。因为网络层数有限，各种调参的方法收效都不明显；而且不幸的是，大幅修改网络参数并没有带来任何启示性信息，因此在初期阻碍了实验的进展；直到添加了一层网络之后，各种调参方法的效果才逐渐显现出来。后来又尝试换用推荐的经典通道数，也就是现在的 “32-32-64-64”，这个网络在深层的总通道数小于之前的网络（ “16-64-128-192”），性能得到了明显提升。

总之，这段经历验证了“小而深的网络性能好于大而浅的网络”的论调，但贸然添加网络层数有其内在风险。网络的复杂度应该和问题的复杂度相关联，但无论是之前的网络还是现在的网络，都在较少的epoch数之后表现出比较强烈的过拟合倾向，这说明网络的结构复杂度还是偏高；但尽管如此，更深层次的网络还是带来了更高的准确率。由于时间成本有限，并没有在推荐参数上做进一步的改进实验，因此结构复杂度的控制仍然需要继续探索与研究。

#### 4.2.2 过拟合现象的抑制
如前所述，本模型采用了许多手段来抑制过拟合。
1. **Batch Normalization**：这是被许多人推荐的正则化方法，事实证明在抑制过拟合上确实有效。另外，通常认为BN比dropout更有效，而且BN层和dropout层组合使用可能会使模型性能降低。
2. **Global Average Pooling**：使用GAP的一个信念是过拟合主要发生在全连接层。事实证明，作为一种结构性优化手段，GAP的效果确实明显，并且要比全连接层更有表达能力。但，只有全连接层的信息是无损的，所以GAP不能完全替代全连接层，至少，最后一层的分类任务还是要靠全连接层实现。
3. **Weight Decay**：这是最经典也是最常用的正则化手段，但作为含有超参数的方法，其调试仍然是一项费时费力的工程。Weight Decay过大显然会降低模型的准确率（欠拟合），但Weight Decay过小不会起到降低方差的作用。经过调试本实验选取的值0.005是比较合适的。

#### 4.2.3 参数调试与优化
深度学习的各个参数相互影响，其实很难找出一条固定的调参优先级路径，但优化算法和学习率的选择永远是第一步。本实验采用Adam算法具有收敛快、自适应学习率的特点，因此得到了广泛采用。尽管如此，Adam的初始学习率仍然需要进行decay过程以使验证集误差免于振荡。这里面初始学习率的选择、decay方法的选择又引入了新的超参数，需要时间与精力进行调试。学习率大会导致方差大、损失振荡，但引入了一定的随机性；学习率小可以让模型平稳收敛，但速度较慢。经过实验，选用初始学习率为0.0001，每隔15个epoch学习率折半的方法比较有效。

另外，batch_size和epoch的选取决定了模型训练效率与准确率的平衡，这些也是要经过反复实验才可以。

#### 4.2.4 其他体会
本次实验的数据集相对较小，容易导致模型过拟合，但如果抑制过拟合，模型在训练时又很难达到令人满意的准确率，这是需要做权衡的部分。另外，数据集本身有意引入了噪声，这使得模型的训练变得更加困难。

因为托福、德国实践、缺乏GPU以及个人时间规划等种种原因，本次作业并没有留足充分的时间训练和调试模型，不免有些许遗憾，但对待神经网络的调试仍然需要充分的耐心方能达到效果。希望下次调试神经网络是能够有更充足的时间和资源。


## 附录一：程序配置/运行环境/版本号
-  `system`: `MacOS 10.15.2`
-  `Anaconda3`:`4.8.0`
-  `Jupyter Notebook`: `1.0.0`
-  `nb_conda`:`2.2.1`
-  `nb_conda_kernels`:`2.2.2`
-  `Python`: `3.7.5`
	- `numpy`:`1.17.3`
	- `pandas`:`0.25.1`
	- `pytorch`:`1.3.1`
	- `scikit-learn`:`0.21.3`

## 附录二：Kaggle版模型运行结果
共计训练40个epoch，提交的模型是第24个epoch的结果
```
epoch: 0
train_accuracy: 0.6955962276214833
train_loss: 0.9872112705579499
validation_accuracy:0.7630000
validation_loss:0.7079363

epoch: 1
train_accuracy: 0.7902093989769822
train_loss: 0.6013252652819504
validation_accuracy:0.7740000
validation_loss:0.6177762

epoch: 2
train_accuracy: 0.8160326086956522
train_loss: 0.5187566021976568
validation_accuracy:0.8106000
validation_loss:0.5289928

epoch: 3
train_accuracy: 0.8309942455242967
train_loss: 0.4746737595638046
validation_accuracy:0.8324000
validation_loss:0.4863320

epoch: 4
train_accuracy: 0.8448049872122763
train_loss: 0.438414677291575
validation_accuracy:0.8360000
validation_loss:0.4646253

epoch: 5
train_accuracy: 0.8495444373401535
train_loss: 0.4168661530212978
validation_accuracy:0.8278000
validation_loss:0.4664564

epoch: 6
train_accuracy: 0.8609814578005115
train_loss: 0.3929597109632419
validation_accuracy:0.8402000
validation_loss:0.4512263

epoch: 7
train_accuracy: 0.8659367007672635
train_loss: 0.37728822829625797
validation_accuracy:0.8508000
validation_loss:0.4310966

epoch: 8
train_accuracy: 0.8734335038363171
train_loss: 0.3604379996771703
validation_accuracy:0.8556000
validation_loss:0.4227646

epoch: 9
train_accuracy: 0.8773577365728901
train_loss: 0.34843485522300693
validation_accuracy:0.8518000
validation_loss:0.4204462

epoch: 10
train_accuracy: 0.8834798593350384
train_loss: 0.336008204942774
validation_accuracy:0.8520000
validation_loss:0.4158883

epoch: 11
train_accuracy: 0.8873081841432225
train_loss: 0.3232527254030223
validation_accuracy:0.8550000
validation_loss:0.4075892

epoch: 12
train_accuracy: 0.8913123401534527
train_loss: 0.31430803910088356
validation_accuracy:0.8556000
validation_loss:0.4063407

epoch: 13
train_accuracy: 0.8951646419437339
train_loss: 0.3045574286785882
validation_accuracy:0.8592000
validation_loss:0.3997236

epoch: 14
train_accuracy: 0.9005994245524296
train_loss: 0.2953056806264936
validation_accuracy:0.8540000
validation_loss:0.4005498

epoch: 15
train_accuracy: 0.9036764705882352
train_loss: 0.2848545802409387
validation_accuracy:0.8564000
validation_loss:0.3940852

epoch: 16
train_accuracy: 0.9102461636828645
train_loss: 0.27483121078947315
validation_accuracy:0.8574000
validation_loss:0.3990617

epoch: 17
train_accuracy: 0.9094629156010231
train_loss: 0.27066883254234136
validation_accuracy:0.8660000
validation_loss:0.3857429

epoch: 18
train_accuracy: 0.9127078005115089
train_loss: 0.2613486043937371
**validation_accuracy:0.8694000**
validation_loss:0.3755697

epoch: 19
train_accuracy: 0.9178308823529412
train_loss: 0.2541560306192359
validation_accuracy:0.8560000
validation_loss:0.3924459

epoch:20
train_accuracy: 0.9194932864450128
train_loss: 0.24658068157065555
validation_accuracy:0.8636000
validation_loss:0.3939892

epoch: 21
train_accuracy: 0.9237052429667519
train_loss: 0.23962790825787714
validation_accuracy:0.8482000
validation_loss:0.4095405

epoch: 22
train_accuracy: 0.9274296675191815
train_loss: 0.2316138546942445
validation_accuracy:0.8638000
validation_loss:0.3794880

epoch: 23
train_accuracy: 0.9292439258312021
train_loss: 0.2269194780103386
**validation_accuracy:0.8684000**
validation_loss:0.3824721

epoch: 24
train_accuracy: 0.9324728260869566
train_loss: 0.21910972925631897
**validation_accuracy:0.8710000**
validation_loss:0.3608968

epoch: 25
train_accuracy: 0.9362292199488491
train_loss: 0.21116364472891058
validation_accuracy:0.8578000
validation_loss:0.3966027

epoch: 26
train_accuracy: 0.9379475703324809
train_loss: 0.20679439817700546
validation_accuracy:0.8652000
validation_loss:0.3810470

epoch: 27
train_accuracy: 0.9432145140664961
train_loss: 0.19868731936987707
validation_accuracy:0.8628000
validation_loss:0.3806930

epoch: 28
train_accuracy: 0.9427269820971866
train_loss: 0.19714391465915743
**validation_accuracy:0.8694000**
validation_loss:0.3679303

epoch: 29
train_accuracy: 0.946283567774936
train_loss: 0.18933799752341512
validation_accuracy:0.8662000
validation_loss:0.3722540

epoch: 30
train_accuracy: 0.9479859335038364
train_loss: 0.18538409595370597
**validation_accuracy:0.8684000**
validation_loss:0.3752102

epoch: 31
train_accuracy: 0.9514785805626598
train_loss: 0.17833379264492208
**validation_accuracy:0.8692000**
validation_loss:0.3722945

epoch: 32
train_accuracy: 0.9524216751918159
train_loss: 0.17620189563202127
validation_accuracy:0.8626000
validation_loss:0.3934737

epoch: 33
train_accuracy: 0.955898337595908
train_loss: 0.16951977426324355
validation_accuracy:0.8630000
validation_loss:0.3848854

epoch: 34
train_accuracy: 0.9575687340153453
train_loss: 0.16509017901842857
**validation_accuracy:0.8698000**
validation_loss:0.3834235

epoch: 35
train_accuracy: 0.9593670076726343
train_loss: 0.16012938067202678
validation_accuracy:0.8616000
validation_loss:0.3929435

epoch: 36
train_accuracy: 0.9616847826086957
train_loss: 0.1542930172570526
validation_accuracy:0.8634000
validation_loss:0.3798049

epoch: 37
train_accuracy: 0.9631154092071611
train_loss: 0.14941601730559184
**validation_accuracy:0.8702000**
validation_loss:0.3636862

epoch: 38
train_accuracy: 0.9644501278772379
train_loss: 0.1484469679253333
validation_accuracy:0.8632000
validation_loss:0.3822201

epoch: 39
train_accuracy: 0.9654651534526854
train_loss: 0.14491089381982603
**validation_accuracy:0.8704000**
validation_loss:0.3780509
```

## 附录三：最终版模型运行结果
```
epoch: 0
model_1
train_accuracy: 0.704507672634271
train_loss: 0.9640275041008239
validation_accuracy:0.7764000
validation_loss:0.6674562
model_2
train_accuracy: 0.7045556265984655
train_loss: 0.9703263047406131
validation_accuracy:0.7636000
validation_loss:0.6775237
model_3
train_accuracy: 0.7095748081841433
train_loss: 0.9492436722111519
validation_accuracy:0.7742000
validation_loss:0.6718438
model_4
train_accuracy: 0.7018542199488491
train_loss: 0.9715272642462455
validation_accuracy:0.7576000
validation_loss:0.6842696
model_5
train_accuracy: 0.7020140664961637
train_loss: 0.975733178884477
validation_accuracy:0.7606000
validation_loss:0.6862850

epoch: 1
model_1
train_accuracy: 0.8013666879795397
train_loss: 0.5764463217666996
validation_accuracy:0.8048000
validation_loss:0.5513464
model_2
train_accuracy: 0.7975223785166241
train_loss: 0.5856866065955832
validation_accuracy:0.7998000
validation_loss:0.5684175
model_3
train_accuracy: 0.8015824808184143
train_loss: 0.5815236408387303
validation_accuracy:0.8050000
validation_loss:0.5543124
model_4
train_accuracy: 0.7931825447570332
train_loss: 0.5924205516305421
validation_accuracy:0.7896000
validation_loss:0.5824558
model_5
train_accuracy: 0.7945092710997442
train_loss: 0.5937398614938302
validation_accuracy:0.8054000
validation_loss:0.5787230

epoch: 2
model_1
train_accuracy: 0.8286205242966752
train_loss: 0.4872963572173472
validation_accuracy:0.8198000
validation_loss:0.5094106
model_2
train_accuracy: 0.826326726342711
train_loss: 0.4975531952612845
validation_accuracy:0.8154000
validation_loss:0.5183626
model_3
train_accuracy: 0.8303148976982097
train_loss: 0.4907672081304633
validation_accuracy:0.8184000
validation_loss:0.5151656
model_4
train_accuracy: 0.8246243606138108
train_loss: 0.502780663616517
validation_accuracy:0.8126000
validation_loss:0.5186358
model_5
train_accuracy: 0.8244725063938618
train_loss: 0.5030498445186469
validation_accuracy:0.8156000
validation_loss:0.5142021

epoch: 3
model_1
train_accuracy: 0.8547634271099743
train_loss: 0.4293107136588572
validation_accuracy:0.8246000
validation_loss:0.4809712
model_2
train_accuracy: 0.8444133631713554
train_loss: 0.44877932908589885
validation_accuracy:0.8264000
validation_loss:0.4832736
model_3
train_accuracy: 0.8496882992327365
train_loss: 0.43839633483868423
validation_accuracy:0.8290000
validation_loss:0.4742365
model_4
train_accuracy: 0.8443334398976982
train_loss: 0.44960379124145067
validation_accuracy:0.8128000
validation_loss:0.5076276
model_5
train_accuracy: 0.8429507672634271
train_loss: 0.4479511740719876
validation_accuracy:0.8300000
validation_loss:0.4767080

epoch: 4
model_1
train_accuracy: 0.8669517263427109
train_loss: 0.39293547862630973
validation_accuracy:0.8368000
validation_loss:0.4586658
model_2
train_accuracy: 0.8573529411764707
train_loss: 0.4088003933429718
validation_accuracy:0.8218000
validation_loss:0.4800149
model_3
train_accuracy: 0.8632113171355499
train_loss: 0.39837878020218265
validation_accuracy:0.8352000
validation_loss:0.4676372
model_4
train_accuracy: 0.8575687340153453
train_loss: 0.41090886271975535
validation_accuracy:0.8310000
validation_loss:0.4615296
model_5
train_accuracy: 0.8620204603580562
train_loss: 0.40604002152562446
validation_accuracy:0.8282000
validation_loss:0.4742877

epoch: 5
model_1
train_accuracy: 0.8788283248081841
train_loss: 0.36129181974989066
validation_accuracy:0.8472000
validation_loss:0.4315242
model_2
train_accuracy: 0.8706042199488491
train_loss: 0.38106250244638196
validation_accuracy:0.8256000
validation_loss:0.4664594
model_3
train_accuracy: 0.8754795396419437
train_loss: 0.36804260378298553
validation_accuracy:0.8394000
validation_loss:0.4487350
model_4
train_accuracy: 0.8712276214833758
train_loss: 0.3775995814281961
validation_accuracy:0.8352000
validation_loss:0.4515488
model_5
train_accuracy: 0.8736892583120205
train_loss: 0.3751681940939725
validation_accuracy:0.8320000
validation_loss:0.4610487

epoch: 6
model_1
train_accuracy: 0.8898097826086956
train_loss: 0.33408745864163275
validation_accuracy:0.8382000
validation_loss:0.4419736
model_2
train_accuracy: 0.8789402173913042
train_loss: 0.35775963020751544
validation_accuracy:0.8388000
validation_loss:0.4473004
model_3
train_accuracy: 0.8857736572890025
train_loss: 0.3424454981561207
validation_accuracy:0.8460000
validation_loss:0.4284091
model_4
train_accuracy: 0.8828804347826087
train_loss: 0.3523585642009135
validation_accuracy:0.8378000
validation_loss:0.4512651
model_5
train_accuracy: 0.8836876598465474
train_loss: 0.34868746778696696
validation_accuracy:0.8406000
validation_loss:0.4418707

epoch: 7
model_1
train_accuracy: 0.8983855498721228
train_loss: 0.31313731866267025
validation_accuracy:0.8514000
validation_loss:0.4131473
model_2
train_accuracy: 0.891480179028133
train_loss: 0.32976459953791043
validation_accuracy:0.8402000
validation_loss:0.4384983
model_3
train_accuracy: 0.8938618925831202
train_loss: 0.32234659829103124
validation_accuracy:0.8412000
validation_loss:0.4245341
model_4
train_accuracy: 0.8948129795396419
train_loss: 0.3258943119088707
validation_accuracy:0.8446000
validation_loss:0.4247088
model_5
train_accuracy: 0.889593989769821
train_loss: 0.32727969999965806
validation_accuracy:0.8354000
validation_loss:0.4479938

epoch: 8
model_1
train_accuracy: 0.9068893861892583
train_loss: 0.2930104946693801
validation_accuracy:0.8442000
validation_loss:0.4189012
model_2
train_accuracy: 0.899352621483376
train_loss: 0.3109819038825877
validation_accuracy:0.8480000
validation_loss:0.4252152
model_3
train_accuracy: 0.9028372762148337
train_loss: 0.2995184899291114
validation_accuracy:0.8370000
validation_loss:0.4358081
model_4
train_accuracy: 0.9031889386189259
train_loss: 0.3047188063106878
validation_accuracy:0.8398000
validation_loss:0.4380545
model_5
train_accuracy: 0.902389705882353
train_loss: 0.30408998506377116
validation_accuracy:0.8448000
validation_loss:0.4231840

epoch: 9
model_1
train_accuracy: 0.9147138746803068
train_loss: 0.27557953944444047
validation_accuracy:0.8396000
validation_loss:0.4324276
model_2
train_accuracy: 0.9067695012787724
train_loss: 0.2930076485285369
validation_accuracy:0.8500000
validation_loss:0.4149044
model_3
train_accuracy: 0.9114689897698209
train_loss: 0.2812921324044542
validation_accuracy:0.8456000
validation_loss:0.4249022
model_4
train_accuracy: 0.9090313299232736
train_loss: 0.28636403252249176
validation_accuracy:0.8436000
validation_loss:0.4212935
model_5
train_accuracy: 0.9089434143222507
train_loss: 0.28660474610907954
validation_accuracy:0.8394000
validation_loss:0.4340781

epoch: 10
model_1
train_accuracy: 0.9207161125319693
train_loss: 0.2579699059200409
validation_accuracy:0.8504000
validation_loss:0.4080100
model_2
train_accuracy: 0.9141144501278772
train_loss: 0.2763770270301863
validation_accuracy:0.8504000
validation_loss:0.4067858
model_3
train_accuracy: 0.916272378516624
train_loss: 0.2652286337998212
validation_accuracy:0.8528000
validation_loss:0.4079022
model_4
train_accuracy: 0.918957800511509
train_loss: 0.2656954114927965
validation_accuracy:0.8420000
validation_loss:0.4350332
model_5
train_accuracy: 0.9175671355498721
train_loss: 0.26758753647432304
validation_accuracy:0.8344000
validation_loss:0.4432419

epoch: 11
model_1
train_accuracy: 0.9275575447570332
train_loss: 0.2424170819618513
validation_accuracy:0.8522000
validation_loss:0.4007713
model_2
train_accuracy: 0.9201886189258311
train_loss: 0.2588860094928376
validation_accuracy:0.8490000
validation_loss:0.4081852
model_3
train_accuracy: 0.926502557544757
train_loss: 0.24519994212767046
validation_accuracy:0.8336000
validation_loss:0.4476879
model_4
train_accuracy: 0.9267503196930946
train_loss: 0.25078698314364306
validation_accuracy:0.8458000
validation_loss:0.4240112
model_5
train_accuracy: 0.9229939258312021
train_loss: 0.2524340020116333
validation_accuracy:0.8216000
validation_loss:0.4813671

epoch: 12
model_1
train_accuracy: 0.9336876598465472
train_loss: 0.2285770119532295
validation_accuracy:0.8540000
validation_loss:0.3913486
model_2
train_accuracy: 0.9248881074168799
train_loss: 0.2469547797766183
validation_accuracy:0.8520000
validation_loss:0.4039216
model_3
train_accuracy: 0.9300911125319693
train_loss: 0.23469137863429915
validation_accuracy:0.8504000
validation_loss:0.4030859
model_4
train_accuracy: 0.9296595268542199
train_loss: 0.2367991341654297
validation_accuracy:0.8424000
validation_loss:0.4237815
model_5
train_accuracy: 0.9272058823529412
train_loss: 0.2398773105553044
validation_accuracy:0.8426000
validation_loss:0.4388457

epoch: 13
model_1
train_accuracy: 0.9412563938618926
train_loss: 0.2144876836282213
validation_accuracy:0.8452000
validation_loss:0.4037638
model_2
train_accuracy: 0.9295796035805627
train_loss: 0.23309908706285154
validation_accuracy:0.8478000
validation_loss:0.4113427
model_3
train_accuracy: 0.9339354219948849
train_loss: 0.2232931100613321
validation_accuracy:0.8546000
validation_loss:0.3991218
model_4
train_accuracy: 0.9374760230179029
train_loss: 0.21922931833492826
validation_accuracy:0.8472000
validation_loss:0.4163629
model_5
train_accuracy: 0.9350383631713556
train_loss: 0.2230225670939821
validation_accuracy:0.8510000
validation_loss:0.4085939

epoch: 14
model_1
train_accuracy: 0.9449488491048594
train_loss: 0.2008642480539544
validation_accuracy:0.8560000
validation_loss:0.3938677
model_2
train_accuracy: 0.9366368286445014
train_loss: 0.21905809581813301
validation_accuracy:0.8496000
validation_loss:0.4069324
model_3
train_accuracy: 0.9398257672634271
train_loss: 0.2102246050487089
validation_accuracy:0.8524000
validation_loss:0.4087239
model_4
train_accuracy: 0.9418558184143222
train_loss: 0.20933150280924404
validation_accuracy:0.8494000
validation_loss:0.4103232
model_5
train_accuracy: 0.9405690537084399
train_loss: 0.21075980338599065
validation_accuracy:0.8508000
validation_loss:0.4061850

epoch: 15
model_1
train_accuracy: 0.9577125959079283
train_loss: 0.17395644251952697
validation_accuracy:0.8546000
validation_loss:0.3957338
model_2
train_accuracy: 0.9506633631713556
train_loss: 0.1898893873633631
validation_accuracy:0.8528000
validation_loss:0.3944550
model_3
train_accuracy: 0.9554347826086956
train_loss: 0.18093407121689423
validation_accuracy:0.8578000
validation_loss:0.3938955
model_4
train_accuracy: 0.9563778772378517
train_loss: 0.1791593442906809
validation_accuracy:0.8518000
validation_loss:0.4007054
model_5
train_accuracy: 0.9538203324808184
train_loss: 0.18218366089074509
validation_accuracy:0.8514000
validation_loss:0.3930803

epoch: 16
model_1
train_accuracy: 0.9609574808184144
train_loss: 0.16640368969086797
validation_accuracy:0.8596000
validation_loss:0.3828442
model_2
train_accuracy: 0.953476662404092
train_loss: 0.18344211629818163
validation_accuracy:0.8572000
validation_loss:0.3923331
model_3
train_accuracy: 0.9582800511508951
train_loss: 0.1736318238670259
validation_accuracy:0.8588000
validation_loss:0.3932755
model_4
train_accuracy: 0.9602141943734016
train_loss: 0.16962342705491865
validation_accuracy:0.8554000
validation_loss:0.3907694
model_5
train_accuracy: 0.9572969948849105
train_loss: 0.17459138583801592
validation_accuracy:0.8544000
validation_loss:0.3937849

epoch: 17
model_1
train_accuracy: 0.9633232097186701
train_loss: 0.15992224172634237
validation_accuracy:0.8494000
validation_loss:0.3869833
model_2
train_accuracy: 0.9573129795396419
train_loss: 0.1739017599264679
validation_accuracy:0.8590000
validation_loss:0.3879570
model_3
train_accuracy: 0.9596946930946292
train_loss: 0.1665559024609568
validation_accuracy:0.8558000
validation_loss:0.3976505
model_4
train_accuracy: 0.9637867647058823
train_loss: 0.16263969762779562
validation_accuracy:0.8606000
validation_loss:0.3842452
model_5
train_accuracy: 0.9595748081841433
train_loss: 0.16874690803572956
validation_accuracy:0.8580000
validation_loss:0.3915659

epoch: 18
model_1
train_accuracy: 0.9654012148337596
train_loss: 0.15441367357893063
validation_accuracy:0.8566000
validation_loss:0.3785068
model_2
train_accuracy: 0.9598705242966752
train_loss: 0.16818613244596956
validation_accuracy:0.8564000
validation_loss:0.3925366
model_3
train_accuracy: 0.961644820971867
train_loss: 0.16195383281125436
validation_accuracy:0.8530000
validation_loss:0.4000170
model_4
train_accuracy: 0.9637867647058823
train_loss: 0.1585723689812071
validation_accuracy:0.8542000
validation_loss:0.3943027
model_5
train_accuracy: 0.9624520460358057
train_loss: 0.16268929191257642
validation_accuracy:0.8492000
validation_loss:0.4160436

epoch: 19
model_1
train_accuracy: 0.9681505754475703
train_loss: 0.1483118221964068
validation_accuracy:0.8584000
validation_loss:0.3850917
model_2
train_accuracy: 0.9633312020460358
train_loss: 0.16272258398401768
validation_accuracy:0.8600000
validation_loss:0.3903271
model_3
train_accuracy: 0.9646499360613812
train_loss: 0.1569660087223248
validation_accuracy:0.8590000
validation_loss:0.3851392
model_4
train_accuracy: 0.9675831202046036
train_loss: 0.15243822384787642
validation_accuracy:0.8516000
validation_loss:0.4013221
model_5
train_accuracy: 0.9641703964194374
train_loss: 0.156753183706947
validation_accuracy:0.8522000
validation_loss:0.3997023

epoch: 20
model_1
train_accuracy: 0.9705003196930946
train_loss: 0.1418014308604438
validation_accuracy:0.8552000
validation_loss:0.3901838
model_2
train_accuracy: 0.9633312020460358
train_loss: 0.15736020689882585
validation_accuracy:0.8562000
validation_loss:0.4029813
model_3
train_accuracy: 0.9658887468030691
train_loss: 0.15175863728879968
validation_accuracy:0.8580000
validation_loss:0.3934448
model_4
train_accuracy: 0.9695012787723785
train_loss: 0.14540602941342326
validation_accuracy:0.8530000
validation_loss:0.3922804
model_5
train_accuracy: 0.9645300511508952
train_loss: 0.1527215615581826
validation_accuracy:0.8566000
validation_loss:0.3837673

epoch: 21
model_1
train_accuracy: 0.9702605498721227
train_loss: 0.13855521125561746
validation_accuracy:0.8536000
validation_loss:0.3870120
model_2
train_accuracy: 0.9672794117647059
train_loss: 0.15015799373083408
validation_accuracy:0.8564000
validation_loss:0.3972549
model_3
train_accuracy: 0.9683264066496164
train_loss: 0.14543972835135277
validation_accuracy:0.8548000
validation_loss:0.3920188
model_4
train_accuracy: 0.9723385549872122
train_loss: 0.13951004702416833
validation_accuracy:0.8562000
validation_loss:0.3873624
model_5
train_accuracy: 0.9674712276214834
train_loss: 0.14588691909676013
validation_accuracy:0.8558000
validation_loss:0.3852157

epoch: 22
model_1
train_accuracy: 0.9743206521739131
train_loss: 0.13401682975004092
validation_accuracy:0.8560000
validation_loss:0.3864774
model_2
train_accuracy: 0.9673753196930945
train_loss: 0.146691400071849
validation_accuracy:0.8610000
validation_loss:0.3910322
model_3
train_accuracy: 0.9692375319693095
train_loss: 0.14132814610476993
validation_accuracy:0.8612000
validation_loss:0.3852203
model_4
train_accuracy: 0.9734255115089514
train_loss: 0.1350847747167358
validation_accuracy:0.8554000
validation_loss:0.3963779
model_5
train_accuracy: 0.9712036445012788
train_loss: 0.1417022863464892
validation_accuracy:0.8528000
validation_loss:0.3925460

epoch: 23
model_1
train_accuracy: 0.9746962915601023
train_loss: 0.1296103830685091
validation_accuracy:0.8572000
validation_loss:0.3784772
model_2
train_accuracy: 0.9704443734015346
train_loss: 0.13972991222844405
validation_accuracy:0.8558000
validation_loss:0.3986524
model_3
train_accuracy: 0.9712595907928389
train_loss: 0.1364486336212634
validation_accuracy:0.8550000
validation_loss:0.3956143
model_4
train_accuracy: 0.9741608056265986
train_loss: 0.13146777461518716
validation_accuracy:0.8560000
validation_loss:0.3992844
model_5
train_accuracy: 0.9713235294117648
train_loss: 0.1353336683738872
validation_accuracy:0.8508000
validation_loss:0.4010101

epoch: 24
model_1
train_accuracy: 0.9750399616368287
train_loss: 0.1261633261466575
validation_accuracy:0.8550000
validation_loss:0.3868944
model_2
train_accuracy: 0.9726182864450128
train_loss: 0.13545214527708185
validation_accuracy:0.8580000
validation_loss:0.3910914
model_3
train_accuracy: 0.9733216112531969
train_loss: 0.13168356675283074
validation_accuracy:0.8596000
validation_loss:0.3883508
model_4
train_accuracy: 0.9763586956521739
train_loss: 0.1276115941841279
validation_accuracy:0.8572000
validation_loss:0.3985096
model_5

train_accuracy: 0.9733455882352942
train_loss: 0.1314836885217968
validation_accuracy:0.8566000
validation_loss:0.3897738

epoch: 25
model_1
train_accuracy: 0.9779411764705882
train_loss: 0.1207209301783758
validation_accuracy:0.8610000
validation_loss:0.3856689
model_2
train_accuracy: 0.9736253196930946
train_loss: 0.13140685140819805
validation_accuracy:0.8526000
validation_loss:0.3989685
model_3
train_accuracy: 0.9745044757033248
train_loss: 0.1285590622526453
validation_accuracy:0.8600000
validation_loss:0.3798013
model_4
train_accuracy: 0.9777813299232737
train_loss: 0.12199163682701643
validation_accuracy:0.8538000
validation_loss:0.4069577
model_5
train_accuracy: 0.9742007672634272
train_loss: 0.1267306368293055
validation_accuracy:0.8566000
validation_loss:0.3857002

epoch: 26
model_1
train_accuracy: 0.9798113810741688
train_loss: 0.11524995810845319
validation_accuracy:0.8542000
validation_loss:0.3854172
model_2
train_accuracy: 0.9751598465473147
train_loss: 0.12731652523931639
validation_accuracy:0.8558000
validation_loss:0.3964780
model_3
train_accuracy: 0.9761588874680307
train_loss: 0.12282702251506583
validation_accuracy:0.8602000
validation_loss:0.3793357
model_4
train_accuracy: 0.9790920716112532
train_loss: 0.11870176015455094
validation_accuracy:0.8502000
validation_loss:0.4040153
model_5
train_accuracy: 0.9774376598465474
train_loss: 0.12297433299367386
validation_accuracy:0.8600000
validation_loss:0.3791066

epoch: 27
model_1
train_accuracy: 0.9793797953964194
train_loss: 0.11360493192777914
validation_accuracy:0.8574000
validation_loss:0.3845889
model_2
train_accuracy: 0.9767822890025576
train_loss: 0.12440004864769519
validation_accuracy:0.8586000
validation_loss:0.3877068
model_3
train_accuracy: 0.9769421355498721
train_loss: 0.12064621529882522
validation_accuracy:0.8644000
validation_loss:0.3855241
model_4
train_accuracy: 0.9811141304347827
train_loss: 0.11539145537159022
validation_accuracy:0.8538000
validation_loss:0.3946031
model_5
train_accuracy: 0.9774376598465474
train_loss: 0.11974850426549496
validation_accuracy:0.8536000
validation_loss:0.3881684

epoch: 28
model_1
train_accuracy: 0.98125
train_loss: 0.10981172957764868
validation_accuracy:0.8554000
validation_loss:0.3846963
model_2
train_accuracy: 0.9777014066496164
train_loss: 0.11971514964538157
validation_accuracy:0.8582000
validation_loss:0.3923825
model_3
train_accuracy: 0.9789002557544757
train_loss: 0.11645001741816931
validation_accuracy:0.8630000
validation_loss:0.3834356
model_4
train_accuracy: 0.9822170716112532
train_loss: 0.1099661486342435
validation_accuracy:0.8544000
validation_loss:0.3938921
model_5
train_accuracy: 0.9789721867007672
train_loss: 0.1154068730714376
validation_accuracy:0.8596000
validation_loss:0.3810913

epoch: 29
model_1
train_accuracy: 0.9827365728900256
train_loss: 0.10573546233994272
validation_accuracy:0.8592000
validation_loss:0.3860329
model_2
train_accuracy: 0.9796195652173914
train_loss: 0.11560429977562726
validation_accuracy:0.8540000
validation_loss:0.3982765
model_3
train_accuracy: 0.9800671355498721
train_loss: 0.11194448902860017
validation_accuracy:0.8580000
validation_loss:0.3971240
model_4
train_accuracy: 0.9831281969309462
train_loss: 0.10728305088513343
validation_accuracy:0.8518000
validation_loss:0.4004046
model_5
train_accuracy: 0.9802589514066496
train_loss: 0.11155339402844534
validation_accuracy:0.8582000
validation_loss:0.3876311

epoch: 30
model_1
train_accuracy: 0.9872122762148338
train_loss: 0.09463741855167063
validation_accuracy:0.8582000
validation_loss:0.3787216
model_2
train_accuracy: 0.9858535805626598
train_loss: 0.10253738875850997
validation_accuracy:0.8594000
validation_loss:0.3927735
model_3
train_accuracy: 0.9855498721227622
train_loss: 0.09936632113078671
validation_accuracy:0.8656000
validation_loss:0.3785216
model_4
train_accuracy: 0.987971547314578
train_loss: 0.09465736850067173
validation_accuracy:0.8568000
validation_loss:0.3884316
model_5
train_accuracy: 0.986053388746803
train_loss: 0.09852994380094816
validation_accuracy:0.8582000
validation_loss:0.3789871

epoch: 31
model_1
train_accuracy: 0.9880115089514067
train_loss: 0.09178793676140364
validation_accuracy:0.8606000
validation_loss:0.3818432
model_2
train_accuracy: 0.9859335038363172
train_loss: 0.09959907750682452
validation_accuracy:0.8614000
validation_loss:0.3861064
model_3
train_accuracy: 0.9862292199488492
train_loss: 0.09673443672907017
validation_accuracy:0.8672000
validation_loss:0.3830128
model_4
train_accuracy: 0.988650895140665
train_loss: 0.09207039082522893
validation_accuracy:0.8548000
validation_loss:0.3923660
model_5
train_accuracy: 0.9871882992327367
train_loss: 0.09609385405469428
validation_accuracy:0.8612000
validation_loss:0.3764067

epoch: 32
model_1
train_accuracy: 0.9886109335038363
train_loss: 0.09050726371309946
validation_accuracy:0.8646000
validation_loss:0.3786483
model_2
train_accuracy: 0.9859894501278773
train_loss: 0.098973244562021
validation_accuracy:0.8590000
validation_loss:0.3874778
model_3
train_accuracy: 0.9867886828644502
train_loss: 0.09572659921653741
validation_accuracy:0.8674000
validation_loss:0.3803164
model_4
train_accuracy: 0.9888906649616368
train_loss: 0.09020105151035597
validation_accuracy:0.8588000
validation_loss:0.3880193
model_5
train_accuracy: 0.9876118925831202
train_loss: 0.09489259619237211
validation_accuracy:0.8600000
validation_loss:0.3727868

epoch: 33
model_1
train_accuracy: 0.9885869565217392
train_loss: 0.08929683389070699
validation_accuracy:0.8608000
validation_loss:0.3790618
model_2
train_accuracy: 0.9878676470588236
train_loss: 0.09666824618073376
validation_accuracy:0.8612000
validation_loss:0.3860024
model_3
train_accuracy: 0.9875879156010231
train_loss: 0.09433951897694327
validation_accuracy:0.8640000
validation_loss:0.3795191
model_4
train_accuracy: 0.9891703964194374
train_loss: 0.08942133436917954
validation_accuracy:0.8570000
validation_loss:0.3927044
model_5
train_accuracy: 0.987531969309463
train_loss: 0.09279255920549488
validation_accuracy:0.8596000
validation_loss:0.3828669

epoch: 34
model_1
train_accuracy: 0.9891703964194374
train_loss: 0.08735530346136569
validation_accuracy:0.8586000
validation_loss:0.3845999
model_2
train_accuracy: 0.9877317774936062
train_loss: 0.09480387981399856
validation_accuracy:0.8618000
validation_loss:0.3938044
model_3
train_accuracy: 0.9872282608695653
train_loss: 0.09302546863284562
validation_accuracy:0.8636000
validation_loss:0.3798387
model_4
train_accuracy: 0.9890105498721228
train_loss: 0.08830611990845721
validation_accuracy:0.8558000
validation_loss:0.3922869
model_5
train_accuracy: 0.9874520460358056
train_loss: 0.0923547342686397
validation_accuracy:0.8582000
validation_loss:0.3733021

epoch: 35
model_1
train_accuracy: 0.989769820971867
train_loss: 0.08585597066889944
validation_accuracy:0.8608000
validation_loss:0.3833197
model_2
train_accuracy: 0.9880115089514067
train_loss: 0.09352552094270507
validation_accuracy:0.8604000
validation_loss:0.3915233
model_3
train_accuracy: 0.9882912404092071
train_loss: 0.0915872893865456
validation_accuracy:0.8648000
validation_loss:0.3827132
model_4
train_accuracy: 0.9901694373401535
train_loss: 0.0850636526427763
validation_accuracy:0.8532000
validation_loss:0.3950935
model_5
train_accuracy: 0.9883471867007673
train_loss: 0.09051870643292241
validation_accuracy:0.8588000
validation_loss:0.3777694

epoch: 36
model_1
train_accuracy: 0.9903053069053709
train_loss: 0.08449024128753815
validation_accuracy:0.8622000
validation_loss:0.3828300
model_2
train_accuracy: 0.9885709718670077
train_loss: 0.0912058016718806
validation_accuracy:0.8596000
validation_loss:0.3880725
model_3
train_accuracy: 0.9891464194373402
train_loss: 0.08880386560621774
validation_accuracy:0.8640000
validation_loss:0.3812376
model_4
train_accuracy: 0.9895859974424553
train_loss: 0.0858938694286072
validation_accuracy:0.8590000
validation_loss:0.3904892
model_5
train_accuracy: 0.9889705882352942
train_loss: 0.08760421171479518
validation_accuracy:0.8556000
validation_loss:0.3867309

epoch: 37
model_1
train_accuracy: 0.9907848465473147
train_loss: 0.08336225826093151
validation_accuracy:0.8590000
validation_loss:0.3778257
model_2
train_accuracy: 0.9893861892583121
train_loss: 0.09002704144743702
validation_accuracy:0.8616000
validation_loss:0.3923069
model_3
train_accuracy: 0.9889466112531969
train_loss: 0.08847950478953778
validation_accuracy:0.8650000
validation_loss:0.3767597
model_4
train_accuracy: 0.99056905370844
train_loss: 0.08297757576684207
validation_accuracy:0.8584000
validation_loss:0.3894552
model_5
train_accuracy: 0.9887468030690537
train_loss: 0.0880317287829221
validation_accuracy:0.8600000
validation_loss:0.3787120

epoch: 38
model_1
train_accuracy: 0.9916080562659847
train_loss: 0.0813778033742057
validation_accuracy:0.8606000
validation_loss:0.3795675
model_2
train_accuracy: 0.9893702046035806
train_loss: 0.08876046477376348
validation_accuracy:0.8580000
validation_loss:0.3934482
model_3
train_accuracy: 0.9901694373401535
train_loss: 0.08562966509510184
validation_accuracy:0.8646000
validation_loss:0.3830978
model_4
train_accuracy: 0.991304347826087
train_loss: 0.08152509799889286
validation_accuracy:0.8574000
validation_loss:0.3920267
model_5
train_accuracy: 0.989769820971867
train_loss: 0.08649551310121556
validation_accuracy:0.8596000
validation_loss:0.3789462

epoch: 39
model_1
train_accuracy: 0.9911445012787724
train_loss: 0.07971446655328622
validation_accuracy:0.8610000
validation_loss:0.3839801
model_2
train_accuracy: 0.98909047314578
train_loss: 0.0876976365838057
validation_accuracy:0.8600000
validation_loss:0.3864322
model_3
train_accuracy: 0.9891863810741689
train_loss: 0.08566697842210455
validation_accuracy:0.8626000
validation_loss:0.3821925
model_4
train_accuracy: 0.9916879795396419
train_loss: 0.07997080525550086
validation_accuracy:0.8584000
validation_loss:0.3935507
model_5
train_accuracy: 0.9904891304347826
train_loss: 0.08409446208259029
validation_accuracy:0.8618000
validation_loss:0.3785302

final_validation_accuracy:0.8760000
final_validation_loss:0.3849371
```
