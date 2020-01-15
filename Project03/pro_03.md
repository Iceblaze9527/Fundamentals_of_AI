# 人工智能基础_第三次大作业实验报告
> 新雅62/CDIE6
> 2016013327 项雨桐

## 1. 任务描述
### 1.1 MountainCar-v0
#### 1.1.1 简介
一辆小车在一维轨道上运动，轨道有两个山峰，我们的目标是把小车开到右侧的山峰上。但是，这辆小车的动力不足以支持其一次爬过山坡。因此，成功的唯一方式是通过前后振荡来获得动量。

#### 1.1.2 观测值
小车的位置和速度，连续型变量

num | observation | min |  max 
-- | -- | -- | --
0 | 位置 | -1.2 | 0.6
1 | 速度 | -0.07 | 0.07
 
#### 1.1.3  行动值
离散型变量

num | action
-- | -- 
0 | 向左推
1 | 不推
2 | 向右推

#### 1.1.4 奖励值 
每一步的奖励值都是 `-1`，直到达到目标位置`0.5`之前。爬到左侧山坡上并不会带来惩罚。

#### 1.1.5 初始位置
在`-0.6`和`-0.4`之间的随机位置，没有速度。

#### 1.1.6 终止条件
达到`0.5`位置 ，或迭代`200`次。

### 1.2 MountainCarContinuous-v0
#### 1.2.1 简介
一辆小车在一维轨道上运动，轨道有两个山峰，我们的目标是把小车开到右侧的山峰上。但是，这辆小车的动力不足以支持其一次爬过山坡。因此，成功的唯一方式是前后振荡来获得动量。在这个任务中，如果使用更少的能量将会获得更多的回报。

#### 1.2.2 观测值
与`MountainCar-v0`相同。

#### 1.2.3 行动值
连续型变量，向右为正，向左为负。

#### 1.2.4 奖励值
到达`0.5`位置的奖励值为`100`，减去从开始到目标行动值的和的平方。这个奖励函数给探索带来了挑战——如果智能体不能尽早到达目标，它就会认为不移动是更好的选择，也就不会找到目标。

#### 1.2.5 初始位置
与`MountainCar-v0`相同。

#### 1.2.6 终止条件
达到`0.5`位置 ，可限制迭代次数。

#### 1.2.7  解决要求
得到`90`以上的奖励值，这个值可以调整。

## 2. 算法原理
本次作业采用Q-learning和DQN两种方式来完成`MountainCar-v0`，采用DDPG来完成`MountainCarContinuous-v0`

### 2.1 Q-learning
#### 2.1.1 时序差分
Q-learning是时序差分方法的一种，是蒙特卡洛思想和动态规划思想的结合。对于每次访问的蒙特卡洛方法，其递推状态评价为：
$$
V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left[G_{t}-V\left(S_{t}\right)\right]$$
在时序差分中，我们用下一时刻的状态价值函数来估计下一时刻的累积回报，就可以得到：
$$
V\left(S_{t}\right) \leftarrow V\left(S_{t}\right)+\alpha\left[R_{t+1}+\gamma V\left(S_{t+1}\right)-V\left(S_{t}\right)\right]
$$
利用时序差分解决控制问题时，我们学习的是动作价值函数，也即：
$$
Q\left(S_{t}, A_{t}\right) \leftarrow Q\left(S_{t}, A_{t}\right)+\alpha\left[R_{t+1}+\gamma Q\left(S_{t+1}, A_{t+1}\right)-Q\left(S_{t}, A_{t}\right)\right]
$$

#### 2.1.2 Q-learning 算法
具体而言，Q-learning  算法的目标策略选择的是贪心算法，也即
$$
\begin{aligned}
&A_{t+1}=\underset{a \in A}{\arg \max } Q\left(S_{t+1}, a\right)\\
&Q\left(S_{t+1}, A_{t+1}\right)=Q\left(S_{t+1}, \argmax _{a \in A} Q\left(S_{t+1}, a\right)\right)=\max _{a \in A} Q\left(S_{t+1}, a\right)
\end{aligned}
$$
而行动策略则选择的是$\varepsilon$-贪心策略：
$$
\pi\left(a | S_{t}\right) \leftarrow\left\{\begin{array}{ll}
{1-\varepsilon+\varepsilon /\left|\mathcal{A}\left(S_{t}\right)\right|} & {\text { if } a=\argmax _{a \in A} Q\left(S_{t}, a\right)} \\
{\varepsilon /\left|\mathcal{A}\left(S_{t}\right)\right|} & {\text { otherwise } }
\end{array}\right.
$$
Q-learning算法以程序形式显示如下:

![01.png](https://i.loli.net/2020/01/15/vOdDTEBRcSoInXY.png)

整个学习过程靠维护一张状态-动作值的Q表来实现，因此需要状态空间和连续空间都是离散的。

### 2.2 Deep Q-Network (DQN)
#### 2.2.1 值函数近似
DQN是在Q-learning基础上利用深度神经网络进行函数拟合得到的算法。对于高维数据，Q-learning中建立和维护状态-动作的Q表变得不切实际。而根据[万能近似定理](https://www.wikiwand.com/en/Universal_approximation_theorem)，只要激活函数合适，一个具有有限个数神经元的单隐含层前馈神经网络可以近似$\mathbb{R}^n$的任意紧致子集上的连续函数。因此，我们可以利用深度神经网络来近似Q值的分布，而无需假定离散的状态空间。

#### 2.2.2 Target网络
在监督学习中，神经网络的训练需要假定一个损失函数，这里的损失函数为当前Q值与目标Q值的均方损失。但是，目标Q值事实上是未知的，我们是用$R_{t+1}+\gamma Q\left(S_{t+1}, A_{t+1}\right)$来估计这个目标Q值，而这本身又和当前的Q值相关。换言之，在调整当前Q值的同时我们也调整了对目标Q值的估计。

为了解决这个问题，DQN引入了Target网络，这个网络的结构参数和用来估计当前Q值的Primary网络完全相同，只不过参数更新不会过于频繁，只在Primary网络迭代若干个周期后将其参数拷贝过来，以保证Target网络的稳定性。

#### 2.2.3 [经验回放](https://www.zhihu.com/question/278182581)
监督学习的一大假设是数据是独立同分布产生的，但在强化学习中，相邻状态之间的相关程度很高，而神经网络拟合的函数通常是连续的（与Q表相区别），这就使得在调整网络参数时也会修改当前状态附近的状态的Q值。而我们对目标Q值的估计又依赖于当前状态附近的其他状态，因此这会起到放大作用，导致算法在朝着同一个方向做梯度下降，使得模型过拟合，方差增大，从而不稳定。

因此，我们引入了经验回放的方法。设置一定长度的缓冲区用来存储过去的$(s,a,r,s’)$。在训练的时候，随机从缓冲区里均匀采样一个batch来调整网络的参数，从而减弱训练数据之间的相关性。

#### 2.2.4 DQN 算法
DQN算法以程序形式显示如下:

![02.png](https://i.loli.net/2020/01/15/JhZLzvY1G7lE9wu.png)
### 2.3 Deep Deterministic Policy Gradient (DDPG)
#### 2.3.1 策略梯度方法
与之前的算法不同，策略梯度的方法学习的是策略函数本身。我们将策略函数参数化为
$$
\pi(a | s, \theta)=\operatorname{P}(A_{t}=a | S_{t}=s, \theta_{t}=\theta)
$$
其中$\theta$表示策略的参数向量。我们可以考虑基于这些参数制定一个优化目标函数$J(\theta)$，利用梯度上升方法更新参数：
$$
\theta_{t+1}=\theta_{t}+\alpha \widehat{\nabla J\left(\theta_{t}\right)}
$$
其中$\widehat{\nabla J\left(\theta_{t}\right)} \in \mathbb{R}^{d^{\prime}}$是个随机估计， 其期望近似于优化目标函数相对于其参数 $\theta_t$的梯度，遵循此一般模式的所有方法，我们都称为策略梯度方法。

有很多种形式的$J(\theta)$可供选择，但无论哪种形式，其梯度都可以表示为
$$
\nabla_{\theta} J(\theta)=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) Q^{\pi_{\theta}(\theta)}(s, a)\right]
$$
这就是策略梯度定理。
策略梯度可以应对连续的动作空间，这是我们选用该方法来完成`MountainCarContinuous-v0`的原因。

#### 2.3.2 Actor-Critic方法
最简单的策略梯度算法是蒙特卡洛策略梯度算法，它用每个时间位置的状态价值函数来估计策略梯度定理中的$Q^{\pi_{\theta}(\theta)}(s, a)$。但其存在速度慢（高方差）、不可在线学习等问题，因此有了Actor-Critic方法。Actor-Critic方法使用Q网络技术来估计价值函数（或其他评估点），以动作价值为例：
$$
Q_{w}(s, a) \approx Q^{\pi_{\theta}}(s, a)
$$
Critic中的DQN网络会更新参数$w$，而Actor会根据Critic输出的价值更新策略梯度的参数$\theta$。

#### 2.3.3 DDPG算法
Actor-Critic算法存在难收敛的问题，因此还需要进一步优化。DDPG的优化借鉴了DQN中的方法，即经验回放和双网络。
##### 确定性策略梯度（DPG）
在连续型动作空间的策略梯度下降中，策略是一个概率分布函数，需要进行采样才能获得对应的动作；而高维动作空间的频繁采样十分消耗计算资源，而且计算策略梯度时需要进行积分，这一般通过蒙特卡洛采样进行估算，也十分耗费算力。所幸，严格的数学推导显示，环境模型无关(model-free)的确定性策略是存在的：
$$
\nabla_{\theta^{\mu}} J_{\beta}(\mu) \approx E_{s \sim \rho^{\beta}}\left[\left.\nabla_{a} Q\left(s, a | \theta^{Q}\right)\right|_{a=\mu(s)} \cdot \nabla \theta^{\mu} \mu\left(s | \theta^{\mu}\right)\right]
$$
其中$\mu$即最优行为策略，满足$A_{t}=\mu\left(S_{t} | \theta^{\mu}\right)$，不再需要采样。这个$\mu$函数可以用神经网络进行拟合，也就是Actor的神经网络。

##### Ornstein-Uhlenbeck过程
在强化学习过程中，我们要兼顾 exploration和exploit，其中exploration的目的是探索潜在的更优策略。所以，训练过程中我们为action的决策机制引入OU噪声，其定义如下：
$$
d x_{t}=\theta\left(\mu-x_{t}\right) d t+\sigma dW
$$
其中，​ $x_t$ 就表示动作的噪声， $\mu$ 表示它的均值， $W$ 表示维纳过程（布朗运动），是一种外界的随机噪声，​$\sigma$ 是随机噪声的权重。

OU过程是一个时间相关的过程，可以使智能体很好的探索具备动量属性的环境。

##### DDPG 网络结构
DDPG共有4个网络：

1. Actor当前网络：负责策略网络参数$\theta^{\mu}$的迭代更新，负责根据当前状态$S$选择当前动作$A$，用于和环境交互生成$S'$，$R$。
2. Actor目标网络：负责根据经验回放池中采样的下一状态$S'$选择最优下一动作$A'$。网络参数$\theta^{\mu'}$定期从$\theta^{\mu}$复制。
3. Critic当前网络：负责价值网络参数$w$的迭代更新，负责计算负责计算当前Q值和估计的目标Q值。
4. Critic目标网络：负责计算目标Q值中的$Q_{\theta^{Q}}\left(S_{t+1}, A_{t+1}\right)$部分。网络参数$\theta^{Q'}$定期从$\theta^{Q}$复制。

这里的参数更新方法是软更新，即：
$$
\begin{aligned}
&\theta^{Q'} \leftarrow \tau \theta^{Q}+(1-\tau) \theta^{Q'}\\
&\theta^{\mu'} \leftarrow \tau \theta^{\mu}+(1-\tau) \theta^{\mu'}
\end{aligned}
$$其中$\tau$是更新系数，一般取的比较小

DDPG的网络框架如图所示：

![04.jpg](https://i.loli.net/2020/01/15/CVocxhEPmaj5GeI.jpg)
##### 算法流程
DDPG 算法以程序形式显示如下:

![03.png](https://i.loli.net/2020/01/15/w2XM65H1YRKF4TI.jpg)
## 3 模型设计与优化
### 3.1 网络结构设计
#### 3.1.1 DQN
DQN输入的状态空间是2，输出的动作空间是3；中间是1层全连接网络，神经元个数为30，使用ReLU进行激活；输出层采用线性激活；优化器采用Adam。

![05.png](https://i.loli.net/2020/01/15/eiBxTvE2CzSAyrq.png)
#### 3.1.2 DDPG
##### Actor网络
Actor网络是全连接网络，输入的状态空间是2维连续变量，输出的动作空间是1维连续变量；中间有2个隐层，神经元个数分别是400和300，采用ReLU进行激活；输出层采用`tanh`函数激活；优化器采用Adam。

![07.png](https://i.loli.net/2020/01/15/KPDjVLfliZCewOn.png)
##### Critic网络
Critic网络分别接受状态空间和动作空间的输入。状态空间输入经历神经元个数分别是400和300的2个隐层，第1层采用ReLU进行激活；动作空间输入经历1个神经元个数为300的隐层；2个隐层合并之后再经过一个ReLU激活的、神经元个数为300的隐层，最后给出1维的价值输出；优化器采用Adam。

![06.png](https://i.loli.net/2020/01/15/VJ9onKAczUq2wSv.png)
### 3.2 奖励优化
#### 3.2.1 MountainCar-v0
这个环境中的默认奖励很简单，每一个时间步都是-1，为了让模型更快的收敛，可以修改奖励。启发式的奖励更新公式为：

 ``reward = base + a1 * abs(position + 0.5) + a2 * max(position - p0,0) + a3 * velocity``

* `base`：基础奖励
* `a1 * abs(position + 0.5)`：初始状态的平均值为`-0.5`，向左或向右都有利于积蓄动量，因此这部分可以避免使智能体陷入谷底不动。`a1`是这部分的系数。
* `a2 * max(position - p0,0)`：小车越靠近终点`0.5`，就应当获得越多的奖励。`p0`为阈值位置，选择在`0-0.2`比较合适。`a2`是这部分的系数。
* `a3 * velocity`：总的来讲，向右移动更有利于小车到达终点，尤其是在后期小车的最高位置接近终点时（或者说，现在的奖励对于向左或者向右的速度给予的即时回报是相同的），因此加上此项可以促进小车向右到达终点。`a3`是这部分的系数。

这里，`base = -10`，按重要程度`a2 > a1 > a3`。

事实证明，优化后的奖励能使模型更快收敛。

#### 3.2.2 MountainCarContinuous-v0
`MountainCarContinuous-v0`最大的问题是智能体易陷入谷底不动，因此奖励函数依然需要修饰，修饰的方法与上面类似，只不过没有设置`base`，而是在原奖励函数的基础上进行叠加。事实证明，优化后的奖励能使智能体免于陷入不动的状态。

## 4. 实验总结
### 4.1 性能比较
模型训练的结果见附录部分。

将Q-learning与DQN进行比较后发现，DQN的收敛速度和要远快于Q-learning（小于200steps的第一个epsiode分别为12和92），但最终模型的性能比较接近，都是稳定在200-300steps之间完成任务。而对于DDPG，算法的收敛速度和性能均更优（小于200steps和小于100steps的第一个epsiode分别为27和40，模型稳定输出在100steps左右）。

考虑到连续动作空间要远远比离散动作空间复杂，则模型总体性能DDPG > DQN > Q-learning。
### 4.2 实验体会
在当前网络结构的基础上：
* 折现因子$\gamma$越大越好
* $\varepsilon$在0.1-0.15间比较合适，可以逐次衰减
* $\alpha$学习率初始值可以设置的较大，随后逐步衰减，但不能太小。
* 神经网络的性能对Target网络更新频率、经验回放的缓冲区总大小、训练的batch大小比较敏感：Target网络更新频率不宜过慢；缓冲区大小要适当；batch越大，模型越稳定。

## 5. 参考文献
### 5.1 Q-learning 
* 模型原理与源码参考：[Deep Q-Learning - 李理的博客](http://fancyerii.github.io/books/dqn/#dqn%E7%AE%97%E6%B3%95)

### 5.2 DQN
* 模型原理：
	*  [Deep Q-Learning - 李理的博客](http://fancyerii.github.io/books/dqn/#dqn%E7%AE%97%E6%B3%95)
	* [学习DQN（Deep Q-Learning Network） | 冰蓝记录思考的地方](http://lanbing510.info/2018/07/17/DQN.html)
* 源码参考：
	* [TensorFlow 2.0 (八) - 强化学习 DQN 玩转 gym Mountain Car | 极客兔兔](https://geektutu.com/post/tensorflow2-gym-dqn.html)

### 5.3 DDPG
* 模型原理：
	* [强化学习——策略梯度与Actor-Critic算法 - 知乎](https://zhuanlan.zhihu.com/p/36494307)
	* [强化学习(十三) 策略梯度(Policy Gradient) - 刘建平Pinard - 博客园](https://www.cnblogs.com/pinard/p/10137696.html)
	* [强化学习(十四) Actor-Critic - 刘建平Pinard - 博客园](https://www.cnblogs.com/pinard/p/10272023.html)
	* [强化学习(十六) 深度确定性策略梯度(DDPG) - 刘建平Pinard - 博客园](https://www.cnblogs.com/pinard/p/10345762.html)
	* [Deep Reinforcement Learning - 1. DDPG原理和算法_kenneth_yu的博客-CSDN博客](https://blog.csdn.net/kenneth_yu/article/details/78478356)
* 源码参考：
	* [RL_experiment/ref_DDPG.py at master · minghchen/RL_experiment](https://github.com/minghchen/RL_experiment/blob/master/ref_DDPG.py)
	* [keras-ddpg/ac_pendulum.py at master · piotrplata/keras-ddpg](https://github.com/piotrplata/keras-ddpg/blob/master/ac_pendulum.py)

## 附录一：程序配置/运行环境/版本号
-  `system`: `MacOS X 10.15.2, Build 19C57`
-  `Anaconda3`:`4.8.1`
-  `Jupyter Notebook`: `1.0.0`
-  `Python`: `3.7.6`
	- `gym` :`0.15.4`
	- `numpy`:`1.18.1`
	- `tensorflow`:`1.13.1`
## 附录二：MountainCar-v0 Q-learning 运行结果
```
----------
alpha = 1.0
Episode 1 completed in 8218 steps
----------
alpha = 1.0
Episode 2 completed in 3513 steps
----------
alpha = 1.0
Episode 3 completed in 1259 steps
----------
alpha = 1.0
Episode 4 completed in 4773 steps
----------
alpha = 1.0
Episode 5 completed in 2647 steps
----------
alpha = 0.8
Episode 6 completed in 4681 steps
----------
alpha = 0.8
Episode 7 completed in 4769 steps
----------
alpha = 0.8
Episode 8 completed in 1330 steps
----------
alpha = 0.8
Episode 9 completed in 904 steps
----------
alpha = 0.8
Episode 10 completed in 569 steps
----------
alpha = 0.6400000000000001
Episode 11 completed in 705 steps
----------
alpha = 0.6400000000000001
Episode 12 completed in 889 steps
----------
alpha = 0.6400000000000001
Episode 13 completed in 587 steps
----------
alpha = 0.6400000000000001
Episode 14 completed in 819 steps
----------
alpha = 0.6400000000000001
Episode 15 completed in 567 steps
----------
alpha = 0.5120000000000001
Episode 16 completed in 643 steps
----------
alpha = 0.5120000000000001
Episode 17 completed in 786 steps
----------
alpha = 0.5120000000000001
Episode 18 completed in 502 steps
----------
alpha = 0.5120000000000001
Episode 19 completed in 806 steps
----------
alpha = 0.5120000000000001
Episode 20 completed in 520 steps
----------
alpha = 0.4096000000000001
Episode 21 completed in 472 steps
----------
alpha = 0.4096000000000001
Episode 22 completed in 348 steps
----------
alpha = 0.4096000000000001
Episode 23 completed in 530 steps
----------
alpha = 0.4096000000000001
Episode 24 completed in 471 steps
----------
alpha = 0.4096000000000001
Episode 25 completed in 543 steps
----------
alpha = 0.3276800000000001
Episode 26 completed in 412 steps
----------
alpha = 0.3276800000000001
Episode 27 completed in 740 steps
----------
alpha = 0.3276800000000001
Episode 28 completed in 358 steps
----------
alpha = 0.3276800000000001
Episode 29 completed in 352 steps
----------
alpha = 0.3276800000000001
Episode 30 completed in 538 steps
----------
alpha = 0.2621440000000001
Episode 31 completed in 1270 steps
----------
alpha = 0.2621440000000001
Episode 32 completed in 675 steps
----------
alpha = 0.2621440000000001
Episode 33 completed in 336 steps
----------
alpha = 0.2621440000000001
Episode 34 completed in 334 steps
----------
alpha = 0.2621440000000001
Episode 35 completed in 406 steps
----------
alpha = 0.20971520000000007
Episode 36 completed in 310 steps
----------
alpha = 0.20971520000000007
Episode 37 completed in 322 steps
----------
alpha = 0.20971520000000007
Episode 38 completed in 309 steps
----------
alpha = 0.20971520000000007
Episode 39 completed in 318 steps
----------
alpha = 0.20971520000000007
Episode 40 completed in 576 steps
----------
alpha = 0.1677721600000001
Episode 41 completed in 335 steps
----------
alpha = 0.1677721600000001
Episode 42 completed in 258 steps
----------
alpha = 0.1677721600000001
Episode 43 completed in 280 steps
----------
alpha = 0.1677721600000001
Episode 44 completed in 313 steps
----------
alpha = 0.1677721600000001
Episode 45 completed in 255 steps
----------
alpha = 0.13421772800000006
Episode 46 completed in 397 steps
----------
alpha = 0.13421772800000006
Episode 47 completed in 429 steps
----------
alpha = 0.13421772800000006
Episode 48 completed in 245 steps
----------
alpha = 0.13421772800000006
Episode 49 completed in 328 steps
----------
alpha = 0.13421772800000006
Episode 50 completed in 323 steps
----------
alpha = 0.10737418240000006
Episode 51 completed in 351 steps
----------
alpha = 0.10737418240000006
Episode 52 completed in 303 steps
----------
alpha = 0.10737418240000006
Episode 53 completed in 316 steps
----------
alpha = 0.10737418240000006
Episode 54 completed in 318 steps
----------
alpha = 0.10737418240000006
Episode 55 completed in 310 steps
----------
alpha = 0.08589934592000005
Episode 56 completed in 342 steps
----------
alpha = 0.08589934592000005
Episode 57 completed in 246 steps
----------
alpha = 0.08589934592000005
Episode 58 completed in 348 steps
----------
alpha = 0.08589934592000005
Episode 59 completed in 251 steps
----------
alpha = 0.08589934592000005
Episode 60 completed in 268 steps
----------
alpha = 0.06871947673600004
Episode 61 completed in 344 steps
----------
alpha = 0.06871947673600004
Episode 62 completed in 286 steps
----------
alpha = 0.06871947673600004
Episode 63 completed in 277 steps
----------
alpha = 0.06871947673600004
Episode 64 completed in 333 steps
----------
alpha = 0.06871947673600004
Episode 65 completed in 332 steps
----------
alpha = 0.054975581388800036
Episode 66 completed in 384 steps
----------
alpha = 0.054975581388800036
Episode 67 completed in 247 steps
----------
alpha = 0.054975581388800036
Episode 68 completed in 304 steps
----------
alpha = 0.054975581388800036
Episode 69 completed in 245 steps
----------
alpha = 0.054975581388800036
Episode 70 completed in 263 steps
----------
alpha = 0.043980465111040035
Episode 71 completed in 301 steps
----------
alpha = 0.043980465111040035
Episode 72 completed in 258 steps
----------
alpha = 0.043980465111040035
Episode 73 completed in 282 steps
----------
alpha = 0.043980465111040035
Episode 74 completed in 226 steps
----------
alpha = 0.043980465111040035
Episode 75 completed in 241 steps
----------
alpha = 0.03518437208883203
Episode 76 completed in 336 steps
----------
alpha = 0.03518437208883203
Episode 77 completed in 224 steps
----------
alpha = 0.03518437208883203
Episode 78 completed in 250 steps
----------
alpha = 0.03518437208883203
Episode 79 completed in 241 steps
----------
alpha = 0.03518437208883203
Episode 80 completed in 340 steps
----------
alpha = 0.028147497671065624
Episode 81 completed in 284 steps
----------
alpha = 0.028147497671065624
Episode 82 completed in 271 steps
----------
alpha = 0.028147497671065624
Episode 83 completed in 287 steps
----------
alpha = 0.028147497671065624
Episode 84 completed in 322 steps
----------
alpha = 0.028147497671065624
Episode 85 completed in 333 steps
----------
alpha = 0.022517998136852502
Episode 86 completed in 241 steps
----------
alpha = 0.022517998136852502
Episode 87 completed in 242 steps
----------
alpha = 0.022517998136852502
Episode 88 completed in 254 steps
----------
alpha = 0.022517998136852502
Episode 89 completed in 314 steps
----------
alpha = 0.022517998136852502
Episode 90 completed in 247 steps
----------
alpha = 0.018014398509482003

Episode 91 completed in 235 steps
----------
alpha = 0.018014398509482003
Episode 92 completed in 194 steps
----------
alpha = 0.018014398509482003
Episode 93 completed in 281 steps
----------
alpha = 0.018014398509482003
Episode 94 completed in 238 steps
----------
alpha = 0.018014398509482003
Episode 95 completed in 331 steps
----------
alpha = 0.014411518807585602
Episode 96 completed in 242 steps
----------
alpha = 0.014411518807585602
Episode 97 completed in 229 steps
----------
alpha = 0.014411518807585602
Episode 98 completed in 246 steps
----------
alpha = 0.014411518807585602
Episode 99 completed in 232 steps
----------
alpha = 0.014411518807585602
Episode 100 completed in 234 steps
----------
alpha = 0.011529215046068483
Episode 101 completed in 197 steps
----------
alpha = 0.011529215046068483
Episode 102 completed in 235 steps
----------
alpha = 0.011529215046068483
Episode 103 completed in 244 steps
----------
alpha = 0.011529215046068483
Episode 104 completed in 244 steps
----------
alpha = 0.011529215046068483
Episode 105 completed in 271 steps
----------
alpha = 0.009223372036854787
Episode 106 completed in 311 steps
----------
alpha = 0.009223372036854787
Episode 107 completed in 255 steps
----------
alpha = 0.009223372036854787
Episode 108 completed in 288 steps
----------
alpha = 0.009223372036854787
Episode 109 completed in 279 steps
----------
alpha = 0.009223372036854787
Episode 110 completed in 248 steps
----------
alpha = 0.00737869762948383
Episode 111 completed in 282 steps
----------
alpha = 0.00737869762948383
Episode 112 completed in 229 steps
----------
alpha = 0.00737869762948383
Episode 113 completed in 233 steps
----------
alpha = 0.00737869762948383
Episode 114 completed in 335 steps
----------
alpha = 0.00737869762948383
Episode 115 completed in 274 steps
----------
alpha = 0.005902958103587064
Episode 116 completed in 244 steps
----------
alpha = 0.005902958103587064
Episode 117 completed in 333 steps
----------
alpha = 0.005902958103587064
Episode 118 completed in 161 steps
----------
alpha = 0.005902958103587064
Episode 119 completed in 243 steps
----------
alpha = 0.005902958103587064
Episode 120 completed in 241 steps
----------
alpha = 0.004722366482869652
Episode 121 completed in 225 steps
----------
alpha = 0.004722366482869652
Episode 122 completed in 217 steps
----------
alpha = 0.004722366482869652
Episode 123 completed in 231 steps
----------
alpha = 0.004722366482869652
Episode 124 completed in 272 steps
----------
alpha = 0.004722366482869652
Episode 125 completed in 238 steps
----------
alpha = 0.0037778931862957215
Episode 126 completed in 245 steps
----------
alpha = 0.0037778931862957215
Episode 127 completed in 327 steps
----------
alpha = 0.0037778931862957215
Episode 128 completed in 347 steps
----------
alpha = 0.0037778931862957215
Episode 129 completed in 244 steps
----------
alpha = 0.0037778931862957215
Episode 130 completed in 285 steps
----------
alpha = 0.0030223145490365774
Episode 131 completed in 240 steps
----------
alpha = 0.0030223145490365774
Episode 132 completed in 199 steps
----------
alpha = 0.0030223145490365774
Episode 133 completed in 236 steps
----------
alpha = 0.0030223145490365774
Episode 134 completed in 285 steps
----------
alpha = 0.0030223145490365774
Episode 135 completed in 272 steps
----------
alpha = 0.002417851639229262
Episode 136 completed in 248 steps
----------
alpha = 0.002417851639229262
Episode 137 completed in 201 steps
----------
alpha = 0.002417851639229262
Episode 138 completed in 198 steps
----------
alpha = 0.002417851639229262
Episode 139 completed in 201 steps
----------
alpha = 0.002417851639229262
Episode 140 completed in 238 steps
----------
alpha = 0.0019342813113834097
Episode 141 completed in 341 steps
----------
alpha = 0.0019342813113834097
Episode 142 completed in 237 steps
----------
alpha = 0.0019342813113834097
Episode 143 completed in 238 steps
----------
alpha = 0.0019342813113834097
Episode 144 completed in 202 steps
----------
alpha = 0.0019342813113834097
Episode 145 completed in 203 steps
----------
alpha = 0.0015474250491067279
Episode 146 completed in 198 steps
----------
alpha = 0.0015474250491067279
Episode 147 completed in 235 steps
----------
alpha = 0.0015474250491067279
Episode 148 completed in 248 steps
----------
alpha = 0.0015474250491067279
Episode 149 completed in 251 steps
----------
alpha = 0.0015474250491067279
Episode 150 completed in 236 steps
```
## 附录三：MountainCar-v0 DQN 运行结果
```
Episode 1 completed in 5465 steps
----------
alpha = 1.0
Episode 2 completed in 973 steps
----------
alpha = 1.0
Episode 3 completed in 2146 steps
----------
alpha = 1.0
Episode 4 completed in 1692 steps
----------
alpha = 1.0
Episode 5 completed in 3905 steps
----------
alpha = 0.8
Episode 6 completed in 965 steps
----------
alpha = 0.8
Episode 7 completed in 5903 steps
----------
alpha = 0.8
Episode 8 completed in 1221 steps
----------
alpha = 0.8
Episode 9 completed in 1592 steps
----------
alpha = 0.8
Episode 10 completed in 790 steps
----------
alpha = 0.6400000000000001
Episode 11 completed in 507 steps
----------
alpha = 0.6400000000000001
Episode 12 completed in 181 steps
----------
alpha = 0.6400000000000001
Episode 13 completed in 208 steps
----------
alpha = 0.6400000000000001
Episode 14 completed in 203 steps
----------
alpha = 0.6400000000000001
Episode 15 completed in 298 steps
----------
alpha = 0.5120000000000001
Episode 16 completed in 285 steps
----------
alpha = 0.5120000000000001
Episode 17 completed in 209 steps
----------
alpha = 0.5120000000000001
Episode 18 completed in 203 steps
----------
alpha = 0.5120000000000001
Episode 19 completed in 203 steps
----------
alpha = 0.5120000000000001
Episode 20 completed in 189 steps
----------
alpha = 0.4096000000000001
Episode 21 completed in 197 steps
----------
alpha = 0.4096000000000001
Episode 22 completed in 203 steps
----------
alpha = 0.4096000000000001
Episode 23 completed in 206 steps
----------
alpha = 0.4096000000000001
Episode 24 completed in 281 steps
----------
alpha = 0.4096000000000001
Episode 25 completed in 326 steps
----------
alpha = 0.3276800000000001
Episode 26 completed in 328 steps
----------
alpha = 0.3276800000000001
Episode 27 completed in 237 steps
----------
alpha = 0.3276800000000001
Episode 28 completed in 251 steps
----------
alpha = 0.3276800000000001
Episode 29 completed in 227 steps
----------
alpha = 0.3276800000000001
Episode 30 completed in 335 steps
----------
alpha = 0.2621440000000001
Episode 31 completed in 264 steps
----------
alpha = 0.2621440000000001
Episode 32 completed in 205 steps
----------
alpha = 0.2621440000000001
Episode 33 completed in 196 steps
----------
alpha = 0.2621440000000001
Episode 34 completed in 199 steps
----------
alpha = 0.2621440000000001
Episode 35 completed in 209 steps
----------
alpha = 0.20971520000000007
Episode 36 completed in 195 steps
----------
alpha = 0.20971520000000007
Episode 37 completed in 204 steps
----------
alpha = 0.20971520000000007
Episode 38 completed in 238 steps
----------
alpha = 0.20971520000000007
Episode 39 completed in 154 steps
----------
alpha = 0.20971520000000007
Episode 40 completed in 208 steps
----------
alpha = 0.1677721600000001
Episode 41 completed in 210 steps
----------
alpha = 0.1677721600000001
Episode 42 completed in 204 steps
----------
alpha = 0.1677721600000001
Episode 43 completed in 218 steps
----------
alpha = 0.1677721600000001
Episode 44 completed in 239 steps
----------
alpha = 0.1677721600000001
Episode 45 completed in 252 steps
----------
alpha = 0.13421772800000006
Episode 46 completed in 187 steps
----------
alpha = 0.13421772800000006
Episode 47 completed in 288 steps
----------
alpha = 0.13421772800000006
Episode 48 completed in 229 steps
----------
alpha = 0.13421772800000006
Episode 49 completed in 309 steps
----------
alpha = 0.13421772800000006
Episode 50 completed in 257 steps
----------
alpha = 0.10737418240000006
Episode 51 completed in 270 steps
----------
alpha = 0.10737418240000006
Episode 52 completed in 297 steps
----------
alpha = 0.10737418240000006
Episode 53 completed in 227 steps
----------
alpha = 0.10737418240000006
Episode 54 completed in 298 steps
----------
alpha = 0.10737418240000006
Episode 55 completed in 426 steps
----------
alpha = 0.08589934592000005
Episode 56 completed in 326 steps
----------
alpha = 0.08589934592000005
Episode 57 completed in 371 steps
----------
alpha = 0.08589934592000005
Episode 58 completed in 301 steps
----------
alpha = 0.08589934592000005
Episode 59 completed in 227 steps
----------
alpha = 0.08589934592000005
Episode 60 completed in 266 steps
----------
alpha = 0.06871947673600004
Episode 61 completed in 389 steps
----------
alpha = 0.06871947673600004
Episode 62 completed in 787 steps
----------
alpha = 0.06871947673600004
Episode 63 completed in 682 steps
----------
alpha = 0.06871947673600004
Episode 64 completed in 225 steps
----------
alpha = 0.06871947673600004
Episode 65 completed in 320 steps
----------
alpha = 0.054975581388800036
Episode 66 completed in 192 steps
----------
alpha = 0.054975581388800036
Episode 67 completed in 226 steps
----------
alpha = 0.054975581388800036
Episode 68 completed in 240 steps
----------
alpha = 0.054975581388800036
Episode 69 completed in 251 steps
----------
alpha = 0.054975581388800036
Episode 70 completed in 1618 steps
----------
alpha = 0.043980465111040035
Episode 71 completed in 459 steps
----------
alpha = 0.043980465111040035
Episode 72 completed in 218 steps
----------
alpha = 0.043980465111040035
Episode 73 completed in 1330 steps
----------
alpha = 0.043980465111040035
Episode 74 completed in 229 steps
----------
alpha = 0.043980465111040035
Episode 75 completed in 247 steps
----------
alpha = 0.03518437208883203
Episode 76 completed in 203 steps
----------
alpha = 0.03518437208883203
Episode 77 completed in 142 steps
----------
alpha = 0.03518437208883203
Episode 78 completed in 223 steps
----------
alpha = 0.03518437208883203
Episode 79 completed in 199 steps
----------
alpha = 0.03518437208883203
Episode 80 completed in 215 steps
----------
alpha = 0.028147497671065624
Episode 81 completed in 1110 steps
----------
alpha = 0.028147497671065624
Episode 82 completed in 396 steps
----------
alpha = 0.028147497671065624
Episode 83 completed in 172 steps
----------
alpha = 0.028147497671065624
Episode 84 completed in 210 steps
----------
alpha = 0.028147497671065624
Episode 85 completed in 237 steps
----------
alpha = 0.022517998136852502
Episode 86 completed in 165 steps
----------
alpha = 0.022517998136852502
Episode 87 completed in 242 steps
----------
alpha = 0.022517998136852502
Episode 88 completed in 258 steps
----------
alpha = 0.022517998136852502
Episode 89 completed in 683 steps
----------
alpha = 0.022517998136852502
Episode 90 completed in 166 steps
----------
alpha = 0.018014398509482003
Episode 91 completed in 156 steps
----------
alpha = 0.018014398509482003
Episode 92 completed in 215 steps
----------
alpha = 0.018014398509482003
Episode 93 completed in 209 steps
----------
alpha = 0.018014398509482003
Episode 94 completed in 206 steps
----------
alpha = 0.018014398509482003
Episode 95 completed in 226 steps
----------
alpha = 0.014411518807585602
Episode 96 completed in 198 steps
----------
alpha = 0.014411518807585602
Episode 97 completed in 191 steps
----------
alpha = 0.014411518807585602
Episode 98 completed in 251 steps
----------
alpha = 0.014411518807585602
Episode 99 completed in 293 steps
----------
alpha = 0.014411518807585602
Episode 100 completed in 263 steps
```
## 附录四：MountainCarContinuous-v0 DDPG 运行结果
```
Episode 1 completed in 1286 steps
Episode 2 completed in 1084 steps
Episode 3 completed in 6966 steps
Episode 4 completed in 617 steps
Episode 5 completed in 367 steps
Episode 6 completed in 1278 steps
Episode 7 completed in 3322 steps
Episode 8 completed in 668 steps
Episode 9 completed in 1684 steps
Episode 10 completed in 872 steps
Episode 11 completed in 496 steps
Episode 12 completed in 463 steps
Episode 13 completed in 758 steps
Episode 14 completed in 452 steps
Episode 15 completed in 414 steps
Episode 16 completed in 225 steps
Episode 17 completed in 571 steps
Episode 18 completed in 577 steps
Episode 19 completed in 259 steps
Episode 20 completed in 878 steps
Episode 21 completed in 277 steps
Episode 22 completed in 257 steps
Episode 23 completed in 378 steps
Episode 24 completed in 168 steps
Episode 25 completed in 340 steps
Episode 26 completed in 372 steps
Episode 27 completed in 119 steps
Episode 28 completed in 182 steps
Episode 29 completed in 253 steps
Episode 30 completed in 185 steps
Episode 31 completed in 1019 steps
Episode 32 completed in 408 steps
Episode 33 completed in 276 steps
Episode 34 completed in 188 steps
Episode 35 completed in 242 steps
Episode 36 completed in 254 steps
Episode 37 completed in 217 steps
Episode 38 completed in 347 steps
Episode 39 completed in 195 steps
Episode 40 completed in 93 steps
Episode 41 completed in 288 steps
Episode 42 completed in 311 steps
Episode 43 completed in 208 steps
Episode 44 completed in 151 steps
Episode 45 completed in 96 steps
Episode 46 completed in 87 steps
Episode 47 completed in 206 steps
Episode 48 completed in 91 steps
Episode 49 completed in 156 steps
Episode 50 completed in 246 steps
Episode 51 completed in 104 steps
Episode 52 completed in 300 steps
Episode 53 completed in 313 steps
Episode 54 completed in 563 steps
Episode 55 completed in 364 steps
Episode 56 completed in 156 steps
Episode 57 completed in 119 steps
Episode 58 completed in 135 steps
Episode 59 completed in 189 steps
Episode 60 completed in 109 steps
Episode 61 completed in 94 steps
Episode 62 completed in 146 steps
Episode 63 completed in 95 steps
Episode 64 completed in 205 steps
Episode 65 completed in 100 steps
Episode 66 completed in 115 steps
Episode 67 completed in 134 steps
Episode 68 completed in 234 steps
Episode 69 completed in 188 steps
Episode 70 completed in 205 steps
Episode 71 completed in 208 steps
Episode 72 completed in 173 steps
Episode 73 completed in 95 steps
Episode 74 completed in 93 steps
Episode 75 completed in 206 steps
Episode 76 completed in 163 steps
Episode 77 completed in 97 steps
Episode 78 completed in 204 steps
Episode 79 completed in 175 steps
Episode 80 completed in 246 steps
Episode 81 completed in 95 steps
Episode 82 completed in 173 steps
Episode 83 completed in 101 steps
Episode 84 completed in 140 steps
Episode 85 completed in 93 steps
Episode 86 completed in 1067 steps
Episode 87 completed in 835 steps
Episode 88 completed in 98 steps
Episode 89 completed in 284 steps
Episode 90 completed in 170 steps
Episode 91 completed in 98 steps
Episode 92 completed in 403 steps
Episode 93 completed in 185 steps
Episode 94 completed in 97 steps
Episode 95 completed in 192 steps
Episode 96 completed in 295 steps
Episode 97 completed in 102 steps
Episode 98 completed in 94 steps
Episode 99 completed in 87 steps
Episode 100 completed in 99 steps
```
