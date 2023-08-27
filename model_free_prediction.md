# Model-free predition

​	无模型的预测，相对于DP算法中对MDP过程中的状态转换以及奖励机制充分了解，**根本不知道MDP过程中的状态转换以及奖励机制**，预测即对价值函数的评估。

## MC方法（蒙特卡罗方法）

​	MC的说明事项：

* MC方法从**完整**的episodes学习，区别TD方法（时间差分）

* MC**没有bootstrapping**（自举获取数据学习）,使用真实的反馈奖励

* MC采用最简单的回报计算，用**完整回报取平均**

* MC是对包含大量随机成分的估计方法

  MC迭代评估价值函数的更新公式：
  $$
  \begin{align}
  v_{k+1}(S_{t})&=\frac{\sum_{i=1}^{N(S_t)}G_i}{N(S_t)}\\
  &=\frac{1}{N(S_t)} (G_t + \sum_{i=1}^{N(S_t)-1} G_i)\\
  &=\frac{1}{N(S_t)}(G_t + (N(S_t)-1)v_{k}(S_{t}))\\
  &=v_{k}(S_{t}) + \frac{1}{N(S_t)}(G_t-v_{k}(S_{t}) )
  \end{align}
  $$
  

前面为预估项，后面一项为误差项，**与TD方法最大的差别就在误差项的不同**。对于非稳定的问题，可以将取平均换成一个超参$\alpha$,
$$
v_{k+1}(S_{t})=v_{k}(S_{t}) + \alpha(G_t-v_{k}(S_{t}) )
$$
这种方法被称为$constant-\alpha$**MC方法**

​	MC方法又分为first-visit和every-visit两类

## TD learning（时间差分学习）

​	TD方法的说明：

* TD从片段的不完整的episodes学习

* TD采用bootstrapping，通过通过前面的猜测得到后面的猜测

* TD更能在MDP中**充分发掘过程的马尔可夫性质**

* TD采用**已经评估的价值函数**求取误差项

  TD迭代的更新公式：
  $$
  v_{k+1}(S_{t})=v_{k}(S_{t}) + \alpha(R_{t+1}+\gamma v_{k}(S_{t+1})-v_{k}(S_{t}) )
  $$
  

这种方法称为TD(0),还有n步时间差分方法，还有**结合资格迹(Eligibility Traces)发挥MC方法和动态规划的优势的TD(λ)**。

## 参考文献

1. [David Silver(UCL)的课件](https://www.davidsilver.uk/teaching/)
2. [RLbook2020_Sutton.pdf (incompleteideas.net)](http://incompleteideas.net/book/RLbook2020.pdf) 