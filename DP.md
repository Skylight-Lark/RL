# DP（动态规划）

​	**动态规划使用价值函数来结构化组织对最优策略的搜索。**因而动态规划的解决问题的两个特性：

* 最优的子结构：
  * 最优化的理论和原则可以适用
  * 可以将问题分解为若干子问题
* 具有类似形式的子问题：
  * 子问题循环出现
  * 子问题可以被存储和重复使用

## 策略评估

​	通过迭代的方式，对计算繁琐或者维度很高的问题的简化，通过迭代收敛到最终的价值函数形式。**利用$v_{\pi}$的贝尔曼方程构建迭代公式**进行更新：
$$
v_{k+1}(s)=\sum_{a} \pi(a|s) \sum_{s',r}p(s',r|s,a)[r+\gamma v_{k}(s')]
$$
​	常见的有两种形式：

1. 用两个数组存储新旧的价值函数，用旧的价值函数更新新的价值函数。类似数值分析中的雅各比迭代
2. 就地更新，更新的新的价值函数与还未更新的价值函数一起更新价值函数。类似高斯赛德尔迭代

## 策略迭代

​	通过不断地先策略评估得到较优的价值函数，再策略改进得到较优策略，循环收敛之最优的策略和价值函数。策略改进的证明：

对于$\pi ,\pi'$两个确定的策略，如果：
$$
q_{\pi}(s,\pi'(s)) \geqslant v_{\pi}(s)
$$
那么则可以确定后者比前者好，即：
$$
v_{\pi'}(s) \geqslant v_{\pi}(s)
$$

$$
\begin{align}
proof:\\
v_{\pi}(s)  &\leqslant q_{\pi}(s,\pi'(s))\\
&=\mathbb{E}_{\pi'}[R_{t+1}+\gamma v_{\pi}(S_{t+1})|S_t = s]\\
&\leqslant \mathbb{E}_{\pi'}[R_{t+1}+\gamma q_{\pi}(S_{t+1},\pi'(S_{t+1}))|S_t = s]\\
&= \mathbb{E}_{\pi'}[R_{t+1}+\gamma \mathbb{E}_{\pi'}[R_{t+2}+\gamma v_{\pi}(S_{t+2})|S_{t+1} = s]|S_t = s]\\
&= \mathbb{E}_{\pi'}[R_{t+1}+\gamma R_{t+2} + \gamma^2  v_{\pi}(S_{t+2})|S_t = s]\\
&\leqslant \mathbb{E}_{\pi'}[R_{t+1}+\gamma R_{t+2} + \gamma^2  R_{t+3}  + \cdots|S_t = s]\\
&=v_{\pi'}(s)
\end{align}

$$
当新的策略和原有的策略一样好，就会得到最优的贝尔曼方程，此即最优的策略

## 价值迭代

​	直接利用贝尔曼最优方程作为更新规则对价值函数进行迭代，当收敛时就直接满足贝尔曼最优方程，此时为最优策略。更新方程：
$$
v_{k+1}=\max_{a}\sum_{s',r}p(s',r|s,a)[r+\gamma v_{k}(s')]
$$

## 参考文献

1. [David Silver(UCL)的课件](https://www.davidsilver.uk/teaching/)
2. [RLbook2020_Sutton.pdf (incompleteideas.net)](http://incompleteideas.net/book/RLbook2020.pdf) 
3. 上述三种迭代方案的伪代码均可在Sutton的书中对应章节中有