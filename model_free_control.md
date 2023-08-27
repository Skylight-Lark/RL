# Model-free control

​	由于状态价值函数与动作没有关系，而control需要选择最佳的策略，即根据当前的状态来选择动作，因而价值函数**选择动作价值函数**进行更新。

​	contorl**主要框架是先策略评估，再策略改进**。

## MC control

​	策略评估：使用动作价值函数为更新的对象，然后类似之前的prediction一样进行更新迭代，收敛之目标动作价值函数。

​	策略改进：可以使用greedy策略，但由于考虑到不能充分探索到环境中的所有动作-状态对，可以采用**ε-greedy策略**

​	如果想要加快迭代的效率或者较小计算资源，可以采用每完成一次episode就可以评估与改进，而不需要等所有的episode一起sample，从而加快速度。同时还可以采用的**衰减的ε-greedy的GLIE MC control方法**，确保最后一定收敛至最优的动作价值函数。

## TD control-sarsa and Q-learning

​	on-policy TD control-sarsa的更新公式：
$$
Q_{k+1}(S_t,A_t) = Q_k(S_t,A_t) + \alpha (R_{t+1} + \gamma Q_k(S_{t+1}, A_{t+1}) - Q_k(S,A))
$$
​	off-policy TD onctrol-Q-learning的更新公式：
$$
Q_{k+1}(S_t,A_t) = Q_k(S_t,A_t) + \alpha (R_{t+1} + \gamma \max_{a} Q_k(S_{t+1}, a) - Q_k(S_t,A_t))
$$
on-policy的意思是：需要学习的是动作价值函数Q本身，下一步动作的选择取决于需要学习的动作价值函数，也就是说**与生成agent决策的行动策略有关**。

off-policy的意思是：学习目标是动作价值函数Q对最优动作价值函数$q_{\star}$的直接近似，**与用于生成智能体决策序列轨迹的动作策略无关**，也就是说可以学习多个旧的或者其他的策略，甚至可以从人类专家的示范策略中学习，更加灵活，学习效率更高。

TD control也可以使用MC control的后面的一些方法，加快计算。

## 参考文献

1. [David Silver(UCL)的课件](https://www.davidsilver.uk/teaching/)
2. [RLbook2020_Sutton.pdf (incompleteideas.net)](http://incompleteideas.net/book/RLbook2020.pdf) 