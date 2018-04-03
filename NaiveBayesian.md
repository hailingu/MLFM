# Naive Bayesian

naive bayesian 基于 bayesian 定理：

<center>$$P(A|B)=\frac{P(B|A)P(A)}{P(B)}$$<center><br/>

其中 $$A$$ 和 $$B$$ 是事件，并且 $$P(B) \neq 0$$。

$$P(A|B)$$ 是条件概率，给出当事件 $$B$$ 发生的时候，事件 $$A$$ 发生的概率有多大，$$P(B|A)$$ 与之类似。

在使用概率的工具的时候要避免直觉上的一些错误认知。例如 wiki 中给出的一个判断一个人是否嗑药的例子。已知一个检测是否嗑药的测试的 sensitive 率为 $$99%$$ ， specific 率也为 $$99%$$，也就是说假设有 $$100$$ 个人嗑药，那么能找出其中的 $$99$$ 个； $$100$$ 个人没有嗑药的人，有 $$1$$ 个人可能会被判断成磕了药。假设所有的人群中有 $$0.5%$$ 的人是嗑药的人群，那么从所有的人中随便挑一个人，这个人是属于嗑药人群的概率是多少？

<center> $$\begin{array}\ P(User|+) & = & \frac{P(+|User)P(User)}{P(+)} \\
& = & \frac{P(+|User)P(User)}{P(+|User)P(User) + P(+ | Non-user)P(Non-user)} \\ & = & \frac{0.99 * 0.005}{0.99 * 0.005 + 0.01 * 0.995}  \approx   33.2 \% \end{array}$$ <center><br/>

可以发现即使是这么准确的测试，从一大堆数据中找到异常数据的概率也是很低的。

另一个需要了解的知识就是联合概率的 chain rule：

<center>$$\begin{array}\ P(C_k, x_1, \ldots, x_n) = & P(x_1 | x_2, \ldots, x_n, C_k) P(x_2, \ldots, x_n, C_k) \\ = & P(x_1 | x_2, \ldots, x_n, C_k) P(x_2 | x_3, \ldots, x_n, C_k) \\ = & \dots \\ = & P(x_1 | x_2, \ldots, x_n, C_k) \ldots P(x_{n-1} | x_n, C_k) P(x_n | C_k) P(C_K) \end{array} $$</center><br/>

在 naive bayesian 中假定每个 Feature 之间是__相互独立的__，所以就存在如下关系：

<center> $$P(x_i | x_{i+1}, \ldots, C_k) = P(x_i | C_k)$$ <center><br/>

那么整合起来就是：

<center>$$\begin{array}\ P(C_k|x_1, \ldots, x_n) = & \frac{P(x_1, \ldots, x_n, C_k)}{P(x_1, \ldots, x_n)} = & \frac{P(x_1|C_k) \ldots P(x_n | C_k) P(C_k)}{P(x_1) \ldots P(x_n)} \\ \end{array} $$</center><br/>

上面这个式子，其实就是 naive bayesian 了。用稍微正式点的方式来描述就是：
