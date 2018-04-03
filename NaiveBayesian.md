# Naive Bayesian

naive bayesian 基于 bayesian 定理：

<center>$$P(A|B)=\frac{P(B|A)P(A)}{P(B)}$$<center><br/>

其中 $$A$$ 和 $$B$$ 是事件，并且 $$P(B) \neq 0$$。

$$P(A|B)$$ 是条件概率，给出当事件 $$B$$ 发生的时候，事件 $$A$$ 发生的概率有多大，$$P(B|A)$$ 与之类似。

在使用概率的工具的时候要避免直觉上的一些错误认知。例如 wiki 中给出的一个判断一个人是否嗑药的例子。已知一个检测是否嗑药的测试的 sensitive 率为 $$99%$$ ， specific 率也为 $$99%$$，也就是说假设有 $$100$$ 个人嗑药，那么能找出其中的 $$99$$ 个； $$100$$ 个人没有嗑药的人，有 $$1$$ 个人可能会被判断成磕了药。假设所有的人群中有 $$0.5%$$ 的人是嗑药的人群，那么从所有的人中随便挑一个人，这个人是属于嗑药人群的概率是多少？

<enter> $$\begin{array} P(User|+) = & \frac{P(+|User)P(User)}{P(+)} \\
= & \frac{P(+|User)P(User)}{P(+|User)P(User) + P(+ | Non-user)P(Non-user)} \\ = & \frac{0.99 * 0.005}{0.99 * 0.005 + 0.01 * 0.995}  \approx  = 33.2 % \end{array}$$ <center><br/>

可以发现，
