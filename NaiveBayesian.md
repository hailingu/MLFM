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

上面这个式子，其实就是 naive bayesian 了。完成 naive bayesian 需要做的几件事情:

* 计算每一个 Feature 的每一个取值 $$P(x_i|C_k)$$ 的值。
* 计算每一个 Feature 的每一个取值 $$P(x_i)$$ 的值。
* 计算 $$P(C_k)$$ 的值，由于 $$P(C_k)$$ 是一个先验的概率值，事先并不知道，可以统计训练的数据中 $$P(C_k)$$ 做为先验的概率。

计算完成上面的结果后，对于一个新到来的数据根据其输入的 Feature 利用 naive bayesian 计算其属于每一个 $$C_k$$ 的概率，然后选择最大的 $$C_k$$ 作为其最后的分类。

通过前面的描述中，细心的人可能发现了 naive bayesian 对于连续的数据似乎不那么友好，的确，在使用这个方法的时候需要事先对那些取值连续的 Feature 进行离散化，然后再使用 navie bayesian 算法。同时，由于 navie bayesian 中，每一次分类判断的时候分母都是一样的，所以在实际的计算中，可以忽略掉分母，只计算分子的部分。还需要注意的是，在计算 $$P(C_k), P(x_i|C_k)$$ 的时候很有可能出现为结果为 0 的情况下，如果这个分类真的不可能存在结果为 0 是正常的，但是我们不是上帝视角，无法知道某个通过上述计算方法计算出来的分类是不是真的不存在，这个时候就会采用一个 Laplace smoothing 的技术，使得 $$P(C_k), P(x_i|C_k)$$ 不为 0 ：

<center>$$ P(x_i = a_i|C_k) = \frac{\sum I(x_i = a_i, C_k) + \lambda}{\sum I(C_k) + m \lambda}$$</center><br/>

其中 $$m$$ 代表 Feature 的数量。

在 navie bayesian 的分子中 $$P(x_1|C_k) \ldots P(x_n | C_k) P(C_k)$$ 可以被视为 likelihood ， 而 $$P(C_k)$$ 可以被视为先验概率，那么这样 naive bayesian 又可以表述成：

<center>$$P(C_k|\mathbf{X})=\frac{likelihood \times prior}{evidence}$$</center><br/>

所以最后选择的那个分类，其实是看到实际的 Feature 后的最大后验分类。写成上面的式子还有一个好处，可以用其他的概率分布来替换上面使用的古典概率模型的 likelihood ，比如可以假设 $$P(x | C_k) = \frac{1}{\sqrt{2 \pi \sigma_k^2}} exp {- \frac{(x-\mu_k)^2}{2 \sigma_k^2}} $$ 是 Gaussian Distribution，当然还可以用其他的分布来替换这部分，替换后要做的就是计算对应分布的参数估计。
