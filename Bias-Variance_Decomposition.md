# Bias-Variance Decomposition

在验证集上通过实际实验计算估计泛化误差，bias-variance 分解从概率统计的角度解释了泛化误差是由那些因素构成

令 $$f_D(x)$$ 为在训练集 $$D$$ 上建立的模型， $$h(x)=E(y|x)$$ 为我们希望得到的最优模型，代表输入和输出的真实关系，是已知 $$x$$ 的条件下，$$y$$ 的期望，则对于某个训练集 $$D$$ 及其 $$f_D(x)$$ 在测试集 $$Test=\{(x_1,y_1),(x_2,y_2),...,(x_t,y_t)\}$$ 上的泛化误差为：

<center>$$ \iint {[(y-f_D(x))^2]p(x,y)dxdy}$$ </center><br/>

对于多个不同的训练集 $$D$$，可计算出多个泛化误差，对这些泛化误差取均值：

<center>$$\begin{aligned}E(f_D)&=E_D \iint {[(y-f_D(x))^2](x,y)dxdy}\\&=E_D \iint {[(y-h(x)+h(x)-f_D(x))^2]p(x,y)dxdy}\\&=E_D \int {(f_D(x)-h(x))^2p(x)dx} + E_D \iint {(y-h(x))^2p(x,y)dxdy} \\ & +2E_D\iint [{(y-h(x)) \cdot  (h(x)-f_D(x))]p(x,y)dxdy}\\\end{aligned}$$ </center><br/>

对于第三项：

<center>$$\begin{aligned}
&\iint {[(y-h(x)) \cdot  (h(x)-f_D(x))]p(x,y)dxdy}\\
= & \iint {[y \cdot h(x) - y \cdot f_D(x) - h(x) \cdot h(x) + h(x) \cdot f_D(x)]p(x,y)dxdy}\\
= & \iint {y(h(x)-f_D(x))p(x,y)dxdy}-\iint {h(x)\cdot (h(x)-f_D(x))p(x,y)dxdy}\\
= & \iint {(h(x)-f_D(x))yp(y|x)p(x)dxdy}-\iint {h(x)\cdot (h(x)-f_D(x))p(x,y)dxdy}\\
= & \int {(h(x)-f_D(x))E(y|x)p(x)dx} - \int {(h(x)-f_D(x))h(x)p(x)dx} = 0\\
\end{aligned}$$<center><br/>

即

<center>$$E(f_D)=E_D \int {(f_D(x)-h(x))^2p(x)dx} + E_D \iint {(y-h(x))^2p(x,y)dxdy}$$ </center><br/>

对于第一项：

<center>$$\begin{aligned}
&E_D \int {(f_D(x)-h(x))^2p(x)dx}\\
=& E\int {[(h(x)-E(f_D(x))+E(f_D(x))-f_D(x))^2p(x)]dx}\\
\end{aligned}$$<center><br/>

对于第三项：

<center>$$\begin{aligned}
& E\int [(h(x)-E(f_D(x))) \cdot (E(f_D(x))-f_D(x))p(x)]dx\\
=& E\int [h(x)E(f_D(x))p(x)]dx-E \int [h(x)f_D(x)p(x)]dx-E\int [E(f_D(x))E(f_D(x))p(x)]dx+E \int [E(f_D(x))f_D(x)p(x)]dx\\
=& (E\int [h(x)E(f_D(x))p(x)]dx-E \int [h(x)f_D(x)p(x)]dx)-(E \int [E(f_D(x))E(f_D(x))p(x)]dx-E \int [E(f_D(x))f_D(x)p(x)]dx)\\
=& 0-0 = 0\\
\end{aligned}$$<center><br/>

其中 $$E\int [h(x)E(f_D(x))p(x)]dx-E\int [h(x)f_D(x)p(x)]dx=0$$ 可能有点难以理解，需要注意 $$E(f_D(x))$$ 是个条件期望，而且外层的 $$E[\dots]$$ 是在训练集合 $$D$$ 上取期望，跟 $$x$$ 无关，调换一下积分顺序就能发现两式相等，实在不行可以将 $$E\int [h(x)f_D(x)p(x)]dx$$ 全部拆开，写成积分或者求和的形式，合并 $$\sum\limits_D {p(D)f_D(x)}=E(f_D(x))$$，即可得证。

因此

<center>$$\begin{aligned}
E(f_D)=&E[\int {(h(x)-E(f_D(x)))^2(x)dx}] + E[\int (E(f_D(x))-f_D(x))^2p(x)dx]+E_D \iint {(y-h(x))^2p(x,y)dxdy}\\
=&\int {(h(x)-E(f_D(x)))^2p(x)dx}+\int E[(f_D(x)-E(f_D(x)))^2]p(x)dx + \iint {(y-h(x))^2p(x,y)dxdy}\\
=&bias^2{f_D(x)}+var{f_D(x)}+var{noise}\\ \end{aligned}$$<center><br/>

另一个对偏差方差分解的一个更好的[推导](https://blog.csdn.net/zhulf0804/article/details/54314683)。

>偏差度量了学习算法的期望预测与真实结果的偏离程度，即刻画了学习算法本身的拟合能力；方差度量了同样大小的训练集的变动所导致的学习性能的变化，即刻画了数据扰动所造成的影响；噪声则表达了在当前任务上任何学习算法所能达到的期望泛化误差的下界，即刻画了学习问题本身的难度，偏差-方差分解说明，泛化性能是由学习算法的能力、数据的充分性以及学习任务本身的难度所共同决定的，给定学习任务，为了取得较好的泛化性能，则需使偏差较小，即能够充分拟合数据，并且使方差较小，即使得数据扰动产生的影响小。 -- 周志华，机器学习

# High Bias or High Variance？

当训练误差和验证集误差相近且都比较大时，模型处于高 bias 状态。当训练误差较小而验证集误差远大于训练误差时，模型处于高 variance 状态。
一些注意事项：

  * 训练样本和测试样本的分布不一致容易导致高 bias 。
  * 高 bias 发生时，改进模型，或者训练方法。
  * 高 variance 发生时，增加训练数据集。

# Bias Variance Tradeoff

一般来说，偏差与方差是有冲突的，这称为 __ Bias-Variance 窘境__，对于某个学习任务，假定我们可以控制学习算法的训练程度，则在训练不足时，学习器的拟合能力不够强，对“真实”的掌握还不够，训练数据的扰动不足以使学习器产生显著变化，此时偏差主导泛化误差，随着训练程度的加深，学习器的拟合能力逐渐加强，训练数据发生的扰动渐渐能被学习器学习到，方差逐渐主导了泛化误差，在训练程度充足后，学习器的拟合能力已非常强，训练数据发生的轻微扰动都会导致学习器发生显著变化，若训练数据自身的，非全局的特性被学习器学习到了，则将发生过拟合。

参数 $$\lambda$$ 控制模型的复杂度，$$\lambda$$ 越大，模型越简单，学习能力越差；$$\lambda$$ 越小，模型越复杂，学习能力越强。

一般的，在模型优化目标的正则化项中，如果正则化系数 $$\lambda$$ 过高，模型容易处于高 bias 状态；$$\lambda$$ 过低，模型容易处于高 variance 状态。

均衡偏差方差的一些套路，出自 Understanding the Bias-Variance Tradeoff

大部分人可能本能的倾向于高 variance ，低 bias ，虽然高 variance 会给出一个长期平均意义上不错的结果，但是实际上建模者一般只会由一个数据集来训练模型，重要的是模型性能，因此尽量不要为了一方而过度牺牲另一方。

Bagging 和 其他再采样技术可以用来减少模型预测中的 variance ，通过多次有放回采样，产生多个训练集，在每个训练集上训练出模型，预测时取各个模型的平均，可大大降低方差（当每个模型相互独立时，$$D(\frac1n\sum\limits_{i=1}^n X_i )=\frac1{n^2}\sum\limits_{i=1}^nD(X_i)$$ ，典型算法如随机森林。

使用渐近性质的算法(大致意思好像是，当训练样本数趋于无穷的时候，偏差能趋近于零的算法，渐进效率可能指的是偏差随训练样本增加而下降的速度)，数据不充分的情况下，抛弃理论。

理解偏差和方差对于理解预测模型的行为非常重要，但总的来说，真正需要关心的是总体的泛化误差，而不是具体的分解，偏差-方差的均衡点是位于随着复杂度的变化，偏差的增减恰好抵消方差减增的点，但在实践中，没有一个分析方法可以直接找到这个点，因此，必须使用准确的估计泛化误差，并探索不同的模型复杂度，然后选择合适的复杂度级别，最小化总体误差，尽量优先采用交叉验证等基于再采样的方法估计泛化误差。
