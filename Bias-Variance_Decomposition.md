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
