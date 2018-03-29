# Bias-Variance Decomposition

在验证集上通过实际实验计算估计泛化误差，bias-variance 分解从概率统计的角度解释了泛化误差是由那些因素构成

令 $$f_D(x)$$ 为在训练集 $$D$$ 上建立的模型， $$h(x)=E(y|x)$$ 为我们希望得到的最优模型，代表输入和输出的真实关系，是已知 $$x$$ 的条件下，$$y$$ 的期望，则对于某个训练集 $$D$$ 及其 $$f_D(x)$$ 在测试集 $$Test=\{(x_1,y_1),(x_2,y_2),...,(x_t,y_t)\}$$ 上的泛化误差为：

<center>$$ \iint {[(y-f_D(x))^2]p(x,y)dxdy}$$ </center><br/>

对于多个不同的训练集 $$D$$，可计算出多个泛化误差，对这些泛化误差取均值：

<center>$$\begin{aligned}E(f_D)&=E_D \iint {[(y-f_D(x))^2](x,y)dxdy}\\&=E_D \iint {[(y-h(x)+h(x)-f_D(x))^2]p(x,y)dxdy}\\&=E_D \int {(f_D(x)-h(x))^2p(x)dx} + E_D \iint {(y-h(x))^2p(x,y)dxdy} \\ & +2E_D\iint [{(y-h(x)) \cdot  (h(x)-f_D(x))]p(x,y)dxdy}\\\end{aligned}$$ </center><br/>

对于第三项：

<center>$$\begin{aligned}
&\iint {[(y-h(x)) \cdot  (h(x)-f_D(x))]p(x,y)dxdy}\\
&=\iint {[y \cdot h(x) - y \cdot f_D(x) - h(x) \cdot h(x) + h(x) \cdot f_D(x)]p(x,y)dxdy}\\
&=\iint {y(h(x)-f_D(x))p(x,y)dxdy}-\iint {h(x)\cdot (h(x)-f_D(x))p(x,y)dxdy}\\
&=\iint {(h(x)-f_D(x))yp(y|x)p(x)dxdy}-\iint {h(x)\cdot (h(x)-f_D(x))p(x,y)dxdy}\\
&=\int {(h(x)-f_D(x))E(y|x)p(x)dx} - \int {(h(x)-f_D(x))h(x)p(x)dx}\\
&=0
\end{aligned}$$<center><br/>

即

<center>$$E(f_D)=E_D \int {(f_D(x)-h(x))^2p(x)dx} + E_D \iint {(y-h(x))^2p(x,y)dxdy}$$ </center><br/>

对于第一项：

<center>$$\begin{aligned}
&E_D \int {(f_D(x)-h(x))^2p(x)dx}\\
&=E\int {[(h(x)-E(f_D(x))+E(f_D(x))-f_D(x))^2p(x)]dx}\\
\end{aligned}$$<center><br/>

对于第三项：

<center>$$\begin{aligned}
&E\int [(h(x)-E(f_D(x))) \cdot (E(f_D(x))-f_D(x))p(x)]dx\\
&=E\int [h(x)E(f_D(x))p(x)]dx-E \int [h(x)f_D(x)p(x)]dx-E\int [E(f_D(x))E(f_D(x))p(x)]dx+E \int [E(f_D(x))f_D(x)p(x)]dx\\
&=(E\int [h(x)E(f_D(x))p(x)]dx-E \int [h(x)f_D(x)p(x)]dx)-(E \int [E(f_D(x))E(f_D(x))p(x)]dx-E \int [E(f_D(x))f_D(x)p(x)]dx)\\
&=0-0\\
&=0
\end{aligned}$$<center><br/>
