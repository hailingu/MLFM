# Linear Regression

Linear Regression 就是为了找到数据的 Feature 和对应 Label 之间的线性关系，这里的线性关系可以是“广义”上的线性关系，“广义”就是说模型的输入可以是直接是数据的 Feature，也可以是 Feature 经过函数变换后的结果
。“ Regression ”的意思是说建立的模型输出不是一个类别变量，而是一个连续的、与原始的 Label 尽可能接近的实数。

最简单的线性回归是一元线性回归，即只有数据的 Feature 数量只有 1 个。可以很容易构造出这样的数据集，例如：

      import numpy as np

      def f(x):
          return 3*x + 2

      noise = np.random.uniform(-1, 1, 100)
      X = np.random.uniform(0, 1, 100)
      Y = f(X) + noise

上面的构造出的数据集画成散点图：

![f3.0.png](assets/f3.0.png)

代码里面定义的函数的结构是 $$ 3 \times x + 2 $$，然后加入了一个高斯随机噪声，如果没有随机噪声，这里得到的所有的点都会呈现一条直线。线性回归要做的就是通过这些散落的点，找出里面的存在的线性关系。

# Model

线性回归的模型很简单，即

<center>$$y= \mathbf{w}^T \cdot \mathbf{x} +b$$</center><br/>

下面要做的和之前的 Perceptron 一样，根据数据找到合适的 $$w$$ 和 $$b$$。这里要说明的是，从这一章节开始，黑体用来表示列向量，如上面的 \\(\mathbf{w} = \begin{bmatrix} w_1 \\\ w_2 \\\ \cdots \\\ w_k \end{bmatrix}\\)，
数据向量 \\(\mathbf{x}\\) 也类似。

如果把上面的 \\( \mathbf{x} \\) 更换成 \\((f_1(x_1), f_2(x_2), \ldots, f_k(x_k))\\) ，那么就变成了前面提到的广义线性模型。

# Learning Algorithm
这次的数据集合是由上面的代码生成的，所以这里就省去数据集说明的部分。
