# Logistic Regression

Logistic Regression 是一个二元的分类模型，[wiki](https://en.wikipedia.org/wiki/Logistic_regression) 说它并不是一个分类器，而是一个概率模型，这个我个人持保留意见。人们在实际的工作和生活中使用 Logistic Regression 的时候更多的时候还是当一个分类器来使用。

简单点来说， Logistic Regression 将建立起一组变量表达的二元标签 $$(0，1)$$ 之间的关系。更加直观的表现的是，如果有一个数据集有标签 $$y$$ 和五个 Feature $$x_1, x_2, x_3, x_4, x_5$$ ，那么 Logistic Regression 可以用来建立起 x 和 y 之间的关系。

Logistic Regression 有很多的应用，比如用来预测美国大选大家投的是民主党还是共和党，比如在医学上有人用于受伤的人是否要死了，比如在广告中用来预测这个广告是否要投放给某个人，比如在营销中用于判断某个人是否会订阅或者购买某个产品，亦如判断一个肿瘤是良性的还是恶性的。所有这里提到的例子，都是建立起一个一组变量表达的二元标签之间的关系。

Logistic Regression 是一个很简单的模型，但也是很有效的模型，至今在很多的项目中使用到了 Logistic Regression ， 不过现在的用法不是那么的简单和直接，人们会花费更多的时间构建数据的表示上。这里第一次提到数据的表示，需要多说几句。不同的数据表示对于模型的构建影响很大，例如有的数据集在 Polar 坐标系下仅仅用简单的直线就可以分开，但是在 Cartesian 坐标系下却需要使用圆才能分开。这样的话，前者只需要构建一个线性的模型，但是后者需要构建一个非线性模型。人们在线性模型的构建上积累了很多的知识、经验、方法，但面对非线性的问题的时候，常常束手无策。所以数据的表示成为了现在工程实践中一个很重要的步骤，这个步骤再加上额外的一些对数据的处理（例如去除噪声点、插值、数据标准化等），就构成了人们口中常说的__特征工程__。

特征工程加上简单模型是过去以至于现在很多问题的主流处理方法。

# Model

Logistic Regression 的模型很简单，即将一个线性模型的结果作为 sigmoid 函数的输入，将 sigmoid 函数的输出作为最终的结果：

<center>$$\begin{array}\ tmp = \mathbf{w}^T \cdot \mathbf{x} \\ y = \frac{1}{1 + exp(-tmp)}  \end{array}$$</center><br/>

在机器学习中，常把 $$g(x) = \frac{1}{1+e^{-x}}$$ 这样的函数叫做 sigmoid 函数，它能将整个实数集映射到 $$[0,1]$$ 区间上：

![f5.0.png](assets/f5.0.png)