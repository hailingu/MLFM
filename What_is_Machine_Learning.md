# 什么是机器学习
前面的内容简要的讲了机器学习的历史和一些实际使用中的机器学习问题，但是至今为止，我都没有写下什么是机器学习。跟什么是 AI 不一样，机器学习是有一个比较权威的定义的：


> 对于某类任务 T 和性能度量 P ，如果一个计算机程序在 T 上以 P 衡量的性能随着经验 E 而自我完善，那么我们称这个计算机程序从经验 E 中学习。 -- Tom Mitchell, Machine

上面的定义中 T 就是机器学习模型要解决的问题， P 就是衡量这个机器学习模型性能的评价方式， E 就是经常提到的数据。这个定义很好的体现了机器学习的过程及核心要素。

这些年来，研究和工程人员一直致力于改善的就是这三个方面：

1. 不断的创造新的模型解决新的问题。
2. 不断的提出新的模型评价方式。
3. 不断的增加用于训练的有效数据。

从如今的具体实践来看，这样做很有效。上面的 1 和 2 适用于研究人员，而 3 则是适用于工程人员，创造新的模型和新的模型评价方式不是一件容易的事情，可是增加用于训练的有效数据往往就能直接的改善原有模型的效果。可换句话来说，这三个部分要付出
的成本都很高，因为现在更多有效的模型就是 Supervised Learning ，大量的数据意味着需要大量对数据做出准确的标注，这需要花费很大的人力和时间的成本，当然这也就等价于付出了金钱的成本。现如今，很多公司（例如百度、Amazon、讯飞）构建了用于标注
数据的数据标注平台，通过付出一定的金钱吸引人们用业余时间来标注公司自己或者他人上传的数据，以用于构建模型。近年来，研究人员也越来越多的从学术界转到拥有资金和数据的企业，存储和处理这些数据需要大量的机器和资金，这是学校往往不具备的，可是
企业能提供这样的环境。

## 机器学习的分类

根据具体要处理的问题的类型，机器学习主要分成了两个大类：

1. Supervised Learning
2. Unsupervised Learning

每个大类之下又有一些具体的算法，这些具体的算法暂时不表，稍后随着章节的进行，可以自行进行体会。

Supervised Learning 常常有人翻译成“监督学习”。在解释什么是 Supervised Learning 之前先得说明数据的表示。通常，我们使用（暂时的）行向量来表示一个数据，向量的每一个分量都称为一个 Feature，例如可以通过（姓名，性别，年龄，学历，家庭住址）
来表示一个人，其中的每一个分量都是一个 Feature 。 如果这个时候，我们能认为这个向量表示一个人的收入的水平，即“收入=（姓名，性别，年龄，学历，家庭住址）”，那么“收入”这个变量就称之为 Label。**如果机器学习模型构建的是 Feature 和 Label 之间的关系，那么这个模型就属于 _Supervised Learning_**。

类似的，Unsupervised Learning的解释就很容易了，**如果机器学习模型构建的数据上的输入只有 Feature 没有 Label，那么这个模型也就是发现数据之间的关系，这样的模型就属于 _Unsupervised Learning_**。

有些资料会把 Label 表述成 Response Variable，会把 Feature 表输出 Attribute , 其实都是一个意思。 