# Multilayer Perceptron

Perceptron 前面已经提到过了， Multilayer Perceptron 和 Perceptron 的区别就在于它是通过多层的 Perceptron 堆叠而成的，所以 Perceptron 是它的基础。在现在的很多资料中，都会提到 Multilayer Perceptron 是一种前馈人工神经网络，所以很多时候 Multilayer Perceptron 和 Nuerual Network 都会混在一起。

Perceptron 有一个不能解决问题，它不能解决 XOR 这样的问题。 XOR 问题说的是，有两个 Feature ($$\mathbf{x} =(x_1,x_2)$$)， 其中的每一个 Feature 的取值范围为 $$\{0，1\}$$ , 其对应的 Label 在 Feature 取不同值的时候为 1，相同值的时候为 0。如果在图像上画出这个整个数据集，很容易验证不能找到一根直线，把 Label 根据 Feature 取值分开：

![f7.0.png](assets/f7.0.png)

仔细看 XOR 问题，可以发现它完全是一个二元分类问题，但是不能通过简单的线性分类器解决，所以 Perceptron 的使用条件中写到了数据集一定要是__线性可分__的，在 XOR 问题上，Perceptron 失效了。面对这个问题，简单直接的想法就是既然线性函数不行，那么就采用非线性的办法吧，如果这么想就会遇到另一个问题：_选择什么样的非线性函数来解决呢_？。
