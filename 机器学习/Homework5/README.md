# 题目

5.1 实现反向传播算法来计算神经网络误差函数的梯度

![Q1](.\imgs\Q1.png)

5.2 利用BP神经网络实现鲍鱼的性别分类（雄、雌和幼崽）

数据集：数据包括鲍鱼性别、大小、重量等9个属性。

> 训练集：abalone_train.data
>
> 测试集：abalone_test.data

工具/平台：

> 工具：python、java等语言。
>
> 建议：可使用TensorFlow（或Pytorch）框架，在TensorFlow中用张量表示数据，用计算图搭建神经网络，
> 用会话执行计算图，优化线上的权重（参数），最后得到模型。

# 解答

5.1[代码](BP-Algorithm.py) ，程序运行结果如下：

![Classification_Dataset_Visualization](.\imgs\Classification_Dataset_Visualization.png)

![Classification_Dataset_Visualization](.\imgs\Accuracy.png)

![Classification_Dataset_Visualization](.\imgs\MSE_Loss.png)

5.2 [代码](main.py)
