<font size=6>第六章作业：</font>

  		1. 根据SVM公式，本章第一题为限定条件下求极值：

$$
w^T*x+b=\pm1\\
\min_{w,b}{1 \over 2}|| w ||^2
$$

带入题目中数据，进而可以得到下列四个式子：前三个式子换元后对第四个式子求极值，最终得出运算结果：
$$
1*w_1+2*w_2+3*w_3+b = +1\\
4*w_1+1*w_2+2*w_3+b = +1\\
(-1)*w_1+2*w_2+(-1)*w_3+b = +1\\
\min_{w,b}{1 \over 2}|| w ||^2
$$
编程代码为：[hyperplane.py](.\hyperplane.py)，运算结果如图所示：

<img src=".\Q1_res_1.png" alt="Q1_res_1" style="zoom:115%;" />

<img src=".\Q1_res_2.jpg" alt="Q1_res_2" style="zoom:25%;" />

2. 编程代码为：[svm.py](.\svm.py)

