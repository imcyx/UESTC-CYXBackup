# 题目

在对高维数据降维之前应先进行"中心化"，常见的是将协方差矩阵 $X X^{T}$ 转化为 $X H H^{T} X^{T}$ ，其中 $H=I-\frac{1}{m} 11^{T}$ ，试析其效果。



# 解答

答: 相当于将 $X$ 变为 $X^{\prime}$ :
$$
\begin{aligned}
X^{\prime} &=X H \\
&=X\left(I-\frac{1}{m} 11^{T}\right) \\
&=X-\frac{1}{m} X 11^{T} \\
&=X-\frac{1}{m}\left[\begin{array}{lll}
x_{1}, & x_{2}, & \cdots
\end{array}\right]\left[\begin{array}{c}
1 \\
1 \\
\vdots
\end{array}\right]\left[\begin{array}{lll}
1 & 1 & \cdots
\end{array}\right] \\
&=X-\frac{1}{m} \sum_{i} x_{i}\left[\begin{array}{lll}
1 & 1 & \cdots
\end{array}\right] \\
&=X-\bar{x}\left[\begin{array}{lll}
1 & 1 & \cdots
\end{array}\right] \\
&=\left[\begin{array}{lll}
x_{1}-\bar{x}, & x_{2}-\bar{x}, & \cdots
\end{array}\right]
\end{aligned}
$$
其效果便是中心化 $x_{i}^{\prime}=x_{i}-\bar{x}$ 。

