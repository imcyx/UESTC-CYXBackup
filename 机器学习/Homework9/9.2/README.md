# 题目

假设数据由混合专家(mixture of experts)模型生成，即数据是基于$k$个成分混合的概率密度生成：$p(x|\theta ) = \sum\limits_{i = 1}^k {{a_i} \cdot p(x|{\theta _i})}$，其中 $\theta  = \{ {\theta _1},{\theta _2}, \cdots ,{\theta _k}\}$ 是模型参数，$p(x|{\theta _i})$是第$i$个混合成分的概率密度，混合系数 ${\alpha _i} \ge 0{\kern 1pt} ,{\kern 1pt} \sum\limits_{i = 1}^k {{\alpha _i}=1}$ 。假设每个混合成分对应一种类别，但每个类别可能包含多个混合成分。试推导出生成式半监督学习算法。

# 解答

答：首先需要假定：

>1. 数据集 $X$ 包括 $M$ 个样本：$$X = \{ {x_j}\} ,j = 1, \cdots ,M$$其中 $l$ 个标记样本，$u$ 个未标记样本：$$M=l+u$$
>2. 样本里共包括 $\left| C \right|$ 个类别：$${y_j} \in C$$
>3. 混合模型含有$N$个混合成分，样本$X_j$可能的混合成分由$m_j$表示：$$\{ {m_j} = i\} ,i=1, \cdots ,N$$若${\theta _i}$表示对应混合成分的模型参数，则对应模型可表示为：$$\textcolor{blue}{f({x_j}|{\theta _i}) = p({x_j}|{m_j} = i,{\theta _i}) = p({x_j}|{\theta _i})}$$

## 最大似然估计

针对给定标记样本集${D_l} = \{ ({x_1},{y_1})\;,\;({x_2},{y_2})\;,\; \cdots \;,\;({x_l},{y_l})\} $和未标定样本集$${D_u} = \{ {x_{l + 1}}\;,\;{x_{l + 2}}\;,\; \cdots \;,\;{x_u}\} $$。用极大似然法来估计高斯混合模型的参数$\{ ({\alpha _i},{\mu _i},{\Sigma _i})|1 \le i \le N\} $，${D_l} \cup {D_u}$的对数似然是：
$$
\begin{aligned}
LL({D_l} \cup {D_u}) 
&= \sum\limits_{(xi,yj){\kern 1pt} {\kern 1pt}  \in {\kern 1pt} {\kern 1pt} {D_l}} {\ln p({x_j},{y_j}{\kern 1pt} \mid {\kern 1pt} {\kern 1pt} \theta )}  + \sum\limits_{{x_i} \in {\kern 1pt} {\kern 1pt} {D_u}} {\ln p({x_j}{\kern 1pt} \mid {\kern 1pt} {\kern 1pt} \theta )}\\
&=\sum_{\left(\mathbf{x}_{i}, c_{j}\right) \in D_{l}} \ln \sum_{i=1}^{N} \alpha_{i} p\left(c_{j} \mid \mathbf{x}_{j}, m_{j}=i, \theta_{i}\right) p\left(\mathbf{x}_{j} \mid m_{j}=i, \theta_{i}\right)+\sum_{\mathbf{x}_{i} \in D_{u}} \ln \sum_{i=1}^{N} \alpha_{i} p\left(\mathbf{x}_{j} \mid m_{j}=i, \theta_{i}\right)\\
&=\sum_{\left(\mathbf{x}_{i}, c_{j}\right) \in D_{l}} \ln \sum_{i=1}^{N} \alpha_{i} p\left(c_{j} \mid \mathbf{x}_{j}, m_{j}=i, \theta_{i}\right) f\left(\mathbf{x}_{j} \mid \theta_{i}\right)+\sum_{\mathbf{x}_{i} \in D_{u}} \ln \sum_{i=1}^{N} \alpha_{i} f\left(\mathbf{x}_{j} \mid \theta_{i}\right)
\end{aligned}
\tag{1}
$$
接下来介绍一下题目中所说的 **每个类别可包含多个混合成分**的混合模型的具体表示：

> 首先，我们知道:
> $$
> p\left(m_{j}=i \mid \mathbf{x}_{j}\right)=\frac{\alpha_{i} \cdot p\left(\mathbf{x}_{j} \mid \theta_{i}\right)}{\sum_{i=1}^{N} \alpha_{i} \cdot p\left(\mathbf{x}_{j} \mid \theta_{i}\right)}\tag{2}
> $$
> 根据(D. J. Miller and H. s. Uyar, 1996)的观点，主要有两种混合方法:
>
> **划分混合模型(The “Partitioned" Mixture Model, PM):**
>
> 混合组分与各个类别具有硬划分的关系，即 $M_{i} \in C_{k}$ ，其中 $M_{i}$ 代表混合组分 $i$ ，也就是说各个类别是由特定的混合组分组合而成， $C_{k}$ 代表类别 $k$ 具有的混合组分形成的集合，混合模型后验概率为：
> $$
> p\left(c_{j}=k \mid \mathbf{x}_{j}\right)=\frac{\sum_{i=1 \wedge M_{i} \in C_{k}}^{N} \alpha_{i} \cdot p\left(\mathbf{x}_{j} \mid \theta_{i}\right)}{\sum_{i=1}^{N} \alpha_{i} \cdot p\left(\mathbf{x}_{j} \mid \theta_{i}\right)}\tag{3}
> $$
> **广义混合模型(The Generalized Mixture Model, GM):**
>
> 每个混合组分 $i \in\{1, \ldots, K\}$ 都有可能是形成某个类别 $k$ 的一个混合成分，定义：
> $$
> p\left(c_{j} \mid m_{j}, \mathbf{x}_{j}\right)=p\left(c_{j} \mid m_{j}\right)=\beta_{c_{j} \mid m_{j}}\tag{4}
> $$
> 其中第二项成立是因为 $\beta_{c_{j} \mid m_{j}}$ 与具体的 $\mathbf{x}_{j}$ 取值无关。在此基础上可知，混合模型后验概率为:
> $$
> p\left(c_{j} \mid \mathbf{x}_{j}\right)=\frac{\sum_{i=1}^{N}\left(\alpha_{i} \cdot p\left(\mathbf{x}_{j} \mid \theta_{i}\right)\right) \beta_{c_{j} \mid i}}{\sum_{i=1}^{N} \alpha_{i} \cdot p\left(\mathbf{x}_{j} \mid \theta_{i}\right)}\tag{5}
> $$
> 显然，令 GM中真正属于 $c_{j}$ 的混合成分 $i$ 均为 $\beta_{c j \mid i}=1$ ，其他 $\beta_{c_{j \mid i}}=0$ ，则此时广义混合模型退化为 PM $_{e}$

在这里，我们采用 $\mathrm{GM}$ ，采用高斯分布作为混合成分，来推导 `EM` 算法的更新参数。

显然，此时：
$$
f\left(\mathbf{x}_{j} \mid \theta_{i}\right)=p\left(\mathbf{x}_{j} \mid \theta_{i}\right)=p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)\tag{*}
$$
则 $(1)$ 变为:
$$
L L\left(D_{l} \cup D_{u}\right)=\sum_{\left(\mathbf{x}_{i}, c_{j}\right) \in D_{l}} \ln \sum_{i=1}^{N} \alpha_{i} p\left(c_{j} \mid \mathbf{x}_{j}, m_{j}=i, \mu_{i}, \Sigma_{i}\right) p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)+\sum_{\mathbf{x}_{i} \in D_{u}} \ln \sum_{i=1}^{N} \alpha_{i} p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)\tag{6}
$$
 $(4)$ 带入 $(6)$ 可得:
$$
L L\left(D_{l} \cup D_{u}\right)=\sum_{\left(\mathbf{x}_{i}, c_{j}\right) \in D_{l}} \ln \sum_{i=1}^{N} \alpha_{i} \beta_{c_{j} \mid i} p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)+\sum_{\mathbf{x}_{i} \in D_{u}} \ln \sum_{i=1}^{N} \alpha_{i} p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)\tag{7}
$$
我们的目的是要求得最优的 $\alpha_{i}, \beta_{c j \mid i}, \mu_{i}, \Sigma_{i}$ 使上式 $(7)$ 取得最大值。

> 在这里，依据对数据完整性的不同看法，可有两种 EM 算法:
>
> <font color=red>**EM-1(假定不含类标记):**</font>
>
> 对于 $\left(\mathbf{x}_{j}, c_{j}\right) \in D_{l}, \mathbf{x}_{j} \in D_{u}$, 均缺乏混合成分 $m_{j}$ 信息，相应的完整数据为 $\left\{\left(\mathbf{x}_{j}, c_{j}, m_{j}\right)\right\}$ 和 $\left\{\left(\mathbf{x}_{j}, m_{j}\right)\right\}$ ，也就是说不用推断 $\mathbf{x}_{j} \in D_{u}$ 的类标记。
>
> <font color=red>**EM-II(假定含类标记):**</font>
>
> 对于 $D_{l}$ 定义同上，但对于 $\mathbf{x}_{j} \in D_{u}$ ，认定其缺少 $m_{j}, c_{j}$ ，因此对应于 $\mathbf{x}_{j} \in D_{u}$ 的完整数据为 $\left\{\left(\mathbf{x}_{j}, c_{j}, m_{j}\right)\right\}$ ，也就是说既要推断 $\mathbf{x}_{j} \in D_{u}$ 的类标记，还要推断 $\mathbf{x}_{j} \in D_{u}$ 的混合成分。



## EM-I

对于混合系数 $\alpha_{i}$, 除了要最大化 $L L\left(D_{l} \cup D_{u}\right)$ ，还应满足隐含条件: $\alpha_{i} \geq 0, \sum_{i=1}^{N} \alpha_{i}=1$ ， 因此考虑对 $L L\left(D_{l} \cup D_{u}\right)$ 使用拉格朗日乘子法，变为优化：
$$
L L\left(D_{l} \cup D_{u}\right)+\lambda\left(\sum_{i=1}^{N} \alpha_{i}-1\right)\tag{8}
$$
将 $(7)$ 带入 $(8)$ ，并令 $(8)$ 对 $\alpha_{i}$ 的导数为 0 ，得到：
$$
\frac{\partial L L\left(D_{l} \cup D_{u}\right)}{\partial \alpha_{i}}=\sum_{\mathbf{x} j \in D_{l}} \frac{\beta_{c j \mid i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)}{\sum_{i=1}^{N} \alpha_{i} \cdot \beta_{c j \mid i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)}+\sum_{\mathbf{x}_{j} \in D_{u}} \frac{p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)}{\sum_{i=1}^{N} \alpha_{i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)}+\lambda=0\tag{9}
$$
令：
$$
p\left(m_{j}=i \mid c_{j}, \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right)=\frac{\alpha_{i} \cdot \beta_{c j i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)}{\sum_{i=1}^{N} \alpha_{i} \cdot \beta_{c j \mid i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)}\tag{10}
$$
同时，将高斯模型 $(*)$ 带入 $(2)$ 可得：
$$
p\left(m_{j}=i \mid \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right)=\frac{\alpha_{i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)}{\sum_{i=1}^{N} \alpha_{i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)}\tag{11}
$$
对 $(9)$ 两边同时乘以 $\alpha_{i}$ ，将 $(10),(11)$ 代入可得：
$$
0=\sum_{\mathrm{x} j \in D_{l}} p\left(m_{j}=i \mid c_{j}, \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right)+\sum_{\mathrm{x} j \in D_{u}} p\left(m_{j}=i \mid \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right)+\alpha_{i} \cdot \lambda\tag{12}
$$
令 $(12)$  对所有高斯混合成分求和：
$$
\begin{aligned}
0 &=\sum_{\mathrm{x}_{j} \in D_{l}} \sum_{i=1}^{N} p\left(m_{j}=i \mid c_{j}, \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right)+\sum_{\mathrm{x} j \in D_{u}} \sum_{i=1}^{N} p\left(m_{j}=i \mid \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right)+\alpha_{i} \cdot \lambda \\
&=\sum_{\mathbf{x}_{j} \in D_{l}} 1+\sum_{\mathbf{x}_{j} \in D_{u}} 1+\lambda \\
&=M+\lambda
\end{aligned}\tag{13}
$$
由 $(13)$ 可得， $\lambda=-M$ ，将其带入 $(12)$ 可得:
$$
\alpha_{i}=\frac{1}{M} \cdot\left(\sum_{\mathrm{x} j \in D_{l}} p\left(m_{j}=i \mid c_{j}, \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right)+\sum_{\mathbf{x}_{j} \in D_{u}} p\left(m_{j}=i \mid \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right)\right)\tag{14}
$$
对于高斯分布，其偏导具有如下性质：
$$
&\frac{\partial p\left(\mathbf{x} \mid \mu_{i}, \Sigma_{i}\right)}{\partial \mu_{i}}=p\left(\mathbf{x} \mid \mu_{i}, \Sigma_{i}\right) \cdot \Sigma_{i}^{-1} \cdot\left(\mu_{i}-\mathbf{x}\right)\tag{15} \\
$$
$$
&\frac{\partial p\left(\mathbf{x} \mid \mu_{i}, \Sigma_{i}\right)}{\partial \Sigma_{i}}=p\left(\mathbf{x} \mid \mu_{i}, \Sigma_{i}\right) \cdot \Sigma_{i}^{-2} \cdot\left(\left(\mathbf{x}-\mu_{i}\right)\left(\mathbf{x}-\mu_{i}\right)^{\top}-\Sigma_{i}\right) \\ \tag{16}
$$

求 $(7)$ 对 $\mu_{i}$ 的偏导，结合 $(15),(10),(11)$ 可得:
$$
\begin{aligned}
\frac{\partial L L\left(D_{l} \cup D_{u}\right)}{\partial \mu_{i}} &=\sum_{\mathbf{x}_{j} \in D_{l}} \frac{\alpha_{i} \cdot \beta_{c j i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)}{\sum_{i=1}^{N} \alpha_{i} \cdot \beta_{c j \mid i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)} \cdot \Sigma_{i}^{-1} \cdot\left(\mu_{i}-\mathbf{x}_{j}\right)+\sum_{\mathbf{x}_{j} \in D_{u}} \frac{\alpha_{i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)}{\sum_{i=1}^{N} \alpha_{i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)} \cdot \Sigma_{i}^{-1} \cdot\left(\mu_{i}-\mathbf{x}_{j}\right) \\
&=\sum_{\mathbf{x} j \in D_{l}} p\left(m_{j}=i \mid c_{j}, \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right) \cdot \Sigma_{i}^{-1} \cdot\left(\mu_{i}-\mathbf{x}_{j}\right)+\sum_{\mathbf{x}_{j} \in D_{u}} p\left(m_{j}=i \mid \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right) \cdot \Sigma_{i}^{-1} \cdot\left(\mu_{i}-\mathbf{x}_{j}\right) \\
&=\Sigma_{i}^{-1} \cdot\left(\sum_{\mathbf{x}_{j} \in D_{l}} p\left(m_{j}=i \mid c_{j}, \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right) \cdot\left(\mu_{i}-\mathbf{x}_{j}\right)+\sum_{\mathbf{x}_{j} \in D_{u}} p\left(m_{j}=i \mid \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right) \cdot\left(\mu_{i}-\mathbf{x}_{j}\right)\right)
\end{aligned}\tag{17}
$$
令 $(17)=0$ ，将 $(14)$ 带入可得:
$$
\mu_{i}=\frac{1}{M \alpha_{i}} \cdot\left(\sum_{x_{j} \in D_{l}} \mathbf{x}_{j} \cdot p\left(m_{j}=i \mid c_{j}, \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right)+\sum_{\mathbf{x}_{j} \in D_{u}} \mathbf{x}_{j} \cdot p\left(m_{j}=i \mid \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right)\right)\tag{18}
$$
同样地，求 $(7)$ 对 $\Sigma_{i}$ 的偏导，结合 $(16),(10),(11)$ 可得:
$$
\begin{aligned}
\frac{\partial L L\left(D_{l} \cup D_{u}\right)}{\partial \Sigma_{i}}=& \sum_{\mathbf{x}_{j} \in D_{l}} \frac{\alpha_{i} \cdot \beta_{c j \mid i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)}{\sum_{i=1}^{N} \alpha_{i} \cdot \beta_{c j \mid i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)} \cdot \Sigma_{i}^{-2} \cdot\left(\left(\mathbf{x}_{j}-\mu_{i}\right)\left(\mathbf{x}_{j}-\mu_{i}\right)^{\top}-\Sigma_{i}\right) \\
&+\sum_{\mathbf{x}_{j} \in D_{u}} \frac{\alpha_{i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)}{\sum_{i=1}^{N} \alpha_{i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)} \cdot \Sigma_{i}^{-2} \cdot\left(\left(\mathbf{x}_{j}-\mu_{i}\right)\left(\mathbf{x}_{j}-\mu_{i}\right)^{\top}-\Sigma_{i}\right) \\
=& \sum_{\mathbf{x}_{j} \in D_{l}} p\left(m_{j}=i \mid c_{j}, \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right) \cdot \Sigma_{i}^{-2} \cdot\left(\left(\mathbf{x}_{j}-\mu_{i}\right)\left(\mathbf{x}_{j}-\mu_{i}\right)^{\top}-\Sigma_{i}\right) \\
&+\sum_{\mathbf{x}_{j} \in D_{u}} p\left(m_{j}=i \mid \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right) \cdot \Sigma_{i}^{-2} \cdot\left(\left(\mathbf{x}_{j}-\mu_{i}\right)\left(\mathbf{x}_{j}-\mu_{i}\right)^{\top}-\Sigma_{i}\right)
\end{aligned}\tag{19}
$$
令 $(19)=0$ ，将 $(14)$ 带入可得:
$$
\begin{aligned}
\Sigma_{i}=\frac{1}{M \alpha_{i}} & \cdot\left(\sum_{\mathbf{x}_{j} \in D_{l}} p\left(m_{j}=i \mid c_{j}, \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right) \cdot\left(\left(\mathbf{x}_{j}-\mu_{i}\right)\left(\mathbf{x}_{j}-\mu_{i}\right)^{\top}\right)\right.\\
+&\left.\sum_{\mathbf{x} j \in D_{u}} p\left(m_{j}=i \mid \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right) \cdot\left(\left(\mathbf{x}_{j}-\mu_{i}\right)\left(\mathbf{x}_{j}-\mu_{i}\right)^{\top}\right)\right)
\end{aligned}\tag{20}
$$
对于系数 $\beta_{k \mid i}$ ，除了要最大化 $L L\left(D_{l} \cup D_{u}\right)$ ，还应满足隐含条件: $\beta_{k \mid i} \geq 0, \sum_{k=1}^{|\mathcal{C}|} \beta_{k \mid i}=1$ ，因此考慮对 $L L\left(D_{l} \cup D_{u}\right)$ 使用拉格朗日乘子法，变为优化：
$$
L L\left(D_{l} \cup D_{u}\right)+\lambda\left(\sum_{k=1}^{|\mathcal{C}|} \beta_{k \mid i}-1\right)\tag{21}
$$
将 $(7)$ 带入 $(21)$ ，并令 $(21)$ 对 $\beta_{k \mid i}$ 的导数为 $0$ ，得到：
$$
\frac{\partial L L\left(D_{l} \cup D_{u}\right)}{\partial \beta_{k \mid i}}=\sum_{\mathbf{x}_{j} \in D_{l} \wedge c j=k} \frac{\alpha_{i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)}{\sum_{i=1}^{N} \alpha_{i} \cdot \beta_{c j \mid i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)}+\lambda=0\tag{22}
$$
两边同时乘以 $\beta_{k \mid i}$, 结合 $(10)$ 得:
$$
\begin{aligned}
0 &=\sum_{\mathbf{x}_{j} \in D_{l} \wedge c_{j}-k} \frac{\alpha_{i} \cdot \beta_{k \mid i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)}{\sum_{i=1}^{N} \alpha_{i} \cdot \beta_{c j \mid i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)}+\beta_{k \mid i} \cdot \lambda \\
&=\sum_{\mathbf{x}_{j} \in D_{l} \wedge c j=k} p\left(m_{j}=i \mid c_{j}, \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right)+\beta_{k \mid i} \cdot \lambda
\end{aligned}\tag{23}
$$
令 $(23)$ 对所有卷标记求和:
$$
\begin{aligned}
0 &=\sum_{k=1}^{|\mathcal{C}|} \sum_{\mathbf{x} j \in D l \wedge c j-k} p\left(m_{j}=i \mid c_{j}, \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right)+\sum_{k=1}^{|\mathcal{C}|} \beta_{k \mid i} \cdot \lambda \\
&=\sum_{\mathbf{x}_{j} \in D_{l}} p\left(m_{j}=i \mid c_{j}, \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right)+\lambda
\end{aligned}\tag{24}
$$
即：
$$
\lambda=-\sum_{\mathbf{x}_{j} \in D_{l}} p\left(m_{j}=i \mid c_{j}, \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right)\tag{25}
$$
将 $(25)$ 带入 $(23)$ 可得:
$$
\beta_{k \mid i}=\frac{\sum_{\mathbf{x} j \in D_{l} \wedge c j=k} p\left(m_{j}=i \mid c_{j}, \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right)}{\sum_{\mathbf{x} j \in D_{l}} p\left(m_{j}=i \mid c_{j}, \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right)}\tag{26}
$$

## EM-II

对于EM-II，由于需要预测 $\mathbf{x}_{j} \in D_{u}$ 下的 $c_{j}$ ，根据贝叶斯定理，$(7)$变为:
$$
\begin{aligned}
L L\left(D_{l} \cup D_{u}\right) &=\sum_{\left(\mathbf{x}_{i}, c j\right) \in D_{l}} \ln \sum_{i=1}^{N} \alpha_{i} \beta_{c j \mid i} p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)+\sum_{\mathbf{x}_{i} \in D_{u}} \ln \sum_{i=1}^{N} \alpha_{i} p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right) \\
&=\sum_{\left(\mathbf{x}_{i}, c_{j}\right) \in D_{l}} \ln \sum_{i=1}^{N} \alpha_{i} \beta_{c j \mid i} p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)+\sum_{\mathbf{x}_{i} \in D_{u}} \ln \sum_{i=1}^{N} \sum_{k=1}^{|\mathcal{C}|} \alpha_{i} p\left(c_{j}=k \mid \mathbf{x}_{j}, m_{j}=i, \mu_{i}, \Sigma_{i}\right) p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right) \\
&=\sum_{(\mathbf{x} i, c j) \in D_{l}} \ln \sum_{i=1}^{N} \alpha_{i} \beta_{c j \mid i} p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)+\sum_{\mathbf{x}_{i} \in D_{u}} \ln \sum_{i=1}^{N} \sum_{k=1}^{|\mathcal{C}|} \alpha_{i} \beta_{k \mid i} p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)
\end{aligned}\tag{27}
$$
显然，此时的模型参数 $\alpha_{i}, \mu_{i}, \Sigma_{i}$ 与 EM-I一致，对于 $\beta_{k \mid i}$ ，同样满足隐含条件: $\beta_{k \mid i} \geq 0, \sum_{k=1}^{|\mathcal{C}|} \beta_{k \mid i}=1$ ，因此同样将 $(27)$ 带入 $(21)$ 求偏导，并令 $(21)$ 对 $\beta_{k \mid i}$ 的导数为 0 ，得到
$$
\frac{\partial L L\left(D_{l} \cup D_{u}\right)}{\partial \beta_{k \mid i}}=\sum_{\mathbf{x}_{j} \in D_{l} \wedge c_{j}=k} \frac{\alpha_{i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)}{\sum_{i=1}^{N} \alpha_{i} \cdot \beta_{c j \mid i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)}+\sum_{\mathbf{x}_{j} \in D_{u}} \frac{\alpha_{i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)}{\sum_{i=1}^{N} \alpha_{i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)}+\lambda=0\tag{28}
$$
$$
p\left(m_{j}=i, c_{j}=k \mid \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right)=\frac{\alpha_{i} \cdot \beta_{k \mid i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)}{\sum_{i=1}^{N} \alpha_{i} \cdot p\left(\mathbf{x}_{j} \mid \mu_{i}, \Sigma_{i}\right)}\tag{29}
$$
对 $(28)$ 两边同乘 $\beta_{k \mid i}$ ， 结合 $(10),(29)$ 可得:
$$
0=\sum_{\mathbf{x} j \in D l \wedge c j=k} p\left(m_{j}=i \mid c_{j}, \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right)+\sum_{\mathbf{x} j \in D_{u}} p\left(m_{j}=i, c_{j}=k \mid \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right)+\beta_{k \mid i} \lambda\tag{30}
$$
对所有类标记求和可得:
$$
\lambda=-M \alpha_{i}\tag{31}
$$
最后，将$(31)$带入$(30)$即可解得:
$$
\beta_{k \mid i}=\frac{1}{M \alpha_{i}}\left(\sum_{\mathbf{x}_{j} \in D_{l} \wedge c_{j}=k} p\left(m_{j}=i \mid c_{j}, \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right)+\sum_{\mathbf{x}_{j} \in D_{u}} p\left(m_{j}=i, c_{j}=k \mid \mathbf{x}_{j}, \mu_{i}, \Sigma_{i}\right)\right)\tag{32}
$$
由此，我们得到了EM-I和EM-II算法下的模型参数 $\alpha_{i}, \mu_{i}, \Sigma_{i}, \beta_{k \mid i}$ 的更新公式。
