# XGBoost总结 
### 写在前面
XGBoost既是我看的第一篇论文，也是我写的第一篇总结。中间断断续续的看了些推荐系统、深度学习方面的论文，再来写XGBoost的总结的时候，已经是又是一年后了。  
XGBoost基本算传统模型的最高峰了：引入树模型的非线形，使用集成学习boost的思想，借鉴了随机森林的抽样和列抽样，借鉴了逻辑回归的L2范数，特征粒度上支持并行，Shrinkage等。   

##  一、原理部分
####  1、基本原理  
使用回归树作为基分类器，所以既能用来分类，也可做回归。将K棵树的预测值相加作为最后的预测值(F是回归树空间)： 
<center> 
<img src=figure/1.png width = 45% div align=center> 
</center> 

####  2、Objective(目标函数)  
损失函数l 衡量预测值与真实值的差距，是可微的凸函数。  
正则项衡量模型的复杂度: T是叶子节点的数目，w是每个节点上的数值。  
<center> <img src=figure/2.png width = 45% div align=center> </center> 

####  3、推导
Addictive Training：  
由于f是树结构，不能使传统的方法--梯度下降去求最优的f，所以使用Addictive Training求解，即每一步都通过最小化L(t)，求解ft
<center> 
<img src=figure/3.png width = 55% div align=center> 
</center> 


泰勒近似：  
将l(yi, yˆi(t−1) + ft(xi))在yˆi(t−1)用泰勒公式2阶展开，近似替代l(yi, yˆi(t−1) + ft(xi))，并移除常数项，得到需要求解的如下形式，其中 gi是l(yi, yˆi(t−1) + ft(xi))在yˆi(t−1)的一阶导数，hi是二阶导数：
<center> 
<img src=figure/4.png width = 60% div align=center>
</center>


根据叶子结点上的样本改写：  
定义Ij = {i|q(xi) = j}为：叶子节点j上的所有样本。上式可写成如下形式：
<center> 
<img src=figure/5.png width = 65% div align=center>
</center> 


二次函数最优解：  
这是个二次函数，最小化L(t)，所以可得每个叶子节点的值wj和最小化的L(t)分别为：
<center> 
<img src=figure/6.png width = 33% div align=center>
</center> 

<center> 
<img src=figure/7.png width = 50% div align=center>
</center> 


寻找最优分裂点：  
这个公式形式上跟ID3算法(采用entropy计算增益)、CART算法(采用gini指数计算增益)是一致的，都是用分裂后的某种值减去分裂前的某种值，从而得到增益。
假定IL和IR为节点分裂后的左子树和右子树上的样本，I=IL+ IR，则分裂后损失的减少为：
<center> 
<img src=figure/8.png width = 90% div align=center>
</center> 

树节点在进行分裂时，我们需要计算每个特征的每个分割点对应的增益，即用贪心法枚举所有可能的分割点：
<center> 
<img src=figure/9.png width = 80% div align=center>  
</center> 

##  二、模型特性
####  1、Shrinkage(缩减)：
XGBoost在进行完一次迭代后，会将叶子节点的权重乘上该系数。  
类似于梯度下降里的learning rate，其基本思想减少每棵树的影响，并预留优化的空间给后面的树(防止过拟合)。

####  2、Column Sub-sampling(列抽样)：  
本人理解是增加树之间的差异性，减少最终结果的方差，最后达到防止过拟合的效果；另外也可加快训练速度，毕竟相当于减少了特征。

#### 3、缺失值处理：
XGBoost在每个特征上分裂的时候，会默认的把缺失值归入左子树或者右子树，最终选取损失函数减少最多的那种情况。  
所以XGBoost能处理缺失值，并学习出它的分裂方向；缺失值还能加速XGBoost的计算(需要尝试分裂的点减少了)。

#### 4、Parallel(并行)：
并行不是tree粒度的并行，xgboost也是一次迭代完才能进行下一次迭代的。而是在特征粒度上的并行。   
决策树的学习最耗时的一个步骤就是对特征的值进行排序，xgboost在训练之前，预先对数据进行了排序，然后保存为block结构在内存中，后面的迭代中重复地使用这个结构。这个block结构也使得并行成为了可能，在进行节点的分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，那么各个特征的增益计算就可以开多线程进行。

## 三、和GBDT比较：
1、GBDT在优化时只用到一阶导数的信息，XGBoost用到了一阶、二阶导数的信息；  
2、XGBoost在代价函数里加入了正则项，包含了叶子节点的数目和每个叶子节点上score的L2的平方和。防止模型过拟合；  
3、Shrinkage: 相当于学习速率(xgboost中的eta)。xgboost在进行完一次迭代后，会将叶子节点的权重乘上该系数，主要是为了削弱每棵树的影响，让后面有更大的学习空间；  
4、列抽样：借鉴随机森林的做法，降低过拟合+减少计算；  
5、对缺失值处理：对于特征的值有缺失的样本，xgboost可以自动学习出它的分裂方向；  
6、支持并行(特征粒度)；

## 四、基础总结
1、决策树：  
只能处理离散特征，不能处理连续特征(连续特征必须分桶)；  
选择分类的特征有多少种类别，就分裂出多少棵子树；  
ID3算法：通过计算信息增益，选取信息增益最大的特征，来进行分裂。因为倾向于选择类别较多的特征(比如省份有34类，而年龄只有2类，因此省份更容易被选为分裂的特征)；  
C4.5算法：为了ID3倾向于选择类别较多的特征，C4.5t通过信息增益比来选择分裂的特征。

2、CART(分类与回归树)：  
回归树：  
只能处理连续特征或者类别有序的离散特征(如年龄)，无序的离散特征必须one-hot处理(如省份)；  
为二叉树；  
损失函数一般为平方损失；  
XGBoost和GBDT都是用的是CART回归树(XGBoost还可以使用线形模型作为基模型)。

分类树：
只能处理离散特征，不能处理连续特征；连续特征最好分桶后one-hot处理。  
为二叉树；

## 五、APPENDIX
论文：https://github.com/sam6209/Machine-Learning/blob/master/machine_learning/XGBoost/XGBoost-2016.pdf  

课件：https://github.com/sam6209/Machine-Learning/blob/master/machine_learning/XGBoost/Introduction-to-Boosted-Tree.pdf  

参考：https://www.zhihu.com/question/41354392/answer/98658997
