支持向量机（SVM）全称Support Vecor Machine，谈及机器学习无论回归还是分类，一定都会拿它进行测试，它是机器学习算法中最受关注的算法之一。
这里本文不过多的去研究它的数学推导公式，而是浅尝辄止的去探究一下它的原理和作用，以及在sklearn当如如何高效的使用。
**想要去推导它数据公式的朋友可以去查看[刘建平的博客](https://www.cnblogs.com/pinard/p/6097604.html)**

## 1、SVM是如何工作的

SVM学习的基本想法是求解能够正确划分训练数据集（下图中实心黑点与空心点）并且几何间隔最大的分离超平面。如下图所示， $w·x+b=0$即为分离超平面作为决策边界，对于线性可分的数据集来说，这样的超平面有无穷多个（即感知机），但是几何间隔最大的分离超平面却是唯一的。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200318100200366.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwMTk1MzYw,size_16,color_FFFFFF,t_70)
注：在几何中，超平面是一个空间的子空间，它是维度比所在空间小一维的空间。 如果数据空间本身是三维的，则其超平面是二维平面，而如果数据空间本身是二维的，则其超平面是一维的直线。在二分类问题中，如果一个超平面能够将数据划分为两个集合，其中每个集合中包含单独的一个类别，我们就说这个超平面是数据的“决策边界“。

SVM目标是"找出边际最大的决策边界"，听起来是一个十分熟悉的表达，这是一个最优化问题，而最优化问题往往和损失函数联系在一起。和逻辑回归中的过程一样，SVM也是通过最小化损失函数来求解一个用于后续模型使用的重要信息：决策边界。

这里梳理一下这整个过程（当然你要死有兴趣的话，可以去看论文再自己推导一下）：

 1. 定义决策边界的数学表达，并基于此表达定义分类函数
 2. 为寻找最大边际引出损失函数：$min \;\; \frac{1}{2}||w||^2  \;$
 3. 为求解能够使边际最大化的$w$和$b$，引入拉格朗日因子α
 4. 引入拉格朗日对偶函数，使求解$w$和$b$得过程转化为对α的求解
 5. 使用SMO或梯度下降等方法求解α，再根据α解出$w$和$b$，最终找出最优决策边界。

## 2、sklearn中的SVM
|  类| 含义 |
|--|--|
| `sklearn.svm.LinearSVC` | 线性支持向量分类 |
| `sklearn.svm.LinearSVR` |线性支持向量回归  |
| `sklearn.svm.NuSVC` | Nu支持向量分类 |
| `sklearn.svm.NuSVR` | Nu支持向量回归 |
|`sklearn.svm.OneClassSVM`  | 无监督异常值检测（后面会有专门模块讲） |
|`sklearn.svm.SVC`  | 支持向量分类 |
| `sklearn.svm.SVR` | 支持向量回归 |
|`sklearn.svm.l1_min_c`|返回参数C的最低边界，这样对于C在(L1_min_C，无穷大)中，模型保证不为空|

对于SVC， NuSVC，和LinearSVC 3个分类的类，SVC和 NuSVC差不多，区别仅仅在于对损失的度量方式不同，而LinearSVC从名字就可以看出，他是线性分类，也就是不支持各种低维到高维的核函数，仅仅支持线性核函数，对线性不可分的数据不能使用。

同样的，对于SVR， NuSVR，和LinearSVR 3个回归的类， SVR和NuSVR差不多，区别也仅仅在于对损失的度量方式不同。LinearSVR是线性回归，只能使用线性核函数。

我们使用这些类的时候，如果有经验知道数据是线性可以拟合的，那么使用LinearSVC去分类 或者LinearSVR去回归，它们不需要我们去慢慢的调参去选择各种核函数以及对应参数， 速度也快。如果我们对数据分布没有什么经验，一般使用SVC去分类或者SVR去回归，这就需要我们选择核函数以及对核函数调参了。

什么特殊场景需要使用NuSVC分类 和 NuSVR 回归呢？如果我们对训练集训练的错误率或者说支持向量的百分比有要求的时候，可以选择NuSVC分类 和 NuSVR 。它们有一个参数来控制这个百分比。


## 3、svm.SVC
### 3.1 SVC参数

```py
sklearn.svm.SVC(C=1.0, 
                kernel='rbf', 
                degree=3, 
                gamma='scale', 
                coef0=0.0, 
                shrinking=True, 
                probability=False, 
                tol=0.001, 
                cache_size=200, 
                class_weight=None, 
                verbose=False, 
                max_iter=-1, 
                decision_function_shape='ovr', 
                break_ties=False, 
                random_state=None
               )
```

 1. `C`：惩罚系数C，默认为1，一般需要通过交叉验证来选择一个合适的C。如果C值设定比较大，那SVC可能会选择边际较小的，能够更好地分类所有训练点的决策边界，不过模型的训练时间也会更长。如果C的设定值较小，那SVC会尽量最大化边界，决策功能会更简单，但代价是训练的准确度。换句话说，C在SVM中的影响就像正则化参数对逻辑回归的影响。
 2. `kernel`：核函数有四种内置选择，‘linear’即线性核函数, ‘poly’即多项式核函数, ‘rbf’即高斯核函数, ‘sigmoid’即sigmoid核函数。如果选择了这些核函数， 对应的核函数参数在后面有单独的参数需要调。默认是高斯核'rbf'。
 3. `degree`：如果我们在kernel参数使用了多项式核函数 'poly'，那么我们就需要对这个参数进行调参，默认是3。
 4. `gamma`：如果我们在kernel参数使用了多项式核函数 'poly'，高斯核函数‘rbf’, 或者sigmoid核函数，那么我们就需要对这个参数进行调参。默认为'auto',即$\frac{1}{特征维度}$
 5. `coef0`：如果我们在kernel参数使用了多项式核函数 'poly'，或者sigmoid核函数，那么我们就需要对这个参数进行调参,默认为0。
 6. `shrinking`：默认为True，是否使用收缩启发式计算(shrinking heuristics), 如果使用，有时可以加速最优化的计算进程（没用过，不敢用）。
 7. `probability`：默认False，是否启用概率估计。必须在调用fit之前启用它， 启用此功能会减慢SVM的运算速度。

 8. `tol`：浮点数，默认1e-3，停止迭代的容差。

 9. `cache_size`：在大样本的时候，缓存大小会影响训练速度，因此如果机器内存大，推荐用500MB甚至1000MB。默认是200，即200MB。
 10. `class_weight`：指定样本各类别的的权重，主要是为了防止训练集某些类别的样本过多，导致训练的决策过于偏向这些类别。这里可以自己指定各个样本的权重，或者用“balanced”，如果使用“balanced”，则算法会自己计算权重，样本量少的类别所对应的样本权重会高。当然，如果你的样本类别分布没有明显的偏倚，则可以不管这个参数，选择默认的"None"。
 11. `verbose`：默认False，启用详细输出。
 12. `max_iter`：整数，默认=-1，最大迭代次数，输入"-1"表示没有限制。

 13. `decision_function_shape`：默认是'ovr'，OvR相对简单，但分类效果相对略差（这里指大多数样本分布情况，某些样本分布下OvR可能更好）。而OvO分类相对精确，但是分类速度没有OvR快。
 14. `break_ties`：默认False，如果为TRUE，Decision_Function_Shape=‘ovr’，且类别数>2，则预测结果将根据Decision_Function的置信度值打破关系，否则返回类别中的第一个类。请注意，与简单的预测相比，打破联系的计算成本相对较高。
 15. `random_state`：随机数种子。

### 3.2 SVC属性和接口列表
| 属性 | 含义 |
|--|--|
`support_`	|支持向量的索引
`support_vectors_`	|支持向量本身
`n_support_`	|每个类的支持向最数
`dual_coef_`	|决策函数中支持向量的系数。对于多分类问题而言，是所有ovo模式分类器中的系数，多类情况下系数的布局不是非常重要
`coef_`	|赋予特征的权重（原始问题中的系数），这个属性仅适用线性内核，`coef_`是从`dual_coef_` 和 `support_vectors_` 派生的只读属性
`intercept_`|决策函数中的常量， 在二维平面上是截距
`fit_status_`|如果正确拟合， 则显示 0,  否则为1  (将发出警告）
`classes_`|类标签
`class_weight_`|每类参数C的乘子。根据类权重参数计算


| 接口 |
|--|
`decision_function(self, X)`|
`fit(self, X, y[, sample_weight])`|
`get_params(self[, deep])`|
`predict(self, X)`|
`score(self, X, y[, sample_weight])`|
`set_params(self, \*\*params)`|

## 4、SVM 真实案例-天气预报

这个数据是从Kaggle上下载的，未经过预处理的澳大利亚天气数据集。我们的目标是在这个数据集上来**预测明天是否会下雨**。预测天气是一个非常非常困难的主题，因为影响天气的因素太多，而Kaggle的这份数据也丝毫不让我们失望，是一份非常难的数据集，难到我们目前学过的所有算法在这个数据集上都不会有太好的结果，尤其是召回率recall，异常地低。

希望使用原数据集的朋友可以到Kaggle下载最原始版本，或者直接从我的GitHub上获取数据。

希望使用原数据集的朋友可以到Kaggle下载最原始版本，或者直接从我的GitHub中获取数据：
[Kaggle下载链接在这里](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)


这里就不直接展示代码了，以后都会将代码放进我的GitHub里面：



