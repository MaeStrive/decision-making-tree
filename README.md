# decision-making-tree
@[toc]
# 一、什么是决策树
**不用语言描述，直接看图**
![在这里插入图片描述](https://img-blog.csdnimg.cn/208df4b58aef4fc586bbb4eb3f531c12.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBATWFlX3N0cml2ZQ==,size_20,color_FFFFFF,t_70,g_se,x_16)
关于是否去上课，首先看的是**是否能起床**，然后**课程**，然后**人数**，然后**是否点名**。
即按特征的先后顺序进行分类；
# 二、对鸢尾花进行分类
决策树API：

```
from sklearn.tree import DecisionTreeClassifier
```

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def decision_iris():
    """
    鸢尾花分类
    :return:
    """
    # 1、获取数据
    iris = load_iris()
    # 2、划分数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)
    # 3、决策树预估
    estimator = DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train, y_train)
    # 4、模型评估
    # (1)
    y_predict = estimator.predict(x_test)
    print(y_predict)
    print(y_predict == y_test)
    # (2)
    score = estimator.score(x_test, y_test)
    print(score)
    return None


if __name__ == '__main__':
    decision_iris()
```

```
[0 2 1 2 1 1 1 1 1 0 2 1 2 2 0 2 1 1 1 1 0 2 0 1 2 0 2 2 2 1 0 0 1 1 1 0 0
 0]
[ True  True  True  True  True  True  True False  True  True  True  True
  True  True  True  True  True  True False  True  True  True  True  True
  True  True  True  True  True False  True  True  True  True  True  True
  True  True]
0.9210526315789473
```
# 三、决策树可视化
API：能够将树结构保存为dot格式
```
from sklearn.tree import export_graphviz
```

```
# 决策树可视化
export_graphviz(estimator, out_file="iris_tree.dot")
```
可以**安装插件**或者去[在线网站](http://www.webgraphviz.com/) 就能够看到对应的图
![在这里插入图片描述](https://img-blog.csdnimg.cn/f192125d53584fcbbc19dd3e4a9c80c1.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBATWFlX3N0cml2ZQ==,size_20,color_FFFFFF,t_70,g_se,x_16)
![在这里插入图片描述](https://img-blog.csdnimg.cn/8ebd264625d942718d88e4ad66c68d91.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBATWFlX3N0cml2ZQ==,size_13,color_FFFFFF,t_70,g_se,x_16)
我们可以传入特征名 使图看得更加直白

```
export_graphviz(estimator, out_file="iris_tree.dot", feature_names=iris.feature_names)
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/4367d718f47749d4821d42670793a5d7.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBATWFlX3N0cml2ZQ==,size_13,color_FFFFFF,t_70,g_se,x_16)

# 三、决策树算法总结
优点：
- 简单理解和解释，树木可视化

缺点：
- 不能很好的推广到数据特别复杂的树
