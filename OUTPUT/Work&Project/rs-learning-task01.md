# 推荐系统 - Deep Crossing 模型

近期参与了 Datawhale 有关深度推荐系统学习的[活动](https://github.com/datawhalechina/team-learning-rs/blob/master/DeepRecommendationModel/DeepCrossing.md "活动")，task-01 为学习微软于 KDD2016 发表的论文《Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial》。该文也是第一篇企业以论文形式公布工业推荐系技术细节的模型。
本文字数1.9k，阅读时间3分钟。

## 内容概述

Deep Crossing 模型论文标题即点出了该模型所要解决的问题：如何避免人工构造特征，使用深度学习模型实现大规模端到端的[推荐模型](https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf "推荐模型")。论文具体通过两个方向在整个推荐系统模型应用深度学习技术，其一是使用 Embedding 层，作用是将离散高维的稀疏特征转为低维的密集特征，使得可以直接输入神经网络训练，其二是使用神经网络结构，作用是自动联合特征与特征交叉，使得整个系统可以实现端到端的训练，预测。

从模型的角度出发，Deep Crossing 使用了 Embedding + MLP + 残差结构，以现在的观点来说属于复杂度不高的模型，但其完整的解决了特征工程、稀疏向量稠密化、多层神经网络优化等深度学习在推荐算法中应用的问题，对推荐系统发展有着重要意义。

Deep Crossing 模型的应用场景为微软 Bing 中的的竞价排名搜索（sponsored search）场景。当用户在搜索引擎中搜索时，除了展示对应的搜索结果，同时还会展示与搜索结果相关的广告信息。竞价排名搜索的对象为：用户、广告商与搜索引擎平台，其目标为通过展示特定队列，为用户提供最匹配其意图的广告。因此，如何提高广告的点击率以及后续的转化率，是 Deep Crossing 模型的优化目标。

## 模型结构

Deep Crossing 模型是一个端到端的模型，模型结构如下

模型自下而上包括 4 层结构：Embedding 层、Stacking 层、Multiple Residual Units 层和 Scoring 层。

### Embedding 层和 Stacking 层

Embedding 层是深度学习最常用的技术之一。其作用是降低输入特征的维度，即将高维度的原始数据（图像，文字等）映射到低纬度空间。论文使用单层神经网络实现，实现形式为：

$$
X_j^o = max(0, W_jX_j^I + b_j)
$$

此处 $j$ 以区别不同特征，输入向量 $X_j^I \in \mathbb{R}^{n_j}$ 与尺寸为 $m_j \times n_j$ 的矩阵相乘在加上偏置 $b_j \in \mathbb{R}_{n_j}$,最终输出向量 $X_j^O \in \mathbb{R}^{m_j}$，当 $m_j < n_j$ 时，维度降低，即实现了 Embedding 操作。

通过 Embedding 层后，Deep Crossing 模型使用 Stacking 层将所有特征组合到一个向量中。

$$
X^O = [X_0^O, X_1^O,\dots,X_K^O]
$$

论文针对两种特征使用不同的处理方案，对于类别特征（one-hot 编码后稀疏的特征）对其进行 Embedding 处理，而对于数值型特征（通常是浮点数）则直接和 Embedding 后稀疏特征放入 Stacking 层合并。

Embedding 层主要使用了 pytorch 的实现，此处参考文档中给出的例子：

```bash
>>> embedding = nn.Embedding(10, 3)
>>> # a batch of 2 samples of 4 indices each
>>> input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
>>> embedding(input)
tensor([[[-0.0251, -1.6902,  0.7172],
         [-0.6431,  0.0748,  0.6969],
         [ 1.4970,  1.3448, -0.9685],
         [-0.3677, -2.7265, -0.1685]],

        [[ 1.4970,  1.3448, -0.9685],
         [ 0.4362, -0.4004,  0.9400],
         [-0.6431,  0.0748,  0.6969],
         [ 0.9124, -2.3616,  1.1151]]])
```

### Multiple Residual Units 层

残差层使用了图像分类常用的 ResNet 模型中的残差结构，当神经网络由于层数过深，而产生的梯度消失与梯度爆炸时，残差结构可以缓解模型退化和梯度消失问题。Deep Crossing 使用此结构进行特征组合，单个残差结构如下图所示。

```python
class Residual_block(nn.Module):
    def __init__(self, dim_stack, hidden_unit):
        super(Residual_block, self).__init__()
        self.linear1 = nn.Linear(dim_stack, hidden_unit)
        self.linear2 = nn.Linear(hidden_unit, dim_stack)
        self.relu = nn.ReLU()
    def forward(self, x):
        orig_x = x.clone()
        x = self.linear1(x)
        x = self.linear2(x)
        out = self.relu(x + orig_x)
        return out
```

### Deep Crossing 网络实现

这里参考了[该文](https://mp.weixin.qq.com/s/RqfUquT0dYZRnQ0oXKc75w)的实现

```python
class Deep_Crossing(nn.Module):
    def __init__(
            self,
            number_of_category_feature,      # 稀疏特征数量
            number_of_total_feature,         # 特征总数
            list_of_nunique_category,        # 分类变量唯一值个数，用于Embedding
            hidden_size_for_residual_block,  # 隐藏层
            embedding_size):                 # 嵌入维度
        super(Deep_Crossing, self).__init__()
        self.number_of_category_feature = number_of_category_feature
        self.number_of_total_feature = number_of_total_feature
        self.list_of_nunique_category = list_of_nunique_category

        total_size_for_single_sample = sum(
            i if i <= embedding_size else embedding_size for i in list_of_nunique_category
        ) + (number_of_total_feature - number_of_category_feature) # 传入残差层之前拼接的维度
        self.hidden_size_for_residual_block = hidden_size_for_residual_block
        self.embedding_size = embedding_size

        self.residual_blocks = nn.ModuleList([
            Residual_block(total_size_for_single_sample, size) for size in hidden_size_for_residual_block
        ]) # 残差模块
        self.Full_connect_layer_after_residual_block = nn.Linear(total_size_for_single_sample, 1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_numeric, x_category = x[:, : -self.number_of_category_feature], x[:, :-self.number_of_category_feature]
        one_hot_list = []
        for i in range(x_category.shape[1]):
            embedded_feature = torch.zeros(
                x.shape[0], self.list_of_nunique_category[i]
            ).scatter_(
                1, x_category.T[i].reshape(-1,1).long(), 1) # 将 feature 转化为 one-hot 类型
            if embedded_feature.shape[-1] > self.embedding_size: # 对于维度大于阈值进行嵌入
                embedded_feature = nn.Linear(self.list_of_nunique_category[i], self.embedding_size)(embedded_feature)
            one_hot_list.append(embedded_feature)
        x_category = torch.cat(one_hot_list, -1)   # 拼接分类变量
        x = torch.cat([x_numeric, x_category], -1) # 拼接分类变量和数值变量
        for block in self.residual_blocks:         # 残差结构
            x = block(x)
        x = self.Full_connect_layer_after_residual_block(x)
        out = self.Sigmoid(x)
        return out
```

## 总结与回顾

- 学习仓库中的数据无法用于模型性能验证，完整数据集的 pipeline 还没有完善，因此只测试了模型的正确性，没有对模型预结果进行调参分析；
- Deep Crossing 模型在推荐系统上实现了端到端的训练，预测。使用 Embedding 层将高维度转为可以用于训练的低维度特征，使用残差结构自动提取特征，以现在的观点来看模型简单，但对深度学习在推荐系统的应用起到了巨大推进作用。
