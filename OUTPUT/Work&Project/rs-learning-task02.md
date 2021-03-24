近期参与了Datawhale有关深度推荐系统学习的活动，task-02为谷歌论文《[Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)》的学习。这篇论文略早于前天提到的Deep Crossing模型，其核心思想在于结合了线性模型的记忆能力（memorization)与深度神经网络模型的泛化能力（generalization），使得推荐系统既可以关注到原始的重要特征，并且还可以关注到传统特征工程未被关注到的特征。本文总字数1300字，阅读时间2分钟。

- 文案写作、模型搭建：章寅
- 模型微调、实验：张研

## 1 应用场景

Wide & Deep模型被用来部署在谷歌应用商店的推荐算法中，当用户想推荐系统发起请求后，模型对用户日志进行分析，并返回一个长度为10的推荐队列，显然整个模型的输入为用户的特征，而输出为综合两部分模型的逻辑回归。

## 2 模型结构

如文章题目所说，该推荐系统可以分成三个视角看待，分别是基于广义线性模型的Wide模型，基于深度神经网络的Deep模型，以及通过统一训练更新参数实现的模型交融。

pic1

### 2.1 Wide模型

Wide模型为一个广义线性模型，输入包括一部分的原始特征，还有一部分经过构造的特征，由于没有特征工程对经验，所以我们具体实现的时候只考虑了原始特征，对构造特征没有进行探索。论文中对于特征构造举了这样的例子，两个布尔特征$$A,B$$（即取值空间为$${0,1}$$)，当$$A,B$$均为真时，该构造特征有效，而其余情况均忽略。这样Wide模型训练完之后留下的特征非常重要，因此体现了整个模型的记忆能力（memorization)

```python
class Linear(nn.Module):
    def __init__(self, input_dim):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=1)
    def forward(self, X):
        return self.linear(X)
```

### 2.2 Deep模型

Deep模型为深度神经网络模型，我们知道随着深度神经网络层数的增加，所提取到的特征就越抽象，那么也提高了模型的泛化能力（因此没有使用残差结构？对于这个部分我们还没有具体实验）具体实现如下:

```python
class DNN(nn.Module):
    def __init__(self, hidden_units, dropout=0.):
        super(DNN, self).__init__()
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))]
        )
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, X):
        for linear in self.dnn_network:
            X = linear(X)
            X = F.relu(X)

        X = self.dropout(X)
        return X
```

### 2.3 Wide模型和Deep模型的联合训练

Wide & Deep模型采用了联合训练的形式，其联合训练的公式如下：
$$
 P(Y=1|x)=\delta(w_{wide}^T[x,\phi(x)] + w_{deep}^T a^{(lf)} + b)
$$
在这篇[文章](https://mp.weixin.qq.com/s/wTB3-RP_pj58GMPT8guaww)中，提到Wide & Deep模型本质上可以看成深度交叉特征和直接特征（或者说宽特征）的线性模型。而论文原文对于两个模型采取了不同的优化方法。对于Wide模型使用的是FTRL算法（此处有些复杂暂时不表，过段时间看懂了再写一篇文章），而Deep模型使用的是Adam。具体实现时，我们对两个模型均采取了Adam优化方法，实现如下：

```python
class self_WDL(nn.Module):
    def __init__(self, feature_colums, hidden_units, dnn_dropout=0.):
        super(self_WDL, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = feature_colums
        # embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=feat['feat_num'], embedding_dim=feat['embed_dim'])
            for i, feat in enumerate(self.sparse_feature_cols)
        })

        hidden_units.insert(0, len(self.dense_feature_cols) + len(self.sparse_feature_cols)*self.sparse_feature_cols[0]['embed_dim'])
        self.dnn_network = DNN(hidden_units)
        self.linear = Linear(len(self.dense_feature_cols))
        self.final_linear = nn.Linear(hidden_units[-1], 1)

    def forward(self, X):
        dense_input, sparse_inputs = X[:, :len(self.dense_feature_cols)], X[:, len(self.dense_feature_cols):]
        sparse_inputs = sparse_inputs.long()
        sparse_embeds = [self.embed_layers['embed_' + str(i)](sparse_inputs[:, i]) for i in
                         range(sparse_inputs.shape[1])]
        sparse_embeds = torch.cat(sparse_embeds, axis=-1)
        dnn_input = torch.cat([sparse_embeds, dense_input], axis=-1)
        # Wide
        wide_out = self.linear(dense_input)

        # Deep
        deep_out = self.dnn_network(dnn_input)
        deep_out = self.final_linear(deep_out)

        # out
        outputs = torch.sigmoid(0.5 * (wide_out + deep_out))

        return outputs
```

Pic2

## 3 模型实验

### 3.1 实验设置

#### 使用SGD优化器

sgd 优化器的学习率设置为0.01 dropout 0.2

pic3

Pic4

#### 使用Adam优化器

adam 优化器的学习率设置为0.0001,dropout 0.2

pic5

Pic6

## 4 总结与回顾

- 这次我们搭建好了完整的实验pipeline，因此对模型的实际性能有了一定的探索，可是量化没有对各个子模型对性能的具体贡献，未来希望对模型做进一步更为细粒度的实验；

- 推荐算法和计算机视觉领域有着较大差别，在实际应用时，推荐模型需要快速迭代，所以整体训练的epoch数不会太多，因此选择优化速度更快的Adam效果更好；

- 论文最重要的意义就是记忆性（Memorization）和泛化性（Generalization）这两个概念。Wide & Deep模型通过共同训练的方式迭代两个模型的参数，在Wide部分有倾向性的强化某些特征的先验性，使得模型有一个较好的预测基准，体现了其记忆性（Memorization），而在Deep部分对其他特征使得整体模型有一些微调，提升了模型的泛化性（Generalization）。

- 文案写作、模型搭建：章寅

- 模型微调、实验：

  | 任务分工           | 队员 |
  | ------------------ | ---- |
  | 文案写作、模型搭建 | 章寅 |
  | 模型微调、实验     | 张研 |

  

  

