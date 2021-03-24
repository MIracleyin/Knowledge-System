# 推荐系统 - DeepFM模型  
近期参与了Datawhale有关深度推荐系统学习的活动，task-03为哈尔滨工业大学与华为公司论文[DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/pdf/1703.04247.pdf)的学习。
DeepFM相对于Wide&Deep,用FM替换了原本的Wide部分,强化了浅层网络部分特征组合能力.  

---
## 内容概述
DeepFM模型是一种可以从原始特征中抽取到各种复杂度特征的端到端模型，没有人工特征工程的困扰.  
* DeepFM模型包含FM和DNN两部分，FM模型可以抽取low-order特征，DNN可以抽取high-order特征。无需Wide&Deep模型人工特征工程。  
* 由于输入仅为原始特征，而且FM和DNN共享输入向量特征，DeepFM模型训练速度很快。  
* 在Benchmark数据集和商业数据集上，DeepFM效果超过目前所有模型。  
---
## 模型结构

### FM模型结构
![FM](/blog/images/posts_imgs/deepfm/图片image-20210225181340313.png)  

从图中大致可以看出FM Layer是由一阶特征和二阶特征Concatenate到一起在经过一个Sigmoid得到logits（结合FM的公式一起看），
所以在实现的时候需要单独考虑linear部分和FM交叉特征部分。 
$$ \hat{y}{FM}(x) = w_0+\sum{i=1}^N w_ix_i + \sum_{i=1}^N \sum_{j=i+1}^N v_i^T v_j x_ix_j $$

### Deep模型结构
![Deep](/blog/images/posts_imgs/deepfm/图片image-20210225181010107.png)  

Deep模块是为了学习高阶的特征组合，在上图中使用用全连接的方式将Dense Embedding输入到Hidden Layer，这里面Dense Embeddings就是为了解决DNN中的参数爆炸问题，这也是推荐模型中常用的处理方法。
Embedding层的输出是将所有id类特征对应的embedding向量concat到到一起输入到DNN中。其中$v_i$表示第i个field的embedding，m是field的数量。 $$ z_1=[v_1, v_2, ..., v_m] $$ 上一层的输出作为下一层的输入，
我们得到： $$ z_L=\sigma(W_{L-1} z_{L-1}+b_{L-1}) $$ 其中$\sigma$表示激活函数，$z, W, b $分别表示该层的输入、权重和偏置。
最后进入DNN部分输出使用sigmod激活函数进行激活： $$ y_{DNN}=\sigma(W^{L}a^L+b^L) $$

### 总览

![DeepFM](/blog/images/posts_imgs/deepfm/图片image-20210225180556628.png)  

* FM模块与DNN模块共享EMbedding层
* FM模块对不同特征域的Embedding两两交叉

## 代码实现

### FM模块实现
```python
class FM(nn.Module):

    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        return cross_term
```


### DNN模块实现
```python
class DNN(nn.Module):

    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input
```

## 实验
* 使用SGD,lr=0.01,epoch=15,在12个epoch得到val_auc=0.67
* 发现使用adagrad优化器,很快就过拟合了,当然样本太小,正常

## 思考
* 如果对于FM采用随机梯度下降SGD训练模型参数，请写出模型各个参数的梯度和FM参数训练的复杂度.
>![梯度](/blog/images/posts_imgs/deepfm/equation.svg)  
>模型的复杂度是O(kn),其中n代表样本特征数量,k为隐向量长度.   
* 对于DeepFM图中所示，根据你的理解Sparse Feature中的不同颜色节点分别表示什么意思?
>黄色节点是一阶特征直接Addition输入FM,灰色节点通过Embedding进行二阶建模,之后两两组合内积输入FM.

| 任务           | 人员 |
| -------------- | ---- |
| 文本撰写、实验 | 张研 |
| 资料整理       | 章寅 |

