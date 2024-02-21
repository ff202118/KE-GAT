import torch
from torch import nn
from torch.nn import Embedding
import torch.nn.functional as F
from layers import *
from functools import reduce

def split(a, n):
    # 将列表a分成n个部分
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

class WordEncoder(nn.Module):
    def __init__(self, embeddings: torch.tensor, hidden_size: int, num_layers: int, dropout: float, max_len: int,
                 max_sent_num: int):
        super(WordEncoder, self).__init__()

        self.L = max_len  # 最大句子长度
        self.sent_num = max_sent_num  # 句子数量
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)  # 创建嵌入层
        self.embed_dim = 300  # 嵌入维度
        self.gru = nn.GRU(input_size=self.embed_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                          dropout=dropout, bidirectional=True)  # 创建双向GRU
        self.h_proj = nn.Linear(hidden_size * 2, hidden_size)  # 创建线性层
        self.utw = nn.Linear(hidden_size, 1, bias=False)  # 创建线性层

    def forward(self, batch):
        # 将数据从[[sent_list[[],[]]],[sent_list[[],[]]]]转换为[[vi],[vi],...]
        # Ns：最大句子数量
        # batch = sum(batch) [batch, Ns, sent_max_len] -> [batch*N, max_len]
        input = self.embed(batch)  # 嵌入输入数据
        hit, _ = self.gru(input)  # 获取GRU输出，维度=[batch*Ns, sent_max_len, hidden_size*2]
        uit = torch.tanh(self.h_proj(hit))  # 计算tanh(Wh+b)，[batch*Ns, L, h]
        ait = torch.exp(self.utw(uit))  # 计算exp(uit*utw)，[batch*Ns, L, 1]

        # ait=exp(uit*utw)/sum(exp(uit*utw))
        aitsum = torch.sum(ait, dim=2, keepdim=True)  # [batch*Ns, 1]
        aitsum.type(torch.float)
        aitsum = aitsum.expand(-1, self.L, -1)  # [batch*Ns, L, 1]
        ait = torch.div(ait, aitsum)  # [batch*Ns, L, 1]

        # vi=sum_{t=1}ait*hit
        ait = torch.transpose(ait, 1, 2)
        vi = torch.bmm(ait, hit)  # [batch*Ns, 1, 2h]
        batch_size = int(vi.shape[0] / self.sent_num)
        vi = list(split(vi, batch_size))  # [batch, Ns, 2h]
        vi = torch.stack(vi)
        vi = torch.squeeze(vi, 2)
        return vi


class SentEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: float):
        super(SentEncoder, self).__init__()
        self.gru = nn.GRU(input_size=hidden_size * 2, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                          dropout=dropout, bidirectional=True)  # 创建双向GRU

    def forward(self, batch):
        x, _ = self.gru(batch)  # 计算GRU输出，维度: [batch_size, N, hiddensize*2]，N：新闻中的句子
        return x


class Classifier(nn.Module):
    def __init__(self, hidden_size: int):
        super(Classifier, self).__init__()
        self.l1 = nn.Linear(hidden_size * 8, hidden_size * 4)  # news sent+kb sent concatenation
        self.l2 = nn.Linear(hidden_size * 4, 2)

    def forward(self, news_batch, entity_desc_batch):
        total = torch.cat((news_batch, entity_desc_batch), 2)
        layer1 = self.l1(total)
        ans = self.l2(layer1)
        y = torch.squeeze(ans, 1)
        # y=F.softmax(ans,dim=1)
        return y


class HGAT(nn.Module):
    def __init__(self, nfeat_list, nhid, nclass, dropout,
                 type_attention=True, node_attention=True,
                 gamma=0.1, sigmoid=False, orphan=True,
                 write_emb=True
                 ):
        super(HGAT, self).__init__()
        self.sigmoid = sigmoid
        self.type_attention = type_attention
        self.node_attention = node_attention

        self.write_emb = write_emb
        if self.write_emb:
            self.emb = None
            self.emb2 = None

        self.nonlinear = F.relu_

        self.nclass = nclass
        self.ntype = len(nfeat_list)

        dim_1st = nhid
        dim_2nd = nclass
        if orphan:
            dim_2nd += self.ntype - 1

        self.gc2 = nn.ModuleList()
        if not self.node_attention:
            self.gc1 = nn.ModuleList()
            for t in range(self.ntype):
                self.gc1.append(GraphConvolution(nfeat_list[t], dim_1st, bias=False))
                self.bias1 = Parameter(torch.FloatTensor(dim_1st))
                stdv = 1. / math.sqrt(dim_1st)
                self.bias1.data.uniform_(-stdv, stdv)
        else:
            self.gc1 = GraphAttentionConvolution(nfeat_list, dim_1st, gamma=gamma)
        self.gc2.append(GraphConvolution(dim_1st, dim_2nd, bias=True))

        if self.type_attention:
            self.at1 = nn.ModuleList()
            self.at2 = nn.ModuleList()
            for t in range(self.ntype):
                self.at1.append(SelfAttention(dim_1st, t, 50))
                self.at2.append(SelfAttention(dim_2nd, t, 50))

        self.dropout = dropout

    def forward(self, x_list, adj_list, adj_all=None):
        x0 = x_list

        if not self.node_attention:
            x1 = [None for _ in range(self.ntype)]
            # First Layer
            for t1 in range(self.ntype):
                x_t1 = []
                for t2 in range(self.ntype):
                    idx = t2
                    x_t1.append(self.gc1[idx](x0[t2], adj_list[t1][t2]) + self.bias1)
                if self.type_attention:
                    x_t1, weights = self.at1[t1](torch.stack(x_t1, dim=1))
                else:
                    x_t1 = reduce(torch.add, x_t1)

                x_t1 = self.nonlinear(x_t1)
                x_t1 = F.dropout(x_t1, self.dropout, training=self.training)
                x1[t1] = x_t1
        else:
            x1 = [None for _ in range(self.ntype)]
            x1_in = self.gc1(x0, adj_list)
            for t1 in range(len(x1_in)):
                x_t1 = x1_in[t1]
                if self.type_attention:
                    x_t1, weights = self.at1[t1](torch.stack(x_t1, dim=1))
                else:
                    x_t1 = reduce(torch.add, x_t1)
                x_t1 = self.nonlinear(x_t1)
                x_t1 = F.dropout(x_t1, self.dropout, training=self.training)
                x1[t1] = x_t1
        if self.write_emb:
            self.emb = x1[0]

        x2 = [None for _ in range(self.ntype)]
        # Second Layer
        for t1 in range(self.ntype):
            x_t1 = []
            for t2 in range(self.ntype):
                if adj_list[t1][t2] is None:
                    continue
                idx = 0
                x_t1.append(self.gc2[idx](x1[t2], adj_list[t1][t2]))
            if self.type_attention:
                x_t1, weights = self.at2[t1](torch.stack(x_t1, dim=1))
            else:
                x_t1 = reduce(torch.add, x_t1)

            x2[t1] = x_t1
            if self.write_emb and t1 == 0:
                self.emb2 = x2[t1]

            # output layer
            if self.sigmoid:
                x2[t1] = torch.sigmoid(x_t1)
            else:
                x2[t1] = F.log_softmax(x_t1, dim=1)

        return x2


class Model(nn.Module):
    def __init__(self,embeddings: torch.tensor, hidden_size: int, word_num_layers: int, dropout: float, max_len: int,
                 sent_num_layers: int, attention_size: int, max_news_sent_num: int, max_desc_sent_num, max_summ_sent_num):
        super(Model, self).__init__()
        self.news_word_encoder = WordEncoder(embeddings, hidden_size, word_num_layers, dropout, max_len,
                                             max_news_sent_num)
        self.desc_word_encoder = WordEncoder(embeddings, hidden_size, word_num_layers, dropout, max_len,
                                             max_desc_sent_num)
        self.summ_word_encoder = WordEncoder(embeddings, hidden_size, word_num_layers, dropout, max_len,
                                             max_summ_sent_num)

        self.sent_encoder = SentEncoder(hidden_size, sent_num_layers, dropout)