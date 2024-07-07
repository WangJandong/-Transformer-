import torch
import torch.nn as nn
import math

def attention(query, key, value, mask=None, dropout=None):
    # 将query矩阵的最后一个维度值作为d_k
    d_k = query.size(-1)
    # 将key的最后两个维度互换(转置)，与query矩阵相乘，除以d_k开根号
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 1, -1e9)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 头的数量要能整除词向量维度
        assert d_model % n_head == 0
        self.d_k = d_model // n_head
        self.n_head = n_head

        # 三个线性变换，一个多头拼接之后的线性变换
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.linear = nn.Linear(d_model, d_model, bias=False)

        # norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, mask=None):
        residual = query
        # 分头
        batch_size = query.size(0)
        query = self.W_Q(query).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        key = self.W_K(key).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        value = self.W_V(value).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        # 计算注意力
        if mask is not None:
            mask = mask.unsqueeze(1)
        context, attn = attention(query, key, value, mask, self.dropout)
        # 拼接
        concat = context.transpose(1, 2).reshape(batch_size, -1, self.n_head * self.d_k)
        output = self.linear(concat)
        return self.norm(output + residual)

