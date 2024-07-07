import torch.nn as nn
import math

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # Embedding层
        self.lut = nn.Embedding(vocab_size, d_model)
        # Embedding维数
        self.d_model = d_model

    def forward(self, x):
        # 返回x对应的embedding矩阵，需要乘以math.sqrt(d_model)
        return self.lut(x) * math.sqrt(self.d_model)