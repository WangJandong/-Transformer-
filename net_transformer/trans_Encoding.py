import torch.nn as nn
from MultiHead import MultiHeadedAttention
from FeedForward import FeedForwardNet
from Embedding import  Embeddings
from PositionalEncoding import PositionalEncoding
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadedAttention(d_model, n_head, dropout)
        self.ffn = FeedForwardNet(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        output = self.mha(x, x, x, mask)
        return self.ffn(output)


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_ff, N=6, dropout=0.1):
        super().__init__()
        self.emb = Embeddings(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(N)
        ])

    def forward(self, x, mask):
        x = self.emb(x)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x