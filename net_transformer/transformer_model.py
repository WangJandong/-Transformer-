import torch
import torch.nn as nn
from trans_Decoding import Decoder
from trans_Encoding import Encoder


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_head, d_ff, N=6, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_head, d_ff, N, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_head, d_ff, N, dropout)
        self.generator = Generator(d_model, tgt_vocab_size)

    def forward(self, src_x, src_mask, tgt_x, tgt_mask):
        memory = self.encoder(src_x, src_mask)
        output = self.decoder(tgt_x, tgt_mask, memory, src_mask)
        return self.generator(output)

def make_model(src_vocab_size, tgt_vocab_size, d_model, n_head, d_ff, N=6, dropout=0.1):
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_head, d_ff, N, dropout)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


