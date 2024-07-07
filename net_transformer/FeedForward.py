import torch.nn as nn
class FeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.relu(self.w1(x))
        x = self.dropout(self.w2(x))
        return self.norm(x + residual)