import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class FeedForwardLayer(nn.Module):
    def __init__(self, in_features):
        super(FeedForwardLayer, self).__init__()
        self.in_features = in_features
        self.model = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=self.in_features*4),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.in_features*4, out_features=self.in_features)
            )

    def forward(self, x):
        out = x.permute(0, 2, 3, 1).reshape(x.size(0)*x.size(2)*x.size(3), x.size(1))
        out = self.model(out)
        out = out.reshape(x.size(0), x.size(2), x.size(3), x.size(1)).permute(0, 3, 1, 2)
        return out


class DilationCNN(nn.Module):
    def __init__(self, block_idx):
        super(DilationCNN, self).__init__()
        if block_idx == 0:
            self.in_channels = 1
        else:
            self.in_channels = 64
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=5, padding=(2, 2), dilation=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=(4, 2), dilation=(2, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=(8, 2), dilation=(4, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=(16, 2), dilation=(8, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=(32, 2), dilation=(16, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, dilation=1),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.model(x)


class SelfAttention(nn.Module):
    def __init__(self, att_dim=64):
        super(SelfAttention, self).__init__()
        self.att_dim = att_dim
        self.conv_query = nn.Conv2d(self.att_dim, self.att_dim, kernel_size=1, bias=False)
        self.conv_keys = nn.Conv2d(self.att_dim, self.att_dim, kernel_size=1, bias=False)
        self.conv_values = nn.Conv2d(self.att_dim, self.att_dim, kernel_size=1, bias=False)

    def _attention(self, query, keys, values):
        '''
        Scaled Dot-Product Attention
        '''
        Q_K = torch.matmul(query, keys.transpose(2, 3)) / math.sqrt(self.att_dim)
        out = torch.matmul(F.softmax(Q_K, dim=1), values)
        return out

    def forward(self, x):
        # x.shape --> [batch, ch, freq, time]
        query = self.conv_query(x)
        keys = self.conv_keys(x)
        values = self.conv_values(x)
        out = self._attention(query, keys, values) # shape --> [batch, ch, freq, time]
        out = out.permute(0, 2, 1, 3) # shape --> [batch, freq, ch, time]
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, block_idx, att_dim=64, num_heads=2):
        super(MultiHeadAttention, self).__init__()
        # Multi-Head Attention parameters:
        self.att_dim = att_dim
        self.num_heads = num_heads
        # Dilation CNN:
        self.dilation_cnn = DilationCNN(block_idx)
        # list of attention heads:
        self.attention_heads = nn.ModuleList()
        for i in range(self.num_heads):
            self.attention_heads.append(SelfAttention(att_dim=self.att_dim))
        # multi-head:
        self.multi_head_conv = nn.Conv2d((513+256)*self.num_heads, 513, kernel_size=3, padding=1)

    def forward(self, x, refer_emb):
        x = torch.cat([x, refer_emb], dim=2) # shape --> [batch, ch, freq, time]
        x = self.dilation_cnn(x) # shape --> [batch, ch, freq, time]
        # Multi-Head Attention:
        head_outs = []
        for head_idx in range(self.num_heads):
            head_outs.append(self.attention_heads[head_idx](x))
        out = self.multi_head_conv(torch.cat(head_outs, dim=1))
        return out

