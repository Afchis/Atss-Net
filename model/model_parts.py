import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardLayer(nn.Module):
    def __init__(self):
        super(FeedForwardLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=, out_fearutes=),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=, out_fearutes=)
            )

    def forward(self, x):
        raise NotImplementedError


class DilationCNN(nn.Module):
    def __init__(self):
        super(DilationCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=(2, 2), dilation=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=(4, 2), dilation=(2, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=(8, 2), dilation=(4, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=(16, 2), dilation=(8, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=(32, 2), dilation=(16, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0, dilation=0),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.model(x)


class SelfAttention(nn.Module):
    def __init__(self, att_dim=64):
        super(HeadAttention, self).__init__()
        self.att_dim = att_dim
        self.conv_query = nn.Conv2d(self.att_dim, self.att_dim, kernel_size=1, bias=False)
        self.conv_keys = nn.Conv2d(self.att_dim, self.att_dim, kernel_size=1, bias=False)
        self.conv_values = nn.Conv2d(self.att_dim, self.att_dim, kernel_size=1, bias=False)

    def _attention(self, query, keys, values):
        '''
        Scaled Dot-Product Attention
        '''
        Q_K = torch.matmul(query, keys.transpose(2, 3)) / torch.sqrt(self.channel_dim)
        out = torch.matmul(F.Softmax(Q_K, dim={DIM}), values)
        return out

    def forward(self, query, keys, values):
        query = self.conv_query(query)
        keys = self.conv_keys(keys)
        values = self.conv_values(values)
        out = self._attention(query, keys, values)
        return = out


class MultiHeadAttention(nn.Module):
    def __init__(self, att_dim=64, num_heads=2):
        super(MultiHeadAttention, self).__init__()
        # Multi-Head Attention parameters:
        self.att_dim = att_dim
        self.num_heads = num_heads
        # Dilation CNN:
        self.dilation_cnn = DilationCNN()
        # to get Q, K, V:
        self.conv_query = nn.Conv2d(64, self.att_dim, kernel_size=1, bias=False)
        self.conv_keys = nn.Conv2d(64, self.att_dim, kernel_size=1, bias=False)
        self.conv_values = nn.Conv2d(64, self.att_dim, kernel_size=1, bias=False)
        # list of attention heads:
        self.attention_heads = nn.ModuleList()
        for i in range(self.num_heads):
            self.attention_heads.append(SelfAttention(att_dim=self.att_dim))
        # multi-head:
        self.multi_head_conv = nn.Conv2d(self.att_dim * self.num_heads, self.att_dim * self.num_heads, kernel_size=3, padding=1)

    def forward(self, refer_emb, noicy_spec):
        x = torch.cat([refer_emb, noicy_spec], dim=2)
        x = self.dilation_cnn(x)
        # get Q, K, V:
        query = self.conv_query(x)
        keys = self.conv_keys(x)
        values = self.conv_values(x)
        # Multi-Head Attention:
        head_outs = []
        for head_idx in range(self.num_heads):
            head_outs.append(self.attention_heads[head_idx])
        out = self.multi_head_conv(torch.cat(head_outs, dim=1))
        return out


