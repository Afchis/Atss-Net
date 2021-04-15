import torch
import torch.nn as nn

from model_parts import MultiHeadAttention, FeedForwardLayer
from speaker_encoder import SpeakerEncoder


class TemporalResudialBlock(nn.Module):
    def __init__(self):
        super(TemporalResudialBlock, self).__init__()
        self.layer_norm = nn.LayerNorm([1, 512, 300])
        self.model = MultiHeadAttention(att_dim=64, num_heads=2)

    def forward(self, refer_emb, noicy_spec):
        x = noicy_spec
        noicy_spec = self.layer_norm(noicy_spec)
        out = self.model(refer_emb, noicy_spec)
        return out + x


class FeedForwardResudialBlock(nn.Module):
    def __init__(self):
        super(FeedForwardResudialBlock, self).__init__()
        self.layer_norm = nn.LayerNorm([1, 512, 300])
        self.model = FeedForwardLayer(in_features=5)

    def forward(self, x):
        out = self.layer_norm(x)
        out = self.model(x)
        return out


class AtssNetBlock(nn.Module):
    def __init__(self):
        super(AtssNetBlock, self).__init__()
        self.tem_res_block = TemporalResudialBlock()
        self.f_f_res_block = FeedForwardResudialBlock()

    def forward(self, x, refer_spec):
        out = self.tem_res_block(x, refer_spec)
        out = self.f_f_res_block(out)
        return out


class AtssNet(nn.Module):
    def __init__(self, num_blocks=3):
        super(AtssNet, self).__init__()
        self.refer_encoder = SpeakerEncoder(num_speakers=999)
        self.num_blocks = num_blocks
        self.model = nn.ModuleList()
        for i in range(self.num_blocks):
            self.model.append(AtssNetBlock())

    def forward(self, x, refer_spec):
        '''
        refer_freq = 64
        noicy_freq = 512
        refer_spec.shape --> [batch, 1, refer_freq, time]
        noicy_spec.shape --> [batch, 1, noicy_freq, time]
        '''
        with torch.no_grad():
            refer_emb = self.refer_encoder(refer_spec) # refer_emb.shape --> [batch, 1, refer_freq, time]
            refer_emb = refer_emb.expand(refer_emb.size(0), refer_emb.size(1), noicy_spec.size(3)).unsqueeze(1)
            # refer_emb.shape --> [batch, 1, refer_freq, time]
        for i in range(self.num_blocks):
            x = self.model[i](x, noicy_spec)
        return x


if __name__ == "__main__":
    x = torch.rand([4, 1, 512, 300])
    refer_spec = torch.rand([4, 1, 64, 300])
    model = AtssNet(num_blocks=3)
    model(x, refer_spec)

