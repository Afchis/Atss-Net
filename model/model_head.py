import torch
import torch.nn as nn

from .model_parts import MultiHeadAttention, FeedForwardLayer
from .speaker_encoders.speaker_encoder import SpeakerEncoder


class TemporalResudialBlock(nn.Module):
    def __init__(self, block_idx):
        super(TemporalResudialBlock, self).__init__()
        if block_idx == 0:
            self.layer_norm = nn.LayerNorm([1, 512, 300])
        else:
            self.layer_norm = nn.LayerNorm([64, 512, 300])
        self.model = MultiHeadAttention(block_idx, att_dim=64, num_heads=2)

    def forward(self, x, refer_emb):
        out = self.layer_norm(x)
        out = self.model(x, refer_emb)
        return out + x


class FeedForwardResudialBlock(nn.Module):
    def __init__(self):
        super(FeedForwardResudialBlock, self).__init__()
        self.layer_norm = nn.LayerNorm([64, 512, 300])
        self.model = FeedForwardLayer(in_features=64)

    def forward(self, x):
        out = self.layer_norm(x)
        out = self.model(x)
        return out


class AtssNetBlock(nn.Module):
    def __init__(self, block_idx):
        super(AtssNetBlock, self).__init__()
        self.tem_res_block = TemporalResudialBlock(block_idx)
        self.f_f_res_block = FeedForwardResudialBlock()

    def forward(self, x, refer_spec):
        out = self.tem_res_block(x, refer_spec)
        out = self.f_f_res_block(out)
        return out


class AtssNet(nn.Module):
    def __init__(self, num_blocks=3):
        super(AtssNet, self).__init__()
        self.refer_encoder = SpeakerEncoder(num_speakers=921)
        self.refer_encoder.train = False
        self.num_blocks = num_blocks
        self.model = nn.ModuleList()
        for block_idx in range(self.num_blocks):
            self.model.append(AtssNetBlock(block_idx))

    def forward(self, x, refer_spec):
        '''
        refer_freq = 64
        noicy_freq = 512
        refer_spec.shape --> [batch, 1, refer_freq, time]
        noicy_spec.shape --> [batch, 1, noicy_freq, time]
        '''
        with torch.no_grad():
            refer_emb = self.refer_encoder(refer_spec) # refer_emb.shape --> [batch, 64]
            refer_emb = refer_emb.reshape(refer_emb.size(0), 1, refer_emb.size(1), 1)
            refer_emb = refer_emb.expand(refer_emb.size(0), 1, refer_emb.size(2), x.size(3))
            # refer_emb.shape --> [batch, 1, refer_freq, time]
        for block_idx in range(self.num_blocks):
            if block_idx == 0:
                x = self.model[block_idx](x, refer_emb)
            else:
                refer_emb = refer_emb.expand(refer_emb.size(0), 64, refer_emb.size(2), x.size(3))
                x = self.model[block_idx](x, refer_emb)
        print("done!")
        return x


if __name__ == "__main__":
    x = torch.rand([4, 1, 512, 300])
    refer_spec = torch.rand([4, 1, 64, 300])
    model = AtssNet(num_blocks=3)
    print(model)
    print(sum(p.numel() for p in model.parameters()))
    model(x, refer_spec)

