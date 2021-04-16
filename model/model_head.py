import torch
import torch.nn as nn

from .model_parts import MultiHeadAttention, FeedForwardLayer
from .speaker_encoders.encoder_dvec import SpeechEmbedder


class TemporalResudialBlock(nn.Module):
    def __init__(self, block_idx):
        super(TemporalResudialBlock, self).__init__()
        if block_idx == 0:
            self.layer_norm = nn.LayerNorm([1, 513])
        else:
            self.layer_norm = nn.LayerNorm([64, 513])
        self.model = MultiHeadAttention(block_idx, att_dim=64, num_heads=2)

    def forward(self, x, refer_emb):
        out = x.permute(0, 3, 1, 2)
        out = self.layer_norm(out)
        out = out.permute(0, 2, 3, 1)
        out = self.model(out, refer_emb)
        return out + x


class FeedForwardResudialBlock(nn.Module):
    def __init__(self):
        super(FeedForwardResudialBlock, self).__init__()
        self.layer_norm = nn.LayerNorm([64, 513])
        self.model = FeedForwardLayer(in_features=64)

    def forward(self, x):
        out = x.permute(0, 3, 1, 2)
        out = self.layer_norm(out)
        out = out.permute(0, 2, 3, 1)
        out = self.model(out)
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
    def __init__(self, num_blocks=3, RE_weights="./ignore/weights/voice_encoder/checkpoint6.pth"):
        super(AtssNet, self).__init__()
        self.refer_encoder = SpeechEmbedder()
        self.refer_encoder.load_state_dict(torch.load(RE_weights))
        self.refer_encoder.train = False
        self.num_blocks = num_blocks
        self.model = nn.ModuleList()
        for block_idx in range(self.num_blocks):
            self.model.append(AtssNetBlock(block_idx))
        self.final_transform = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, refer_spec):
        '''
        dvec = 256
        noicy_freq = 513
        refer_spec.shape --> [batch, 1, refer_freq, time]
        noicy_spec.shape --> [batch, 1, noicy_freq, time]
        '''
        with torch.no_grad():
            d_vec = list()
            for batch in refer_spec:
                batch = batch.reshape(1, 1, batch.size(0), batch.size(1), batch.size(2))
                d_vec.append(self.refer_encoder(batch))
            d_vec = torch.stack(d_vec, dim=0) # [b, 1, c]
            d_vec = d_vec.reshape(d_vec.size(0), 1, d_vec.size(2), 1)
            d_vec = d_vec.expand(d_vec.size(0), 1, d_vec.size(2), x.size(3))
            # refer_emb.shape --> [batch, 1, refer_freq, time]
        for block_idx in range(self.num_blocks):
            if block_idx == 0:
                x = self.model[block_idx](x, d_vec)
            else:
                d_vec = d_vec.expand(d_vec.size(0), 64, d_vec.size(2), x.size(3))
                x = self.model[block_idx](x, d_vec)
        out = self.final_transform(x)
        out = self.sigmoid(out)
        return out

