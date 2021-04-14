import torch
import torch.nn as nn

from .model_parts import MultiHeadAttention
from .speaker_encoder import SpeakerEncoder


class TemporalResudialBlock(nn.Module):
    def __init__(self):
        super(TemporalResudialBlock, self).__init__()
        self.layer_norm = nn.LayerNorm([1, 512, 300])
        self.model = MultiHeadAttention(att_dim=64, num_heads=2)

    def forward(self, refer_emb, noicy_spec):
        x = noicy_spec
        noicy_spec = self.layer_norm(noicy_spec)
        out = self.model(refer_emb, noicy_spec)


class AtssNetBlock(nn.Module):
    def __init__(self):
        super(AtssNet, self).__init__()
        self.refer_encoder = SpeakerEncoder(num_speakers=....)
        

    def forward(self, refer_spec, noicy_spec):
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





        raise NotImplementedError
