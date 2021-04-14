import torch
import torch.nn as nn

from .model_parts import LayerNorm, TemporalAttention
from .speaker_encoder import SpeakerEncoder


class AtssNetBlock(nn.Module):
    def __init__(self):
        super(AtssNet, self).__init__()
        self.refer_encoder = SpeakerEncoder(num_speakers=....)
        self.layer_norm = nn.LayerNorm()
        self.temporal_attention = TemporalAttention()

    def forward(self, refer_spec, noicy_spec):
        with torch.no_grad():
            refer_emb = self.refer_encoder(refer_spec)
            refer_emb = refer_emb.expand(refer_emb.size(0), refer_emb.size(1), noicy_spec.size(3)).unsqueeze(1)
            # refer_emb.shape --> [batch, 1, self.refer_freq, time]
        noicy_spec = self.layer_norm_TA(noicy_spec) 

        raise NotImplementedError
