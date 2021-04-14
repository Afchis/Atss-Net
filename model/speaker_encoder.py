import torch
import torch.nn as nn


class ResudialBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding):
        super(ResudialBlock, self).__init__()
        self.convrelu = nn.Sequential(
            nn.Conv2d(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel=out_channel, out_channel=out_channel, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.convrelu(x) + x 


class SpeakerEncoder(nn.Module):
    def __init__(self, num_speakers):
        super(SpeakerEncoder, self).__init__()
        self.resnet18 = nn.Sequential(
            ResudialBlock(in_channel=1, out_channel=16, kernel_size=3, padding=1),
            ResudialBlock(in_channel=16, out_channel=32, kernel_size=3, padding=1),
            ResudialBlock(in_channel=32, out_channel=64, kernel_size=3, padding=1),
            ResudialBlock(in_channel=64, out_channel=128, kernel_size=3, padding=1)
            )
        self.fc1 = nn.Linear(in_features=256, out_features=64) # out_features --> must be equal dimension of the frequency bins
        self.fc2 = nn.Linear(in_features=64, out_features=num_speakers) # not use for inference

    def _pool_layer(self, x):
        '''
        input.shape --> [batch, 128, freq, time]
        output.shape --> [batch, 2*128]
        ''' 
        x = x.reshape(x.size(0), x.size(1), -1)
        return torch.cat([x.mean(dim=2), x.max(dim=2)], dim=1)


    def forward(self, x):
        '''
        x.shape --> [batch, 1, freq, time]; freq and time can be of random lenght
        ''' 
        out = self.resnet18(x) # shape --> [batch, 128, freq, time]
        out = self._pool_layer(out) # shape --> [batch, 256]
        out = self.fc1(out)
        if self.train:
            out = self.fc2(out)
        return out
