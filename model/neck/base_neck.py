import torch
import torch.nn as nn


# 1×1 Conv reduce channel
class BaseNeck(torch.nn.Module):
    def __init__(self, in_channels, out_channel):
        super(BaseNeck, self).__init__()
        self.necks = nn.ModuleList()
        for in_channel in in_channels:
            self.necks.append(nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False))

    def forward(self, features):
        outs = []
        for neck, feature in zip(self.necks, features):
            outs.append(neck(feature))
        return outs
