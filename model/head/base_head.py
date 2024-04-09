import torch
import torch.nn as nn


# 1×1 Conv reduce channel
class BaseHead(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BaseHead, self).__init__()
        self.head = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, feature):
        prediction = nn.functional.interpolate(self.head(feature), scale_factor=4, mode='bilinear', align_corners=True)
        return prediction

