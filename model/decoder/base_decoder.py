import torch
import torch.nn as nn
from model.blocks.base_blocks import BasicConv2d


class BaseDecoder(torch.nn.Module):
    def __init__(self, option):
        super(BaseDecoder, self).__init__()
        self.decoders = nn.ModuleList()
        for i in range(4):
            self.decoders.append(BasicConv2d(option['neck_channel'], option['neck_channel'], kernel_size=3, padding=1, norm=True, act=True))

    def forward(self, features):
        decoder_outs = []
        for feature, decoder in zip(features, self.decoders):
            decoder_outs.append(decoder(feature))
        up_feats = []
        for i, decoder_out in enumerate(decoder_outs):
            up_feats.append(
                nn.functional.interpolate(decoder_out, scale_factor=(2 ** i), mode='bilinear', align_corners=True))
        decoder_outs = torch.cat(up_feats, dim=1)
        return decoder_outs
