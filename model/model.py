import torch
import re
from model.backbone.get_backbone import get_backbone
from model.neck.base_neck import BaseNeck
from model.head.base_head import BaseHead
from model.decoder.get_decoder import get_decoder

import torch.nn as nn


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            #nn.init.xavier_normal_(m.weight, gain=1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()


class Model(torch.nn.Module):
    def __init__(self, option):
        super(Model, self).__init__()
        self.opt = option
        self.backbone, self.channel_list = get_backbone(option)
        self.decoder = get_decoder(option)
        self.initialize()

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, img, shape=None):
        features = self.backbone(img)
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        if shape is None:
            shape = img.size()[2:]

        P5, P1 = self.decoder(x1, x2, x3, x4, shape)
        return P5, P1

    def initialize(self):
        if self.opt['checkpoint'] is None:
            weight_init(self)







def get_model(option):
    model = Model(option=option)
    # freezing
    if option['freeze'] == 'backbone':
        for name, parameter in model.named_parameters():
            if re.match(r'backbone.*', name) is not None:
                parameter.requires_grad = False
            elif re.match(r'decoder', name) is not None:
                if re.match(r'side_conv', name) is not None or re.match(r'conv_block', name) is not None or re.match(
                        r'predtrans5', name) is not None:
                    parameter.requires_grad = False
    elif option['freeze'] == 'decoder':
        for name, parameter in model.named_parameters():
            if re.match(r'decoder', name) is not None:
                if re.match(r'side_conv', name) is None and re.match(r'conv_block', name) is None and re.match(
                        r'predtrans5', name) is None:
                    parameter.requires_grad = False
    
    if option['thaw'] == 'backbone':
        for name, parameter in model.named_parameters():
            if re.match(r'backbone.*', name) is not None:
                parameter.requires_grad = True
            elif re.match(r'decoder', name) is not None:
                if re.match(r'side_conv', name) is not None or re.match(r'conv_block', name) is not None or re.match(
                        r'predtrans5', name) is not None:
                    parameter.requires_grad = True
    elif option['thaw'] == 'decoder':
        for name, parameter in model.named_parameters():
            if re.match(r'decoder', name) is not None:
                if re.match(r'side_conv', name) is None and re.match(r'conv_block', name) is None and re.match(
                        r'predtrans5', name) is None:
                    parameter.requires_grad = True

    model = model.cuda()
    if option['checkpoint'] is not None:
        model.load_state_dict(torch.load(option['checkpoint']))
        print('Load checkpoint from {}'.format(option['checkpoint']))
    return model



