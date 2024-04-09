import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import numbers
from einops import rearrange
from model.blocks.base_blocks import DConv, weight_init
from model.blocks.DCNv3 import DCNv3_pytorch as DCNv3


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def edge_prediction(map):
    laplace = np.array(([-1, -1, -1], [-1, 8, -1], [-1, -1, -1]), dtype=np.float32)
    laplace = laplace[np.newaxis, np.newaxis, ...]
    laplace = torch.Tensor(laplace).cuda()
    edge = F.conv2d(map, laplace, padding=1)
    edge = F.relu(torch.tanh(edge))
    return edge

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

    def initialize(self):
        weight_init(self)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

    def initialize(self):
        weight_init(self)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        # x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.dwconv(x)
        # x = F.gelu(x1) * x2
        x = F.gelu(x)
        x = self.project_out(x)
        return x

    def initialize(self):
        weight_init(self)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkv1conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv3conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        q = self.qkv1conv(self.qkv_0(x))
        k = self.qkv2conv(self.qkv_1(x))
        v = self.qkv3conv(self.qkv_2(x))
        if mask is not None:
            q = q * mask
            k = k * mask

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

    def initialize(self):
        weight_init(self)


# class SeparableAttention(nn.Module):
#     def __init__(self, dim, bias):
#         super().__init__()
#
#         med_channels = int(4 * dim)
#         self.pwconv1 = nn.Conv2d(dim, med_channels, kernel_size=1, bias=bias)
#         self.act1 = nn.ReLU(True)
#         self.dwconv = nn.Conv2d(
#             med_channels, med_channels, kernel_size=7, padding=3, groups=med_channels, bias=bias)
#         self.pwconv2 = nn.Conv2d(med_channels, dim, kernel_size=1, bias=bias)
#
#     def forward(self, x, mask=None):
#         x = self.pwconv1(x)
#         x = self.act1(x)
#         x = self.dwconv(x)
#         x = self.pwconv2(x)
#         return x
#
#     def initialize(self):
#         weight_init(self)


class TraditionAttention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_heads = num_heads
        if self.num_heads == 0:
            self.num_heads = 1
        self.attention_dim = dim
        self.q = nn.Linear(dim, self.attention_dim, bias=bias)
        self.k = nn.Linear(dim, self.attention_dim, bias=bias)
        self.v = nn.Linear(dim, self.attention_dim, bias=bias)
        self.proj = nn.Linear(self.attention_dim, dim, bias=bias)

    def forward(self, x, mask=None):
        B, C, H, W = x.shape

        if mask is not None:
            x_m = x * mask
        else:
            x_m = x
        x = x.permute(0, 2, 3, 1)
        x_m = x_m.permute(0, 2, 3, 1)
        q = self.q(x_m)
        k = self.k(x_m)
        v = self.v(x)
        q = rearrange(q, 'b h w (head c) -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b h w (head c) -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b h w (head c) -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        x = rearrange(x, 'b head c (h w) -> b h w (head c)', h=H, w=W)
        x = self.proj(x)
        x = x.permute(0, 3, 1, 2)
        return x

    def initialize(self):
        weight_init(self)


class CA(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = TraditionAttention(dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

    def initialize(self):
        weight_init(self)


class MA(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = TraditionAttention(dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

    def initialize(self):
        weight_init(self)

class DA(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = DCNv3(dim)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

    def initialize(self):
        weight_init(self)


class Stage(nn.Module):
    def __init__(self, is_DA=False, dim=128):
        super().__init__()
        self.F_TA = CA(dim)
        if is_DA:
            self.TA = DA(dim)
        else:
            self.TA = MA(dim)
        self.Fuse = nn.Conv2d(2 * dim, dim, kernel_size=3, padding=1)
        self.Fuse2 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1), nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(dim), nn.ReLU(inplace=True))
        self.is_DA = is_DA
    def forward(self, x, side_x, mask):
        N, C, H, W = x.shape
        mask = F.interpolate(mask, size=x.size()[2:], mode='bilinear')
        mask_d = mask.detach()
        xf = self.F_TA(x)
        if self.is_DA:
            mask_d = edge_prediction(mask_d)
        mask_d = torch.sigmoid(mask_d)
        x = self.TA(x, mask_d)
        x = torch.cat((xf, x), 1)
        x = x.view(N, 2 * C, H, W)
        x = self.Fuse(x)
        if side_x is not None:
            D = self.Fuse2(side_x + side_x * x)
        else:
            D = self.Fuse2(x)
        return D

    def initialize(self):
        weight_init(self)


class GlobalGuider(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels * 4, channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.ReLU(True)

    def forward(self, input1, input2, input3, inputs4):
        fuse = torch.cat((input1, input2, input3, inputs4), 1)
        fuse = self.act1(self.bn1(self.conv1(fuse)))
        fuse = self.act2(self.bn2(self.conv2(fuse)))
        return fuse

    def initialize(self):
        weight_init(self)


class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()

        self.side_conv1 = nn.Conv2d(512, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv2 = nn.Conv2d(320, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv3 = nn.Conv2d(128, channels, kernel_size=3, stride=1, padding=1)
        self.side_conv4 = nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1)

        self.conv_block = GlobalGuider(channels)

        self.fuse1 = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channels))
        self.fuse2 = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channels))
        self.fuse3 = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channels))
        self.fuse4 = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(channels))

        self.Stage4 = Stage(is_DA=False, dim=channels)
        self.Stage3 = Stage(is_DA=False, dim=channels)
        self.Stage2 = Stage(is_DA=True, dim=channels)
        self.Stage1 = Stage(is_DA=True, dim=channels)

        self.predtrans1 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.predtrans5 = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

        self.initialize()

    def forward(self, E4, E3, E2, E1, shape):
        E4, E3, E2, E1 = self.side_conv1(E4), self.side_conv2(E3), self.side_conv3(E2), self.side_conv4(E1)

        P4 = F.interpolate(E4, size=E1.size()[2:], mode='bilinear')
        P3 = F.interpolate(E3, size=E1.size()[2:], mode='bilinear')
        P2 = F.interpolate(E2, size=E1.size()[2:], mode='bilinear')

        E5 = self.conv_block(P4, P3, P2, E1)
        P5 = self.predtrans5(E5)
        out5 = F.interpolate(P5, size=shape, mode='bilinear')

        E1 = torch.cat((E1, E5), 1)
        E1 = F.relu(self.fuse4(E1), inplace=True)
        E5 = F.interpolate(E5, size=E2.size()[2:], mode='bilinear')
        E2 = torch.cat((E2, E5), 1)
        E2 = F.relu(self.fuse3(E2), inplace=True)
        E5 = F.interpolate(E5, size=E3.size()[2:], mode='bilinear')
        E3 = torch.cat((E3, E5), 1)
        E3 = F.relu(self.fuse2(E3), inplace=True)
        E5 = F.interpolate(E5, size=E4.size()[2:], mode='bilinear')
        E4 = torch.cat((E4, E5), 1)
        E4 = F.relu(self.fuse1(E4), inplace=True)

        D4 = self.Stage4(E4, None, P5)
        D4 = F.interpolate(D4, size=E3.size()[2:], mode='bilinear')

        D3 = self.Stage3(D4, E3, P5)
        D3 = F.interpolate(D3, size=E2.size()[2:], mode='bilinear')

        D2 = self.Stage2(D3, E2, P5)
        D2 = F.interpolate(D2, size=E1.size()[2:], mode='bilinear')

        D1 = self.Stage1(D2, E1, P5)
        P1 = self.predtrans1(D1)

        out1 = F.interpolate(P1, size=shape, mode='bilinear')
        return out5, out1

    def initialize(self):
        weight_init(self)

