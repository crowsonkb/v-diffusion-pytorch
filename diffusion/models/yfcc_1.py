import math

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, dropout_last=True):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.ReLU(inplace=True) if dropout_last else nn.Identity(),
        ], skip)


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class SelfAttention2d(nn.Module):
    def __init__(self, c_in, n_head=1, dropout_rate=0.1):
        super().__init__()
        assert c_in % n_head == 0
        self.norm = nn.GroupNorm(1, c_in)
        self.n_head = n_head
        self.qkv_proj = nn.Conv2d(c_in, c_in * 3, 1)
        self.out_proj = nn.Conv2d(c_in, c_in, 1)
        self.dropout = nn.Identity()  # nn.Dropout2d(dropout_rate, inplace=True)

    def forward(self, input):
        n, c, h, w = input.shape
        qkv = self.qkv_proj(self.norm(input))
        qkv = qkv.view([n, self.n_head * 3, c // self.n_head, h * w]).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=1)
        scale = k.shape[3]**-0.25
        att = ((q * scale) @ (k.transpose(2, 3) * scale)).softmax(3)
        y = (att @ v).transpose(2, 3).contiguous().view([n, c, h, w])
        return input + self.dropout(self.out_proj(y))


def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])


class YFCC1Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = (3, 512, 512)
        self.min_t = 0.
        self.max_t = 1.

        c = 128  # The base channel count

        # The inputs to timestep_embed will approximately fall into the range
        # -10 to 10, so use std 0.2 for the Fourier Features.
        self.timestep_embed = FourierFeatures(1, 16)

        self.net = nn.Sequential(   # 512x512
            ResConvBlock(3 + 16, c, c),
            ResConvBlock(c, c, c),
            ResConvBlock(c, c, c),
            ResConvBlock(c, c, c),
            SkipBlock([
                nn.AvgPool2d(2),  # 512x512 -> 256x256
                ResConvBlock(c, c, c),
                ResConvBlock(c, c, c),
                ResConvBlock(c, c, c),
                ResConvBlock(c, c, c),
                SkipBlock([
                    nn.AvgPool2d(2),  # 256x256 -> 128x128
                    ResConvBlock(c, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    SkipBlock([
                        nn.AvgPool2d(2),  # 128x128 -> 64x64
                        ResConvBlock(c * 2, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        SkipBlock([
                            nn.AvgPool2d(2),  # 64x64 -> 32x32
                            ResConvBlock(c * 2, c * 4, c * 4),
                            ResConvBlock(c * 4, c * 4, c * 4),
                            ResConvBlock(c * 4, c * 4, c * 4),
                            ResConvBlock(c * 4, c * 4, c * 4),
                            SkipBlock([
                                nn.AvgPool2d(2),  # 32x32 -> 16x16
                                ResConvBlock(c * 4, c * 4, c * 4),
                                SelfAttention2d(c * 4, c * 4 // 64),
                                ResConvBlock(c * 4, c * 4, c * 4),
                                SelfAttention2d(c * 4, c * 4 // 64),
                                ResConvBlock(c * 4, c * 4, c * 4),
                                SelfAttention2d(c * 4, c * 4 // 64),
                                ResConvBlock(c * 4, c * 4, c * 4),
                                SelfAttention2d(c * 4, c * 4 // 64),
                                SkipBlock([
                                    nn.AvgPool2d(2),  # 16x16 -> 8x8
                                    ResConvBlock(c * 4, c * 8, c * 8),
                                    SelfAttention2d(c * 8, c * 8 // 64),
                                    ResConvBlock(c * 8, c * 8, c * 8),
                                    SelfAttention2d(c * 8, c * 8 // 64),
                                    ResConvBlock(c * 8, c * 8, c * 8),
                                    SelfAttention2d(c * 8, c * 8 // 64),
                                    ResConvBlock(c * 8, c * 8, c * 8),
                                    SelfAttention2d(c * 8, c * 8 // 64),
                                    SkipBlock([
                                        nn.AvgPool2d(2),  # 8x8 -> 4x4
                                        ResConvBlock(c * 8, c * 8, c * 8),
                                        SelfAttention2d(c * 8, c * 8 // 64),
                                        ResConvBlock(c * 8, c * 8, c * 8),
                                        SelfAttention2d(c * 8, c * 8 // 64),
                                        ResConvBlock(c * 8, c * 8, c * 8),
                                        SelfAttention2d(c * 8, c * 8 // 64),
                                        ResConvBlock(c * 8, c * 8, c * 8),
                                        SelfAttention2d(c * 8, c * 8 // 64),
                                        ResConvBlock(c * 8, c * 8, c * 8),
                                        SelfAttention2d(c * 8, c * 8 // 64),
                                        ResConvBlock(c * 8, c * 8, c * 8),
                                        SelfAttention2d(c * 8, c * 8 // 64),
                                        ResConvBlock(c * 8, c * 8, c * 8),
                                        SelfAttention2d(c * 8, c * 8 // 64),
                                        ResConvBlock(c * 8, c * 8, c * 8),
                                        SelfAttention2d(c * 8, c * 8 // 64),
                                        nn.Upsample(scale_factor=2, mode='bilinear',
                                                    align_corners=False),
                                    ]),
                                    ResConvBlock(c * 16, c * 8, c * 8),
                                    SelfAttention2d(c * 8, c * 8 // 64),
                                    ResConvBlock(c * 8, c * 8, c * 8),
                                    SelfAttention2d(c * 8, c * 8 // 64),
                                    ResConvBlock(c * 8, c * 8, c * 8),
                                    SelfAttention2d(c * 8, c * 8 // 64),
                                    ResConvBlock(c * 8, c * 8, c * 4),
                                    SelfAttention2d(c * 4, c * 4 // 64),
                                    nn.Upsample(scale_factor=2, mode='bilinear',
                                                align_corners=False),
                                ]),
                                ResConvBlock(c * 8, c * 4, c * 4),
                                SelfAttention2d(c * 4, c * 4 // 64),
                                ResConvBlock(c * 4, c * 4, c * 4),
                                SelfAttention2d(c * 4, c * 4 // 64),
                                ResConvBlock(c * 4, c * 4, c * 4),
                                SelfAttention2d(c * 4, c * 4 // 64),
                                ResConvBlock(c * 4, c * 4, c * 4),
                                SelfAttention2d(c * 4, c * 4 // 64),
                                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                            ]),
                            ResConvBlock(c * 8, c * 4, c * 4),
                            ResConvBlock(c * 4, c * 4, c * 4),
                            ResConvBlock(c * 4, c * 4, c * 4),
                            ResConvBlock(c * 4, c * 4, c * 2),
                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                        ]),
                        ResConvBlock(c * 4, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    ]),
                    ResConvBlock(c * 4, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ]),
                ResConvBlock(c * 2, c, c),
                ResConvBlock(c, c, c),
                ResConvBlock(c, c, c),
                ResConvBlock(c, c, c),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ]),
            ResConvBlock(c * 2, c, c),
            ResConvBlock(c, c, c),
            ResConvBlock(c, c, c),
            ResConvBlock(c, c, 3, dropout_last=False),
        )

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        return self.net(torch.cat([input, timestep_embed], dim=1))
