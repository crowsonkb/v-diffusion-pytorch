import math

import torch
from torch import nn

from .. import utils


class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.skip = skip if skip else nn.Identity()
        self.main = nn.Sequential(*main)

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
        ], skip)


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.skip(input), self.main(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])


class WikiArt128Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = (3, 128, 128)
        self.min_t = utils.get_ddpm_schedule(torch.tensor(0.)).item()
        self.max_t = utils.get_ddpm_schedule(torch.tensor(1.)).item()

        c = 128  # The base channel count
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8]

        self.timestep_embed = FourierFeatures(1, 16, std=0.2)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.net = nn.Sequential(   # 128x128
            ResConvBlock(3 + 16, cs[0], cs[0]),
            ResConvBlock(cs[0], cs[0], cs[0]),
            ResConvBlock(cs[0], cs[0], cs[0]),
            ResConvBlock(cs[0], cs[0], cs[0]),
            SkipBlock([
                self.down,  # 128x128 -> 64x64
                ResConvBlock(cs[0], cs[1], cs[1]),
                ResConvBlock(cs[1], cs[1], cs[1]),
                ResConvBlock(cs[1], cs[1], cs[1]),
                ResConvBlock(cs[1], cs[1], cs[1]),
                SkipBlock([
                    self.down,  # 64x64 -> 32x32
                    ResConvBlock(cs[1], cs[2], cs[2]),
                    ResConvBlock(cs[2], cs[2], cs[2]),
                    ResConvBlock(cs[2], cs[2], cs[2]),
                    ResConvBlock(cs[2], cs[2], cs[2]),
                    SkipBlock([
                        self.down,  # 32x32 -> 16x16
                        ResConvBlock(cs[2], cs[3], cs[3]),
                        ResConvBlock(cs[3], cs[3], cs[3]),
                        ResConvBlock(cs[3], cs[3], cs[3]),
                        ResConvBlock(cs[3], cs[3], cs[3]),
                        SkipBlock([
                            self.down,  # 16x16 -> 8x8
                            ResConvBlock(cs[3], cs[4], cs[4]),
                            ResConvBlock(cs[4], cs[4], cs[4]),
                            ResConvBlock(cs[4], cs[4], cs[4]),
                            ResConvBlock(cs[4], cs[4], cs[4]),
                            SkipBlock([
                                self.down,  # 8x8 -> 4x4
                                ResConvBlock(cs[4], cs[5], cs[5]),
                                ResConvBlock(cs[5], cs[5], cs[5]),
                                ResConvBlock(cs[5], cs[5], cs[5]),
                                ResConvBlock(cs[5], cs[5], cs[5]),
                                ResConvBlock(cs[5], cs[5], cs[5]),
                                ResConvBlock(cs[5], cs[5], cs[5]),
                                ResConvBlock(cs[5], cs[5], cs[5]),
                                ResConvBlock(cs[5], cs[5], cs[4]),
                                self.up,  # 4x4 -> 8x8
                            ]),
                            ResConvBlock(cs[4] * 2, cs[4], cs[4]),
                            ResConvBlock(cs[4], cs[4], cs[4]),
                            ResConvBlock(cs[4], cs[4], cs[4]),
                            ResConvBlock(cs[4], cs[4], cs[3]),
                            self.up,  # 8x8 -> 16x16
                        ]),
                        ResConvBlock(cs[3] * 2, cs[3], cs[3]),
                        ResConvBlock(cs[3], cs[3], cs[3]),
                        ResConvBlock(cs[3], cs[3], cs[3]),
                        ResConvBlock(cs[3], cs[3], cs[2]),
                        self.up,  # 16x16 -> 32x32
                    ]),
                    ResConvBlock(cs[2] * 2, cs[2], cs[2]),
                    ResConvBlock(cs[2], cs[2], cs[2]),
                    ResConvBlock(cs[2], cs[2], cs[2]),
                    ResConvBlock(cs[2], cs[2], cs[1]),
                    self.up,  # 32x32 -> 64x64
                ]),
                ResConvBlock(cs[1] * 2, cs[1], cs[1]),
                ResConvBlock(cs[1], cs[1], cs[1]),
                ResConvBlock(cs[1], cs[1], cs[1]),
                ResConvBlock(cs[1], cs[1], cs[0]),
                self.up,  # 64x64 -> 128x128
            ]),
            ResConvBlock(cs[0] * 2, cs[0], cs[0]),
            ResConvBlock(cs[0], cs[0], cs[0]),
            ResConvBlock(cs[0], cs[0], cs[0]),
            ResConvBlock(cs[0], cs[0], 3),
        )

    def forward(self, input, t):
        log_snr = utils.alpha_sigma_to_log_snr(*utils.t_to_alpha_sigma(t))
        timestep_embed = expand_to_planes(self.timestep_embed(log_snr[:, None]), input.shape)
        return self.net(torch.cat([input, timestep_embed], dim=1))
