"""Feature Pyramid Attention blocks

See:
- https://arxiv.org/abs/1805.10180 - Pyramid Attention Network for Semantic Segmentation

"""

import torch.nn as nn


class FeaturePyramidAttention(nn.Module):
    """Feature Pyramid Attetion (FPA) block
       See https://arxiv.org/abs/1805.10180 Figure 3 b
    """

    def __init__(self, num_in, num_out):
        super().__init__()

        # no batch norm for tensors of shape NxCx1x1
        self.top1x1 = nn.Sequential(nn.Conv2d(num_in, num_out, 1, bias=False), nn.ReLU(inplace=True))

        self.mid1x1 = ConvBnRelu(num_in, num_out, 1)

        self.bot5x5 = ConvBnRelu(num_in, num_in, 5, stride=2, padding=2)
        self.bot3x3 = ConvBnRelu(num_in, num_in, 3, stride=2, padding=1)

        self.lat5x5 = ConvBnRelu(num_in, num_out, 5, stride=1, padding=2)
        self.lat3x3 = ConvBnRelu(num_in, num_out, 3, stride=1, padding=1)

    def forward(self, x):
        assert x.size()[-1] % 8 == 0 and x.size()[-2] % 8 == 0, "size has to be divisible by 8 for fpa"

        # global pooling top pathway
        top = self.top1x1(nn.functional.adaptive_avg_pool2d(x, 1))
        top = nn.functional.interpolate(top, size=x.size()[-2:], mode="bilinear")

        # conv middle pathway
        mid = self.mid1x1(x)

        # multi-scale bottom and lateral pathways
        bot0 = self.bot5x5(x)
        bot1 = self.bot3x3(bot0)

        lat0 = self.lat5x5(bot0)
        lat1 = self.lat3x3(bot1)

        # upward accumulation pathways
        up = lat0 + nn.functional.interpolate(lat1, scale_factor=2, mode="bilinear")
        up = nn.functional.interpolate(up, scale_factor=2, mode="bilinear")

        return up * mid + top


def ConvBnRelu(num_in, num_out, kernel_size, stride=1, padding=0, bias=False):
    return nn.Sequential(
        nn.Conv2d(num_in, num_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(num_out, num_out),
        nn.ReLU(inplace=True),
    )
