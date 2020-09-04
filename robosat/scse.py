"""Squeeze and Excitation blocks - attention for classification and segmentation

See:
- https://arxiv.org/abs/1709.01507 - Squeeze-and-Excitation Networks
- https://arxiv.org/abs/1803.02579 - Concurrent Spatial and Channel 'Squeeze & Excitation' in Fully Convolutional Networks

"""

import torch
import torch.nn as nn


class SpatialSqChannelEx(nn.Module):
    """Spatial Squeeze and Channel Excitation (cSE) block
       See https://arxiv.org/abs/1803.02579 Figure 1 b
    """

    def __init__(self, num_in, r):
        super().__init__()
        self.fc0 = Conv1x1(num_in, num_in // r)
        self.fc1 = Conv1x1(num_in // r, num_in)

    def forward(self, x):
        xx = nn.functional.adaptive_avg_pool2d(x, 1)
        xx = self.fc0(xx)
        xx = nn.functional.relu(xx, inplace=True)
        xx = self.fc1(xx)
        xx = torch.sigmoid(xx)
        return x * xx


class ChannelSqSpatialEx(nn.Module):
    """Channel Squeeze and Spatial Excitation (sSE) block
       See https://arxiv.org/abs/1803.02579 Figure 1 c
    """

    def __init__(self, num_in):
        super().__init__()
        self.conv = Conv1x1(num_in, 1)

    def forward(self, x):
        xx = self.conv(x)
        xx = torch.sigmoid(xx)
        return x * xx


class SpatialChannelSqChannelEx(nn.Module):
    """Concurrent Spatial and Channel Squeeze and Channel Excitation (csSE) block
       See https://arxiv.org/abs/1803.02579 Figure 1 d
    """

    def __init__(self, num_in, r=16):
        super().__init__()

        self.cse = SpatialSqChannelEx(num_in, r)
        self.sse = ChannelSqSpatialEx(num_in)

    def forward(self, x):
        return self.cse(x) + self.sse(x)


def Conv1x1(num_in, num_out):
    return nn.Conv2d(num_in, num_out, kernel_size=1, bias=False)
