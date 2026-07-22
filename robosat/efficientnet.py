"""EfficientNet architecture.

See:
- https://arxiv.org/abs/1905.11946 - EfficientNet
- https://arxiv.org/abs/1801.04381 - MobileNet V2
- https://arxiv.org/abs/1905.02244 - MobileNet V3
- https://arxiv.org/abs/1709.01507 - Squeeze-and-Excitation
- https://arxiv.org/abs/1803.02579 - Concurrent spatial and channel squeeze-and-excitation
- https://arxiv.org/abs/1812.01187 - Bag of Tricks for Image Classification with Convolutional Neural Networks


Known issues:

- Not using swish activation function: unclear where, if, and how
  much it helps. Needs more experimentation. See also MobileNet V3.

- Not using squeeze and excitation blocks: I had significantly worse
  results with scse blocks, and cse blocks alone did not help, too.
  Needs more experimentation as it was done on small datasets only.

- Not using DropConnect: no efficient native implementation in PyTorch.
  Unclear if and how much it helps over Dropout.
"""

import math
import collections

import torch
import torch.nn as nn


EfficientNetParam = collections.namedtuple("EfficientNetParam", [
    "width", "depth", "resolution", "dropout"])

EfficientNetParams = {
  "B0": EfficientNetParam(1.0, 1.0, 224, 0.2),
  "B1": EfficientNetParam(1.0, 1.1, 240, 0.2),
  "B2": EfficientNetParam(1.1, 1.2, 260, 0.3),
  "B3": EfficientNetParam(1.2, 1.4, 300, 0.3),
  "B4": EfficientNetParam(1.4, 1.8, 380, 0.4),
  "B5": EfficientNetParam(1.6, 2.2, 456, 0.4),
  "B6": EfficientNetParam(1.8, 2.6, 528, 0.5),
  "B7": EfficientNetParam(2.0, 3.1, 600, 0.5)}


def efficientnet0(pretrained=False, progress=False, num_classes=1000):
    return EfficientNet(param=EfficientNetParams["B0"], num_classes=num_classes)

def efficientnet1(pretrained=False, progress=False, num_classes=1000):
    return EfficientNet(param=EfficientNetParams["B1"], num_classes=num_classes)

def efficientnet2(pretrained=False, progress=False, num_classes=1000):
    return EfficientNet(param=EfficientNetParams["B2"], num_classes=num_classes)

def efficientnet3(pretrained=False, progress=False, num_classes=1000):
    return EfficientNet(param=EfficientNetParams["B3"], num_classes=num_classes)

def efficientnet4(pretrained=False, progress=False, num_classes=1000):
    return EfficientNet(param=EfficientNetParams["B4"], num_classes=num_classes)

def efficientnet5(pretrained=False, progress=False, num_classes=1000):
    return EfficientNet(param=EfficientNetParams["B5"], num_classes=num_classes)

def efficientnet6(pretrained=False, progress=False, num_classes=1000):
    return EfficientNet(param=EfficientNetParams["B6"], num_classes=num_classes)

def efficientnet7(pretrained=False, progress=False, num_classes=1000):
    return EfficientNet(param=EfficientNetParams["B7"], num_classes=num_classes)


class EfficientNet(nn.Module):
    def __init__(self, param, num_classes=1000):
        super().__init__()

        # For the exact scaling technique we follow the official implementation as the paper does not tell us
        # https://github.com/tensorflow/tpu/blob/01574500090fa9c011cb8418c61d442286720211/models/official/efficientnet/efficientnet_model.py#L101-L125

        def scaled_depth(n):
            return int(math.ceil(n * param.depth))

        # Snap number of channels to multiple of 8 for optimized implementations
        def scaled_width(n):
            n = n * param.width
            m = max(8, int(n + 8 / 2) // 8 * 8)

            if m < 0.9 * n:
                m = m + 8

            return int(m)

        self.conv1 = nn.Conv2d(3, scaled_width(32), kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(scaled_width(32))
        self.relu = nn.ReLU6(inplace=True)

        self.layer1 = self._make_layer(n=scaled_depth(1), expansion=1, cin=scaled_width(32), cout=scaled_width(16), kernel_size=3, stride=1)
        self.layer2 = self._make_layer(n=scaled_depth(2), expansion=6, cin=scaled_width(16), cout=scaled_width(24), kernel_size=3, stride=2)
        self.layer3 = self._make_layer(n=scaled_depth(2), expansion=6, cin=scaled_width(24), cout=scaled_width(40), kernel_size=5, stride=2)
        self.layer4 = self._make_layer(n=scaled_depth(3), expansion=6, cin=scaled_width(40), cout=scaled_width(80), kernel_size=3, stride=2)
        self.layer5 = self._make_layer(n=scaled_depth(3), expansion=6, cin=scaled_width(80), cout=scaled_width(112), kernel_size=5, stride=1)
        self.layer6 = self._make_layer(n=scaled_depth(4), expansion=6, cin=scaled_width(112), cout=scaled_width(192), kernel_size=5, stride=2)
        self.layer7 = self._make_layer(n=scaled_depth(1), expansion=6, cin=scaled_width(192), cout=scaled_width(320), kernel_size=3, stride=1)

        self.features = nn.Conv2d(scaled_width(320), scaled_width(1280), kernel_size=1, bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(param.dropout, inplace=True)
        self.fc = nn.Linear(scaled_width(1280), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        # Zero BatchNorm weight at end of res-blocks: identity by default
        # See https://arxiv.org/abs/1812.01187 Section 3.1
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.zeros_(m.linear[1].weight)


    def _make_layer(self, n, expansion, cin, cout, kernel_size=3, stride=1):
        layers = []

        for i in range(n):
            if i == 0:
                planes = cin
                expand = cin * expansion
                squeeze = cout
                stride = stride
            else:
                planes = cout
                expand = cout * expansion
                squeeze = cout
                stride = 1

            layers += [Bottleneck(planes, expand, squeeze, kernel_size=kernel_size, stride=stride)]

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.features(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class Bottleneck(nn.Module):
    def __init__(self, planes, expand, squeeze, kernel_size, stride):
        super().__init__()

        self.expand = nn.Identity() if planes == expand else nn.Sequential(
            nn.Conv2d(planes, expand, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand),
            nn.ReLU6(inplace=True))

        self.depthwise = nn.Sequential(
            nn.Conv2d(expand, expand, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=expand, bias=False),
            nn.BatchNorm2d(expand),
            nn.ReLU6(inplace=True))

        self.linear = nn.Sequential(
            nn.Conv2d(expand, squeeze, kernel_size=1, bias=False),
            nn.BatchNorm2d(squeeze))

        # Make all blocks skip-able via AvgPool + 1x1 Conv
        # See https://arxiv.org/abs/1812.01187 Figure 2 c

        downsample = []

        if stride != 1:
            downsample += [nn.AvgPool2d(kernel_size=stride, stride=stride)]

        if planes != squeeze:
            downsample += [
                nn.Conv2d(planes, squeeze, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(squeeze)]

        self.downsample = nn.Identity() if not downsample else nn.Sequential(*downsample)


    def forward(self, x):
        xx = self.expand(x)
        xx = self.depthwise(xx)
        xx = self.linear(xx)

        x = self.downsample(x)
        xx.add_(x)

        return xx
