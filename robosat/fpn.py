"""Feature Pyramid Network (FPN) on top of ResNet. Comes with task-specific heads on top of it.

See:
- https://arxiv.org/abs/1612.03144 - Feature Pyramid Networks for Object Detection
- http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf - A Unified Architecture for Instance
                                                               and Semantic Segmentation

"""

import torch
import torch.nn as nn

from torchvision.models import resnet50


class FPN(nn.Module):
    """Feature Pyramid Network (FPN): top-down architecture with lateral connections.
       Can be used as feature extractor for object detection or segmentation.
    """

    def __init__(self, num_filters=256, pretrained=True):
        """Creates an `FPN` instance for feature extraction.

        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super().__init__()

        self.resnet = resnet50(pretrained=pretrained)

        # Access resnet directly in forward pass; do not store refs here due to
        # https://github.com/pytorch/pytorch/issues/8392

        self.lateral4 = Conv1x1(2048, num_filters)
        self.lateral3 = Conv1x1(1024, num_filters)
        self.lateral2 = Conv1x1(512, num_filters)
        self.lateral1 = Conv1x1(256, num_filters)

        self.smooth4 = Conv3x3(num_filters, num_filters)
        self.smooth3 = Conv3x3(num_filters, num_filters)
        self.smooth2 = Conv3x3(num_filters, num_filters)
        self.smooth1 = Conv3x3(num_filters, num_filters)

    def forward(self, x):
        # Bottom-up pathway, from ResNet

        enc0 = self.resnet.conv1(x)
        enc0 = self.resnet.bn1(enc0)
        enc0 = self.resnet.relu(enc0)
        enc0 = self.resnet.maxpool(enc0)

        enc1 = self.resnet.layer1(enc0)
        enc2 = self.resnet.layer2(enc1)
        enc3 = self.resnet.layer3(enc2)
        enc4 = self.resnet.layer4(enc3)

        # Lateral connections

        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)

        # Top-down pathway

        map4 = lateral4
        map3 = lateral3 + nn.functional.upsample(map4, scale_factor=2, mode="nearest")
        map2 = lateral2 + nn.functional.upsample(map3, scale_factor=2, mode="nearest")
        map1 = lateral1 + nn.functional.upsample(map2, scale_factor=2, mode="nearest")

        # Reduce aliasing effect of upsampling

        map4 = self.smooth4(map4)
        map3 = self.smooth3(map3)
        map2 = self.smooth2(map2)
        map1 = self.smooth1(map1)

        return map1, map2, map3, map4


class FPNSegmentation(nn.Module):
    """Semantic segmentation model on top of a Feature Pyramid Network (FPN).
    """

    def __init__(self, num_classes, num_filters=128, num_filters_fpn=256, pretrained=True):
        """Creates an `FPNSegmentation` instance for feature extraction.

        Args:
          num_classes: number of classes to predict
          num_filters: the number of filters in each segmentation head pyramid level
          num_filters_fpn: the number of filters in each FPN output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.

        self.fpn = FPN(num_filters=num_filters_fpn, pretrained=pretrained)

        # The segmentation heads on top of the FPN

        self.head1 = nn.Sequential(Conv3x3(num_filters_fpn, num_filters), Conv3x3(num_filters, num_filters))
        self.head2 = nn.Sequential(Conv3x3(num_filters_fpn, num_filters), Conv3x3(num_filters, num_filters))
        self.head3 = nn.Sequential(Conv3x3(num_filters_fpn, num_filters), Conv3x3(num_filters, num_filters))
        self.head4 = nn.Sequential(Conv3x3(num_filters_fpn, num_filters), Conv3x3(num_filters, num_filters))

        self.final = nn.Conv2d(4 * num_filters, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        map1, map2, map3, map4 = self.fpn(x)

        map4 = nn.functional.upsample(self.head4(map4), scale_factor=8, mode="nearest")
        map3 = nn.functional.upsample(self.head3(map3), scale_factor=4, mode="nearest")
        map2 = nn.functional.upsample(self.head2(map2), scale_factor=2, mode="nearest")
        map1 = self.head1(map1)

        final = self.final(torch.cat([map4, map3, map2, map1], dim=1))

        return nn.functional.upsample(final, scale_factor=4, mode="bilinear", align_corners=False)


class Conv1x1(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.block = nn.Conv2d(num_in, num_out, kernel_size=1, bias=False)

    def forward(self, x):
        return self.block(x)


class Conv3x3(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.block = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return self.block(x)
