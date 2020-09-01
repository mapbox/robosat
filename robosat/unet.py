"""U-Net inspired encoder-decoder architecture with a ResNet encoder as proposed by Alexander Buslaev.

See:
- https://arxiv.org/abs/1505.04597 - U-Net: Convolutional Networks for Biomedical Image Segmentation
- https://arxiv.org/abs/1411.4038  - Fully Convolutional Networks for Semantic Segmentation
- https://arxiv.org/abs/1512.03385 - Deep Residual Learning for Image Recognition
- https://arxiv.org/abs/1801.05746 - TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation
- https://arxiv.org/abs/1806.00844 - TernausNetV2: Fully Convolutional Network for Instance Segmentation

"""

import torch
import torch.nn as nn

from robosat.efficientnet import efficientnet0


class ConvRelu(nn.Module):
    """3x3 convolution followed by ReLU activation building block.
    """

    def __init__(self, num_in, num_out):
        """Creates a `ConvReLU` building block.

        Args:
          num_in: number of input feature maps
          num_out: number of output feature maps
        """

        super().__init__()

        self.block = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        """The networks forward pass for which autograd synthesizes the backwards pass.

        Args:
          x: the input tensor

        Returns:
          The networks output tensor.
        """

        return nn.functional.relu(self.block(x), inplace=True)


class DecoderBlock(nn.Module):
    """Decoder building block upsampling resolution by a factor of two.
    """

    def __init__(self, num_in, num_out):
        """Creates a `DecoderBlock` building block.

        Args:
          num_in: number of input feature maps
          num_out: number of output feature maps
        """

        super().__init__()

        self.block = ConvRelu(num_in, num_out)

    def forward(self, x):
        """The networks forward pass for which autograd synthesizes the backwards pass.

        Args:
          x: the input tensor

        Returns:
          The networks output tensor.
        """

        return self.block(nn.functional.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    """The "U-Net" architecture for semantic segmentation, adapted by changing the encoder to a ResNet feature extractor.

       Also known as AlbuNet due to its inventor Alexander Buslaev.
    """

    def __init__(self, num_classes, num_filters=32, pretrained=True):
        """Creates an `UNet` instance for semantic segmentation.

        Args:
          num_classes: number of classes to predict.
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super().__init__()

        # Todo: make input channels configurable, not hard-coded to three channels for RGB

        self.net = efficientnet0(pretrained=pretrained)

        # Access backbone directly in forward pass; do not store refs here due to
        # https://github.com/pytorch/pytorch/issues/8392

        self.center = DecoderBlock(1280, num_filters * 8)

        self.dec0 = DecoderBlock(1280 + num_filters * 8, num_filters * 8)
        self.dec1 = DecoderBlock(112 + num_filters * 8, num_filters * 8)
        self.dec2 = DecoderBlock(40 + num_filters * 8, num_filters * 2)
        self.dec3 = DecoderBlock(24 + num_filters * 2, num_filters * 2 * 2)
        self.dec4 = DecoderBlock(num_filters * 2 * 2, num_filters)
        self.dec5 = ConvRelu(num_filters, num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        """The networks forward pass for which autograd synthesizes the backwards pass.

        Args:
          x: the input tensor

        Returns:
          The networks output tensor.
        """
        size = x.size()
        assert size[-1] % 32 == 0 and size[-2] % 32 == 0, "image resolution has to be divisible by 32 for backbone"

        # 1, 3, 512, 512
        enc0 = self.net.conv1(x)
        enc0 = self.net.bn1(enc0)
        enc0 = self.net.relu(enc0)
        # 1, 32, 256, 256
        enc0 = self.net.layer1(enc0)
        # 1, 16, 256, 256

        enc1 = self.net.layer2(enc0)
        # 1, 24, 128, 128

        enc2 = self.net.layer3(enc1)
        # 1, 40, 64, 64

        enc3 = self.net.layer4(enc2)
        # 1, 80, 32, 32
        enc3 = self.net.layer5(enc3)
        # 1, 112, 32, 32

        enc4 = self.net.layer6(enc3)
        # 1, 192, 16, 16
        enc4 = self.net.layer7(enc4)
        # 1, 320, 16, 16
        enc4 = self.net.features(enc4)
        # 1, 1280, 16, 16

        center = self.center(nn.functional.max_pool2d(enc4, kernel_size=2, stride=2))

        dec0 = self.dec0(torch.cat([enc4, center], dim=1))
        dec1 = self.dec1(torch.cat([enc3, dec0], dim=1))
        dec2 = self.dec2(torch.cat([enc2, dec1], dim=1))
        dec3 = self.dec3(torch.cat([enc1, dec2], dim=1))
        dec4 = self.dec4(dec3)
        dec5 = self.dec5(dec4)

        return self.final(dec5)
