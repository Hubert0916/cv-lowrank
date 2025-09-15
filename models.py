# models.py
"""
Model definitions for TinyCNN and its low-rank / depthwise variants.
Includes:
    - LowRankConv2d: Factorized convolution layer (kxk + 1x1).
    - DepthwiseSeparableConv2d: Depthwise (kxk) + Pointwise (1x1) convolution.
    - TinyCNN: Small CNN architecture supporting multiple convolution types.
"""

import torch.nn as nn


class LowRankConv2d(nn.Module):
    """
    Low-rank convolution layer inspired by SVD decomposition.

    Implements two-step convolution:
        1. kxk convolution projecting input channels -> rank (dimension reduction)
        2. 1x1 convolution projecting rank -> out_channels (channel mixing)

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolution kernel size (default: 3).
        rank (int): Bottleneck dimension (controls model capacity).
        stride (int): Convolution stride.
        padding (int): Padding (default: same padding).
        dilation (int): Dilation factor.
        bias (bool): Whether to use bias term in the 1x1 convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        rank=16,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
    ):
        super().__init__()
        k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        if padding is None:
            padding = k // 2  # use "same" padding for preserving spatial size

        # First convolution: kxk, Cin -> rank, no bias
        self.conv_k = nn.Conv2d(
            in_channels,
            rank,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )

        # Second convolution: 1x1, rank -> Cout, bias allowed
        self.conv_1 = nn.Conv2d(rank, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.conv_k(x)
        x = self.conv_1(x)
        return x


class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise separable convolution layer.

    Implements:
        - Depthwise convolution: independent kxk conv per channel (groups=Cin)
        - Pointwise convolution: 1x1 conv for channel mixing

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolution kernel size (default: 3).
        stride (int): Convolution stride.
        padding (int): Padding (default: same padding).
        dilation (int): Dilation factor.
        bias (bool): Whether to use bias term in the pointwise convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
    ):
        super().__init__()
        k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        if padding is None:
            padding = k // 2

        # Depthwise convolution (groups=in_channels)
        self.dw = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )

        # Pointwise convolution (1x1)
        self.pw = nn.Conv2d(in_channels, out_channels,
                            kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pw(self.dw(x))


def conv3x3(in_ch, out_ch, stride=1):
    """
    Helper function: Standard 3x3 convolution with padding=1, no bias.
    """
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)


class TinyCNN(nn.Module):
    """
    Simple CNN architecture with selectable convolution blocks.

    Architecture:
        - Two convolutional blocks -> MaxPool
        - Two convolutional blocks -> MaxPool
        - Global average pooling -> Fully connected classifier

    Args:
        stem (int): Number of channels in the first layer (default: 64).
        num_classes (int): Number of output classes (default: 10).
        block (str): Type of convolution block ("base", "lowrank", "dwsep").
        rank (int): Rank used for low-rank convolution.
    """

    def __init__(self, stem=64, num_classes=10, block="base", rank=16):
        super().__init__()

        # Select convolution type
        Conv = conv3x3
        if block == "lowrank":
            def Conv(cin, cout, s=1):
                return LowRankConv2d(cin, cout, 3, rank=rank, stride=s)
        elif block == "dwsep":
            def Conv(cin, cout, s=1):
                return DepthwiseSeparableConv2d(cin, cout, 3, stride=s)

        # Feature extractor
        self.features = nn.Sequential(
            Conv(3, stem),
            nn.BatchNorm2d(stem),
            nn.ReLU(inplace=True),

            Conv(stem, stem),
            nn.BatchNorm2d(stem),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            Conv(stem, stem * 2),
            nn.BatchNorm2d(stem * 2),
            nn.ReLU(inplace=True),

            Conv(stem * 2, stem * 2),
            nn.BatchNorm2d(stem * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # global average pooling
            nn.Flatten(),
            nn.Linear(stem * 2, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
