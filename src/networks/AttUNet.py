"""
    :filename AttUNet.py
    :author Tibor Kubik
    :email xkubik34@stud.fit.vutbr.cz

    Attention U-Net architecture.
    The original U-Net paper: https://arxiv.org/abs/1804.03999.

    Original implementation comes from following source file and was edited according to my purposes:
    https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/Models.py.

    Original author:
    Bjarte Mehus Sunde, 2018

    Original author's mail:
    BjarteSunde@outlook.com

    Licence:
    MIT License

    Copyright (c) 2018 Bjarte Mehus Sunde

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import torch

from torch import nn

from src.networks.UNet import UpConv


class DoubleConvBatchNorm(nn.Module):
    """Double convolution with additional BN: Conv -> BN -> ReLU -> Conv -> BN -> ReLU."""

    def __init__(self, in_channels, out_channels):
        super(DoubleConvBatchNorm, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),  # Bias = False is needed if I do not want to get the Batch Norm cancelled by conv

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),  # Bias = False is needed if I do not want to get the Batch Norm cancelled by conv

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, image):
        return self.conv(image)


class AttentionBlock(nn.Module):
    """Attention block in Attention U-Net. See the scheme in the original paper."""

    def __init__(self, g, x_l, f_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(x_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttUNet(nn.Module):
    """Attention U-Net architecture implementation."""

    def __init__(self, in_channels=1, out_channels=1):
        super(AttUNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # features = [64, 128, 256, 512, 1024]
        features = [32, 64, 128, 256, 512]
        # features = [16, 32, 64, 128, 256]
        self.Conv1 = DoubleConvBatchNorm(in_channels=in_channels, out_channels=features[0])
        self.Conv2 = DoubleConvBatchNorm(in_channels=features[0], out_channels=features[1])
        self.Conv3 = DoubleConvBatchNorm(in_channels=features[1], out_channels=features[2])
        # self.Conv4 = DoubleConvBatchNorm(in_channels=features[2], out_channels=features[3])
        # self.Conv5 = DoubleConvBatchNorm(in_channels=features[3], out_channels=features[4])

        # self.Up5 = UpConv(ch_in=features[4], ch_out=features[3])
        # self.Att5 = AttentionBlock(g=features[3], x_l=features[3], f_int=features[2])
        # self.Up_conv5 = DoubleConvBatchNorm(in_channels=features[4], out_channels=features[3])
        #
        # self.Up4 = UpConv(ch_in=features[3], ch_out=features[2])
        # self.Att4 = AttentionBlock(g=features[2], x_l=features[2], f_int=features[1])
        # self.Up_conv4 = DoubleConvBatchNorm(in_channels=features[3], out_channels=features[2])

        self.Up3 = UpConv(ch_in=features[2], ch_out=features[1])
        self.Att3 = AttentionBlock(g=features[1], x_l=features[1], f_int=features[0])
        self.Up_conv3 = DoubleConvBatchNorm(in_channels=features[2], out_channels=features[1])

        self.Up2 = UpConv(ch_in=features[1], ch_out=features[0])
        self.Att2 = AttentionBlock(g=features[0], x_l=features[0], f_int=32)
        self.Up_conv2 = DoubleConvBatchNorm(in_channels=features[1], out_channels=features[0])

        self.Conv_1x1 = nn.Conv2d(features[0], out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        # x4 = self.Maxpool(x3)
        # x4 = self.Conv4(x4)
        #
        # x5 = self.Maxpool(x4)
        # x5 = self.Conv5(x5)
        #
        # d5 = self.Up5(x5)
        # x4 = self.Att5(g=d5, x=x4)
        # d5 = torch.cat((x4, d5), dim=1)
        # d5 = self.Up_conv5(d5)
        #
        # d4 = self.Up4(d5)
        # x3 = self.Att4(g=d4, x=x3)
        # d4 = torch.cat((x3, d4), dim=1)
        # d4 = self.Up_conv4(d4)

        d3 = self.Up3(x3)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
