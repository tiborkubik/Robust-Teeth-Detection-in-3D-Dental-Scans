"""
    :filename NestedUNet.py
    :author Tibor Kubik
    :email xkubik34@stud.fit.vutbr.cz

    Nested U-Net architecture.
    The original U-Net paper: https://arxiv.org/abs/1807.10165.

    Although I followed the paper during the implementation, I validated/edited my solution with the one from
    https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/Models.py#L485.
    Thus, I cite their code according to the licence.

    Original author:
    Malav Bateriwala, 2019

    Original author's mail:
    malav.b93@gmail.com

    Licence:
    MIT License

    Copyright (c) 2019 Malav Bateriwala

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


class ConvNested(nn.Module):
    """
    Nested double convolution.
    Three channels are needed.
    ReLU -> Conv(in, mid) -> BN -> Conv(mid, out) -> BN.
    """

    def __init__(self, in_ch, mid_ch, out_ch):
        super(ConvNested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output


class NestedUNet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1):
        super(NestedUNet, self).__init__()

        features = [64, 128, 256, 512, 1024]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = ConvNested(in_channels, features[0], features[0])

        self.conv1_0 = ConvNested(features[0], features[1], features[1])
        self.conv2_0 = ConvNested(features[1], features[2], features[2])
        self.conv3_0 = ConvNested(features[2], features[3], features[3])
        self.conv4_0 = ConvNested(features[3], features[4], features[4])

        self.conv0_1 = ConvNested(features[0] + features[1], features[0], features[0])
        self.conv1_1 = ConvNested(features[1] + features[2], features[1], features[1])
        self.conv2_1 = ConvNested(features[2] + features[3], features[2], features[2])
        self.conv3_1 = ConvNested(features[3] + features[4], features[3], features[3])

        self.conv0_2 = ConvNested(features[0] * 2 + features[1], features[0], features[0])
        self.conv1_2 = ConvNested(features[1] * 2 + features[2], features[1], features[1])
        self.conv2_2 = ConvNested(features[2] * 2 + features[3], features[2], features[2])

        self.conv0_3 = ConvNested(features[0] * 3 + features[1], features[0], features[0])
        self.conv1_3 = ConvNested(features[1] * 3 + features[2], features[1], features[1])

        self.conv0_4 = ConvNested(features[0] * 4 + features[1], features[0], features[0])

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)

        return output
