"""
    :filename UNet.py
    :author Tibor Kubik
    :email xkubik34@stud.fit.vutbr.cz

    Modified U-Net architecutre.
    The original U-Net paper: https://arxiv.org/abs/1505.04597.

    This implementation contains modified U-Net architecture.
    The modification is in terms of added Batch Normalisation layer between Convolution and ReLU layers.
    Another modification is the possibility of the usage of upsampling in the decoder part.
    This implemenation also uses 1:1 input output dimension sizes, i.e., if an input has dimensions of c x 128 x 128,
    the output is o_x x 128 x 128.
"""

import torch
import torchvision.transforms.functional as TF

from torch import nn


class DoubleConv(nn.Module):
    """Double convolution: Conv -> ReLU -> Conv -> ReLU."""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),

            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, image):
        return self.conv(image)


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
                      bias=False),  # Bias = False is needed if I do not want to get the Batch Norm cancelled by conv

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),  # Bias = False is needed if I do not want to get the Batch Norm cancelled by conv

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, image):
        return self.conv(image)


class UpConv(nn.Module):
    """Upconvolution with added BN."""
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class UNet(nn.Module):
    """
    Modified U-Net architecture.
    What can be specified:

    in_channels:    The number of input feature channels. In this work is used one input channel.
    out_channels:   The number of output channels. It should correspond with the number of output segmented classes.
                    In case of the landmark regression, it should correspond with the number of detected landmarks
                    in the picture.
    batch_norm:     This flag specifies whether to add addition BN layers. Recommended value is True, as it helps
                    during the training.
    decoder_mode:   It specifies whether the Transposed Convolution layers should be used in the decoder part or the
                    Bilinear Upsampling.
    """
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 batch_norm=True,
                 decoder_mode='upconv'):
        super(UNet, self).__init__()

        assert(decoder_mode == 'upconv' or decoder_mode == 'upsample')

        self.batch_norm = batch_norm
        self.decoder_mode = decoder_mode

        features = [64, 128, 256, 512]

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder part
        for feature in features:
            if self.batch_norm:
                self.downs.append(
                    DoubleConvBatchNorm(in_channels=in_channels,
                                        out_channels=feature
                                        )
                )
            else:
                self.downs.append(
                    DoubleConv(in_channels=in_channels,
                               out_channels=feature
                               )
                )
            in_channels = feature

        # Decoder part
        for feature in reversed(features):
            if self.decoder_mode == 'upconv':
                self.ups.append(
                    nn.ConvTranspose2d(
                        in_channels=feature*2,
                        out_channels=feature,
                        kernel_size=2,
                        stride=2
                    )
                )
            else:
                self.ups.append(
                    nn.Sequential(
                        nn.Upsample(mode='bilinear', scale_factor=2),
                        nn.Conv2d(feature*2, feature, kernel_size=1),
                    )
                )

            if self.batch_norm:
                self.ups.append(
                    DoubleConvBatchNorm(
                        in_channels=feature*2,
                        out_channels=feature
                    )
                )
            else:
                self.ups.append(
                    DoubleConv(
                        in_channels=feature*2,
                        out_channels=feature
                    )
                )

        if self.batch_norm:
            self.bottleneck = DoubleConvBatchNorm(
                in_channels=features[-1],
                out_channels=features[-1]*2,
            )
        else:
            self.bottleneck = DoubleConv(
                in_channels=features[-1],
                out_channels=features[-1] * 2,
            )

        self.final = nn.Conv2d(
            in_channels=features[0],
            out_channels=out_channels,  # the expected number of output channels, 1 for each landmark heatmap
            kernel_size=1  # the w and h of image would not be changed
        )

    def forward(self, image):
        skips_conns = []  # For skip connections storing, typical for Unet-like architectures

        for down_step in self.downs:
            image = down_step(image)
            skips_conns.append(image)
            image = self.max_pool(image)

        image = self.bottleneck(image)
        skips_conns = skips_conns[::-1]  # Reverse the skip conns list to access proper element

        for idx in range(0, len(self.ups), 2):
            image = self.ups[idx](image)
            skips_conn = skips_conns[idx//2]

            if image.shape != skips_conn.shape:
                image = TF.resize(image, size=skips_conn.shape[2:])  # If they dont match before concatenating, reshape

            concat = torch.cat((skips_conn, image), dim=1)

            image = self.ups[idx+1](concat)

        return self.final(image)
