import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True))


def deconv_block(in_channels, out_channels):
    return nn.ConvTranspose1d(in_channels,
                              out_channels,
                              kernel_size=2,
                              stride=2)


class UNet(nn.Module):
    def __init__(self, n_classes=3):
        super(UNet, self).__init__()
        self.encoder_conv1 = conv_block(1, 64)
        self.encoder_conv2 = conv_block(64, 128)
        self.encoder_conv3 = conv_block(128, 256)
        self.encoder_conv4 = conv_block(256, 512)
        self.encoder_conv5 = conv_block(512, 1024)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.decode_trans4 = deconv_block(1024, 512)
        self.decode_conv4 = conv_block(1024, 512)
        self.decode_trans3 = deconv_block(512, 256)
        self.decode_conv3 = conv_block(512, 256)
        self.decode_trans2 = deconv_block(256, 128)
        self.decode_conv2 = conv_block(256, 128)
        self.decode_trans1 = deconv_block(128, 64)
        self.decode_conv1 = conv_block(128, 64)
        self.out = nn.Conv1d(64, n_classes, kernel_size=1)

    def encode(self, seq):
        encode_x1 = self.encoder_conv1(seq)

        encode_x2 = self.max_pool(encode_x1)
        encode_x2 = self.encoder_conv2(encode_x2)

        encode_x3 = self.max_pool(encode_x2)
        encode_x3 = self.encoder_conv3(encode_x3)

        encode_x4 = self.max_pool(encode_x3)
        encode_x4 = self.encoder_conv4(encode_x4)

        encode_x5 = self.max_pool(encode_x4)
        encode_x5 = self.encoder_conv5(encode_x5)

        return encode_x1, encode_x2, encode_x3, encode_x4, encode_x5

    def forward(self, seq):
        encode_x1, encode_x2, encode_x3, encode_x4, encode_x5 = self.encode(
            seq)

        concat_x4 = torch.cat([self.decode_trans4(encode_x5), encode_x4], 1)
        decode_x4 = self.decode_conv4(concat_x4)

        concat_x3 = torch.cat([self.decode_trans3(encode_x4), encode_x3], 1)
        decode_x3 = self.decode_conv3(concat_x3)

        concat_x2 = torch.cat([self.decode_trans2(encode_x3), encode_x2], 1)
        decode_x2 = self.decode_conv2(concat_x2)

        concat_x1 = torch.cat([self.decode_trans1(encode_x2), encode_x1], 1)
        decode_x1 = self.decode_conv1(concat_x1)

        return self.out(decode_x1)
