import numpy as np
from scipy import ndimage
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

def conv3x3(in_planes, out_planes, stride=1, dilation=1, bias=False):
    "3x3 convolution with padding"

    kernel_size = np.asarray((3, 3))

    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2

    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=full_padding,
        dilation=dilation,
        bias=bias,
    )

def conv1x1(in_planes, out_planes, stride=1, dilation=1, bias=False):
    "1x1 convolution with padding"

    kernel_size = np.asarray((1, 1))

    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size

    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size - 1) // 2

    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=full_padding,
        dilation=dilation,
        bias=bias,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(BasicBlock, self).__init__()

        self.stride = stride
        self.dilation = dilation
        self.downsample = downsample

        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        # self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual

        out = self.relu(out)
        return out

class Interpolate(nn.Module):
    def __init__(
            self,
            size=None,
            scale_factor=None,
            mode="bilinear",
            align_corners=None,
    ):
        super().__init__()

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(
            input, self.size, self.scale_factor, self.mode, self.align_corners
        )

class InHandConv(nn.Module):
    def __init__(self, patch_shape):
        super().__init__()
        self.in_hand_conv = nn.Sequential(OrderedDict([
            ('cnn_conv1', nn.Conv2d(patch_shape[0], 64, kernel_size=3)),
            ('cnn_relu1', nn.ReLU(inplace=True)),
            ('cnn_conv2', nn.Conv2d(64, 128, kernel_size=3)),
            ('cnn_relu2', nn.ReLU(inplace=True)),
            ('cnn_pool2', nn.MaxPool2d(2)),
            ('cnn_conv3', nn.Conv2d(128, 256, kernel_size=3)),
            ('cnn_relu3', nn.ReLU(inplace=True)),
        ]))

    def forward(self, in_hand):
        return self.in_hand_conv(in_hand)

class ResUBase:
    def __init__(self, n_input_channel=1):
        self.conv_down_1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        "enc-conv0",
                        nn.Conv2d(
                            n_input_channel,
                            32,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                        ),
                    ),
                    ("enc-relu0", nn.ReLU(inplace=True)),
                    (
                        'enc-res1',
                        BasicBlock(
                            32, 32,
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_down_2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-pool2',
                        nn.MaxPool2d(2)
                    ),
                    (
                        'enc-res2',
                        BasicBlock(
                            32, 64,
                            downsample=nn.Sequential(
                                nn.Conv2d(32, 64, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_down_4 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-pool3',
                        nn.MaxPool2d(2)
                    ),
                    (
                        'enc-res3',
                        BasicBlock(
                            64, 128,
                            downsample=nn.Sequential(
                                nn.Conv2d(64, 128, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_down_8 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-pool4',
                        nn.MaxPool2d(2)
                    ),
                    (
                        'enc-res4',
                        BasicBlock(
                            128, 256,
                            downsample=nn.Sequential(
                                nn.Conv2d(128, 256, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )
        self.conv_down_16 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-pool5',
                        nn.MaxPool2d(2)
                    ),
                    (
                        'enc-res5',
                        BasicBlock(
                            256, 512,
                            downsample=nn.Sequential(
                                nn.Conv2d(256, 512, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    ),
                    (
                        'enc-conv5',
                        nn.Conv2d(512, 256, kernel_size=1, bias=False)
                    )
                ]
            )
        )

        self.conv_up_8 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dec-res1',
                        BasicBlock(
                            512, 256,
                            downsample=nn.Sequential(
                                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    ),
                    (
                        'dec-conv1',
                        nn.Conv2d(256, 128, kernel_size=1, bias=False)
                    )
                ]
            )
        )
        self.conv_up_4 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dec-res2',
                        BasicBlock(
                            256, 128,
                            downsample=nn.Sequential(
                                nn.Conv2d(256, 128, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    ),
                    (
                        'dec-conv2',
                        nn.Conv2d(128, 64, kernel_size=1, bias=False)
                    )
                ]
            )
        )
        self.conv_up_2 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dec-res3',
                        BasicBlock(
                            128, 64,
                            downsample=nn.Sequential(
                                nn.Conv2d(128, 64, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    ),
                    (
                        'dec-conv3',
                        nn.Conv2d(64, 32, kernel_size=1, bias=False)
                    )
                ]
            )
        )
        self.conv_up_1 = nn.Sequential(
            OrderedDict(
                [
                    (
                        'dec-res1',
                        BasicBlock(
                            64, 32,
                            downsample=nn.Sequential(
                                nn.Conv2d(64, 32, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )

class ResUCat(nn.Module, ResUBase):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100)):
        super().__init__()
        ResUBase.__init__(self, n_input_channel)
        self.conv_cat_in_hand = nn.Sequential(
            OrderedDict(
                [
                    (
                        'enc-res6',
                        BasicBlock(
                            512, 256,
                            downsample=nn.Sequential(
                                nn.Conv2d(512, 256, kernel_size=1, bias=False),
                            ),
                            dilation=1
                        )
                    )
                ]
            )
        )

        self.in_hand_conv = InHandConv(patch_shape)

        self.q_values = nn.Conv2d(32, n_primitives, kernel_size=1, stride=1)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def forward(self, obs, in_hand):
        feature_map_1 = self.conv_down_1(obs)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)

        in_hand_out = self.in_hand_conv(in_hand)
        feature_map_16 = self.conv_cat_in_hand(torch.cat((feature_map_16, in_hand_out), dim=1))

        feature_map_up_8 = self.conv_up_8(torch.cat((feature_map_8,
                                                     F.interpolate(feature_map_16, size=feature_map_8.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_4 = self.conv_up_4(torch.cat((feature_map_4,
                                                     F.interpolate(feature_map_up_8, size=feature_map_4.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_2 = self.conv_up_2(torch.cat((feature_map_2,
                                                     F.interpolate(feature_map_up_4, size=feature_map_2.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_1 = self.conv_up_1(torch.cat((feature_map_1,
                                                     F.interpolate(feature_map_up_2, size=feature_map_1.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))

        q_values = self.q_values(feature_map_up_1)
        return q_values

class ResUCatShared(ResUCat):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100)):
        super().__init__(n_input_channel, n_primitives, patch_shape, domain_shape)

    def forward(self, obs, in_hand):
        feature_map_1 = self.conv_down_1(obs)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)

        in_hand_out = self.in_hand_conv(in_hand)
        feature_map_up_16 = self.conv_cat_in_hand(torch.cat((feature_map_16, F.interpolate(in_hand_out, size=feature_map_16.shape[-1], mode='bilinear', align_corners=False)), dim=1))

        feature_map_up_8 = self.conv_up_8(torch.cat((feature_map_8, F.interpolate(feature_map_up_16, size=feature_map_8.shape[-1], mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_4 = self.conv_up_4(torch.cat((feature_map_4, F.interpolate(feature_map_up_8, size=feature_map_4.shape[-1], mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_2 = self.conv_up_2(torch.cat((feature_map_2, F.interpolate(feature_map_up_4, size=feature_map_2.shape[-1], mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_1 = self.conv_up_1(torch.cat((feature_map_1, F.interpolate(feature_map_up_2, size=feature_map_1.shape[-1], mode='bilinear', align_corners=False)), dim=1))

        q_values = self.q_values(feature_map_up_1)
        return q_values, feature_map_up_16

class UCat(nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 100, 100)):
        super().__init__()
        self.conv_down_0 = torch.nn.Sequential(OrderedDict([
            ('enc-conv-0', nn.Conv2d(n_input_channel, 32, kernel_size=3, stride=1, padding=1)),
            ('enc-relu-0', nn.ReLU(inplace=True)),
        ]))
        self.conv_down_1 = torch.nn.Sequential(OrderedDict([
            ('enc-conv-1', nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            ('enc-relu-1', nn.ReLU(inplace=True)),
        ]))
        self.conv_down_2 = nn.Sequential(OrderedDict([
            ('enc-conv-2', nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)),
            ('enc-relu-2', nn.ReLU(inplace=True)),
            ('enc-conv-3', nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            ('enc-relu-3', nn.ReLU(inplace=True)),
        ]))
        self.conv_down_4 = nn.Sequential(OrderedDict([
            ('enc-conv-4', nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)),
            ('enc-relu-4', nn.ReLU(inplace=True)),
            ('enc-conv-5', nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
            ('enc-relu-5', nn.ReLU(inplace=True)),
        ]))
        self.conv_down_8 = nn.Sequential(OrderedDict([
            ('enc-conv-6', nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)),
            ('enc-relu-6', nn.ReLU(inplace=True)),
            ('enc-conv-7', nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            ('enc-relu-7', nn.ReLU(inplace=True)),
        ]))
        self.conv_down_16 = nn.Sequential(OrderedDict([
            ('enc-conv-8', nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)),
            ('enc-relu-8', nn.ReLU(inplace=True)),
            ('enc-conv-9', nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)),
            ('enc-relu-9', nn.ReLU(inplace=True)),
        ]))

        self.conv_up_8 = nn.Sequential(OrderedDict([
            ('dec-conv-0', nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)),
            ('enc-relu-0', nn.ReLU(inplace=True)),
            ('dec-conv-1', nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)),
            ('enc-relu-1', nn.ReLU(inplace=True)),
        ]))
        self.conv_up_4 = nn.Sequential(OrderedDict([
            ('dec-conv-2', nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)),
            ('enc-relu-2', nn.ReLU(inplace=True)),
            ('dec-conv-3', nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)),
            ('enc-relu-3', nn.ReLU(inplace=True)),
        ]))
        self.conv_up_2 = nn.Sequential(OrderedDict([
            ('dec-conv-4', nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)),
            ('enc-relu-4', nn.ReLU(inplace=True)),
            ('dec-conv-5', nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)),
            ('enc-relu-5', nn.ReLU(inplace=True)),
        ]))
        self.conv_up_1 = nn.Sequential(OrderedDict([
            ('dec-conv-6', nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)),
            ('enc-relu-6', nn.ReLU(inplace=True)),
            ('dec-conv-7', nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            ('enc-relu-7', nn.ReLU(inplace=True)),
        ]))

        self.in_hand_conv = torch.nn.Sequential(OrderedDict([
            ('cnn_conv1', torch.nn.Conv2d(1, 64, kernel_size=3)),
            ('cnn_relu1', torch.nn.ReLU(inplace=True)),
            ('cnn_conv2', torch.nn.Conv2d(64, 128, kernel_size=3)),
            ('cnn_relu2', torch.nn.ReLU(inplace=True)),
            ('cnn_pool2', torch.nn.MaxPool2d(2)),
            ('cnn_conv3', torch.nn.Conv2d(128, 256, kernel_size=3)),
            ('cnn_relu3', torch.nn.ReLU(inplace=True)),
        ]))

        self.conv_cat_in_hand = torch.nn.Sequential(OrderedDict([
            ('cat_conv', nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)),
            ('cat-relu', nn.ReLU(inplace=True)),
        ]))

        self.q_values = nn.Conv2d(32, n_primitives, kernel_size=3, stride=1, padding=1)

    def forward(self, obs, in_hand):
        feature_map_1 = self.conv_down_1(self.conv_down_0(obs))
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)

        in_hand_out = self.in_hand_conv(in_hand)
        feature_map_up_16 = self.conv_cat_in_hand(torch.cat((feature_map_16, in_hand_out), dim=1))

        feature_map_up_8 = self.conv_up_8(torch.cat((feature_map_8,
                                                     F.interpolate(feature_map_up_16, size=feature_map_8.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_4 = self.conv_up_4(torch.cat((feature_map_4,
                                                     F.interpolate(feature_map_up_8, size=feature_map_4.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_2 = self.conv_up_2(torch.cat((feature_map_2,
                                                     F.interpolate(feature_map_up_4, size=feature_map_2.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_1 = self.conv_up_1(torch.cat((feature_map_1,
                                                     F.interpolate(feature_map_up_2, size=feature_map_1.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))

        q_values = self.q_values(feature_map_up_1)
        return q_values, feature_map_up_16


class CNNShared(nn.Module):
    def __init__(self, image_shape, n_outputs):
        super().__init__()
        self.patch_conv = InHandConv(image_shape)
        conv_out_size = self._getConvOut(image_shape)
        self.fc1 = nn.Linear(conv_out_size, 1024)
        self.fc2 = nn.Linear(1024, n_outputs)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def _getConvOut(self, patch_shape):
        o1 = self.patch_conv(torch.zeros(1, *patch_shape))
        return int(np.prod(o1.size())) + 256*8*8

    def forward(self, obs_encoding, patch):
        # obs_encoding = obs_encoding.view(obs_encoding.size(0), -1)
        obs_encoding = obs_encoding.reshape(obs_encoding.size(0), -1)

        patch_conv_out = self.patch_conv(patch)
        patch_conv_out = patch_conv_out.view(patch.size(0), -1)

        x = torch.cat((obs_encoding, patch_conv_out), dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class CNNShared5l(nn.Module):
    def __init__(self, image_shape, n_outputs):
        super().__init__()
        self.patch_conv = torch.nn.Sequential(
            nn.Conv2d(image_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            torch.nn.Linear(128*(image_shape[1]//8)*(image_shape[1]//8), 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
        )
        self.es_fc = torch.nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            torch.nn.Linear(128*4*4, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(1024, n_outputs)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def _getConvOut(self, patch_shape):
        o1 = self.patch_conv(torch.zeros(1, *patch_shape))
        return int(np.prod(o1.size()))*2

    def forward(self, obs_encoding, patch):
        obs_encoding = self.es_fc(obs_encoding)
        patch_conv_out = self.patch_conv(patch)

        x = torch.cat((obs_encoding, patch_conv_out), dim=1)
        x = self.fc(x)
        return x

class CNNSepEnc(nn.Module):
    def __init__(self, image_shape, n_outputs):
        super().__init__()
        self.obs_conv = torch.nn.Sequential(OrderedDict([
            ('enc-conv-0', nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)),
            ('enc-relu-0', nn.ReLU(inplace=True)),
            ('enc-conv-1', nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
            ('enc-relu-1', nn.ReLU(inplace=True)),
            ('enc-conv-2', nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)),
            ('enc-relu-2', nn.ReLU(inplace=True)),
            ('enc-conv-3', nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)),
            ('enc-relu-3', nn.ReLU(inplace=True)),
            ('enc-conv-4', nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)),
            ('enc-relu-4', nn.ReLU(inplace=True)),
            ('enc-conv-5', nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)),
            ('enc-relu-5', nn.ReLU(inplace=True)),
        ]))

        self.patch_conv = InHandConv(image_shape)
        conv_out_size = self._getConvOut(image_shape)
        self.fc1 = nn.Linear(conv_out_size, 1024)
        self.fc2 = nn.Linear(1024, n_outputs)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def _getConvOut(self, patch_shape):
        o1 = self.patch_conv(torch.zeros(1, *patch_shape))
        return int(np.prod(o1.size())) * 2

    def forward(self, obs, patch):
        obs_encoding = self.obs_conv(obs)
        obs_encoding = obs_encoding.view(patch.size(0), -1)

        patch_conv_out = self.patch_conv(patch)
        patch_conv_out = patch_conv_out.view(patch.size(0), -1)

        x = torch.cat((obs_encoding, patch_conv_out), dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class CNNPatchOnly(nn.Module):
    def __init__(self, image_shape, n_outputs):
        super().__init__()
        self.patch_conv = InHandConv(image_shape)
        conv_out_size = self._getConvOut(image_shape)
        self.fc1 = nn.Linear(conv_out_size, 1024)
        self.fc2 = nn.Linear(1024, n_outputs)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def _getConvOut(self, patch_shape):
        o1 = self.patch_conv(torch.zeros(1, *patch_shape))
        return int(np.prod(o1.size()))

    def forward(self, obs, patch):
        patch_conv_out = self.patch_conv(patch)
        patch_conv_out = patch_conv_out.view(patch.size(0), -1)

        x = self.fc1(patch_conv_out)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class ResU(nn.Module):
    def __init__(self, n_input_channel=1, n_output_channel=3):
        super(ResU, self).__init__()
        self.conv_down_1 = nn.Sequential(OrderedDict([
            ('enc-conv-0', nn.Conv2d(n_input_channel, 32, kernel_size=3, stride=1, padding=1)),
            ('enc-relu-0', nn.ReLU(inplace=True)),
            ('enc-res1', BasicBlock(32, 32))
        ]))
        self.conv_down_2 = nn.Sequential(OrderedDict([
            ('enc-pool2', nn.MaxPool2d(2)),
            ('enc-res2', BasicBlock(32, 64, downsample=nn.Sequential(nn.Conv2d(32, 64, kernel_size=1, bias=False))))
        ]))
        self.conv_down_4 = nn.Sequential(OrderedDict([
            ('enc-pool3', nn.MaxPool2d(2)),
            ('enc-res3', BasicBlock(64, 128, downsample=nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False))))
        ]))
        self.conv_down_8 = nn.Sequential(OrderedDict([
            ('enc-pool4', nn.MaxPool2d(2)),
            ('enc-res4', BasicBlock(128, 256, downsample=nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False))))
        ]))
        self.conv_down_16 = nn.Sequential(OrderedDict([
            ('enc-pool5', nn.MaxPool2d(2)),
            ('enc-res5', BasicBlock(256, 512, downsample=nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, bias=False)))),
            ('enc-conv5', nn.Conv2d(512, 256, kernel_size=1, bias=False))
        ]))

        self.conv_up_8 = nn.Sequential(OrderedDict([
            ('dec-res1', BasicBlock(512, 256, downsample=nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, bias=False)))),
            ('dec-conv1', nn.Conv2d(256, 128, kernel_size=1, bias=False))
        ]))
        self.conv_up_4 = nn.Sequential(OrderedDict([
            ('dec-res2', BasicBlock(256, 128, downsample=nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, bias=False)))),
            ('dec-conv2', nn.Conv2d(128, 64, kernel_size=1, bias=False))
        ]))
        self.conv_up_2 = nn.Sequential(OrderedDict([
            ('dec-res3', BasicBlock(128, 64, downsample=nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False)))),
            ('dec-conv3', nn.Conv2d(64, 32, kernel_size=1, bias=False))
        ]))
        self.conv_up_1 = nn.Sequential(OrderedDict([
            ('dec-res4', BasicBlock(64, 32, downsample=nn.Sequential(nn.Conv2d(64, 32, kernel_size=1, bias=False)))),
            ('dec-conv4', nn.Conv2d(32, n_output_channel, kernel_size=1))
        ]))

    def forward(self, x):
        feature_map_1 = self.conv_down_1(x)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)

        feature_map_up_8 = self.conv_up_8(torch.cat((feature_map_8,
                                                     F.interpolate(feature_map_16, size=feature_map_8.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_4 = self.conv_up_4(torch.cat((feature_map_4,
                                                     F.interpolate(feature_map_up_8, size=feature_map_4.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_2 = self.conv_up_2(torch.cat((feature_map_2,
                                                     F.interpolate(feature_map_up_4, size=feature_map_2.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        feature_map_up_1 = self.conv_up_1(torch.cat((feature_map_1,
                                                     F.interpolate(feature_map_up_2, size=feature_map_1.shape[-1],
                                                                   mode='bilinear', align_corners=False)), dim=1))
        return feature_map_up_1

class ResUTransport(nn.Module):
    def __init__(self, n_input_channel, n_rot=8, half_rotation=True):
        super(ResUTransport, self).__init__()
        self.feature = ResU(n_input_channel, n_output_channel=3)
        self.filter = ResU(n_input_channel, n_output_channel=3)

        self.n_rot = n_rot
        if half_rotation:
            max_rot = np.pi
        else:
            max_rot = 2*np.pi
        self.rzs = torch.from_numpy(np.linspace(0, max_rot, self.n_rot, endpoint=False)).float()

        affine_mats = []
        for rotate_theta in self.rzs:
            rotate_theta = -rotate_theta
            affine_mat = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],
                                     [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat.shape = (2, 3, 1)
            affine_mat = torch.from_numpy(affine_mat).permute(2, 0, 1).float()
            affine_mats.append(affine_mat)
        self.affine_mats = torch.cat(affine_mats)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def forward(self, obs, patch):
        batch_size = obs.shape[0]
        patch_size = patch.shape[-1]
        df_pad = (patch_size//2-1, patch_size//2, patch_size//2-1, patch_size//2)

        patch = patch.unsqueeze(1).repeat(1, self.n_rot, 1, 1, 1)
        patch = patch.reshape(patch.size(0) * patch.size(1), patch.size(2), patch.size(3), patch.size(4))
        affine_mats = self.affine_mats.repeat(batch_size, 1, 1).to(obs.device)
        flow_grid_before = F.affine_grid(affine_mats, patch.size(), align_corners=False)
        patch = F.grid_sample(patch, flow_grid_before, mode='nearest', align_corners=False)

        feature = self.feature(obs)
        weight = self.filter(patch)
        df_channel = weight.shape[1]
        df_window = weight.shape[-1]
        weight = weight.reshape(batch_size*self.n_rot, df_channel, df_window, df_window)
        feature = feature.reshape(1, feature.size(0)*feature.size(1), feature.size(2), feature.size(3))
        feature = F.conv2d(F.pad(feature, df_pad), weight=weight, groups=batch_size)
        feature = feature.reshape(batch_size, -1, feature.size(2), feature.size(3))
        return feature

class ResUTransportRegress(nn.Module):
    def __init__(self, n_input_channel, n_rot=8, half_rotation=True):
        super(ResUTransportRegress, self).__init__()
        self.feature = ResU(n_input_channel, n_output_channel=8)
        self.filter = ResU(n_input_channel, n_output_channel=8)
        self.out = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

        self.n_rot = n_rot
        if half_rotation:
            max_rot = np.pi
        else:
            max_rot = 2*np.pi
        self.rzs = torch.from_numpy(np.linspace(0, max_rot, self.n_rot, endpoint=False)).float()

        affine_mats = []
        for rotate_theta in self.rzs:
            rotate_theta = -rotate_theta
            affine_mat = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],
                                     [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat.shape = (2, 3, 1)
            affine_mat = torch.from_numpy(affine_mat).permute(2, 0, 1).float()
            affine_mats.append(affine_mat)
        self.affine_mats = torch.cat(affine_mats)

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def forward(self, obs, patch):
        batch_size = obs.shape[0]

        patch = patch.unsqueeze(1).repeat(1, self.n_rot, 1, 1, 1)
        patch = patch.reshape(patch.size(0) * patch.size(1), patch.size(2), patch.size(3), patch.size(4))
        affine_mats = self.affine_mats.repeat(batch_size, 1, 1).to(obs.device)
        flow_grid_before = F.affine_grid(affine_mats, patch.size(), align_corners=False)
        patch = F.grid_sample(patch, flow_grid_before, mode='bilinear', align_corners=False)

        feature = self.feature(obs)
        weight = self.filter(patch)
        df_channel = weight.shape[1]
        df_window = weight.shape[-1]
        feature_size = feature.shape[-1]
        # permute to BxDFxR to group by BxDF
        weight = weight.reshape(batch_size, self.n_rot, df_channel, df_window, df_window)
        weight = weight.permute(0, 2, 1, 3, 4)
        weight = weight.reshape(batch_size*df_channel*self.n_rot, 1, df_window, df_window)

        feature = feature.reshape(1, feature.size(0)*feature.size(1), feature_size, feature_size)
        feature = F.conv2d(F.pad(feature, (11, 12, 11, 12)), weight=weight, groups=batch_size*df_channel)

        # permute back
        feature = feature.reshape(batch_size, df_channel, self.n_rot, feature_size, feature_size)
        feature = feature.permute(0, 2, 1, 3, 4)

        feature = feature.reshape(batch_size*self.n_rot, df_channel, feature_size, feature_size)
        feature = F.relu(feature)
        feature = self.out(feature)
        feature = feature.reshape(batch_size, self.n_rot, feature_size, feature_size)
        return feature

class ResURot(nn.Module):
    def __init__(self, n_input_channel, n_rot=8, half_rotation=True):
        super(ResURot, self).__init__()
        self.feature = ResU(n_input_channel, n_output_channel=1)
        self.n_rot = n_rot
        if half_rotation:
            max_rot = np.pi
        else:
            max_rot = 2*np.pi
        self.rzs = torch.from_numpy(np.linspace(0, max_rot, self.n_rot, endpoint=False)).float()

        for m in self.named_modules():
            if isinstance(m[1], nn.Conv2d):
                # nn.init.kaiming_normal_(m[1].weight.data)
                nn.init.xavier_normal_(m[1].weight.data)
            elif isinstance(m[1], nn.BatchNorm2d):
                m[1].weight.data.fill_(1)
                m[1].bias.data.zero_()

    def getAffineMatrices(self, n):
        rotations = [self.rzs for _ in range(n)]
        affine_mats_before = []
        affine_mats_after = []
        for i in range(n):
            for rotate_theta in rotations[i]:
                affine_mat_before = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],
                                                [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                affine_mat_before.shape = (2, 3, 1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
                affine_mats_before.append(affine_mat_before)

                affine_mat_after = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
                                               [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                affine_mat_after.shape = (2, 3, 1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                affine_mats_after.append(affine_mat_after)

        affine_mats_before = torch.cat(affine_mats_before)
        affine_mats_after = torch.cat(affine_mats_after)
        return affine_mats_before, affine_mats_after


    def forward(self, obs):
        batch_size = obs.shape[0]
        diag_length = float(obs.size(2)) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 32) * 32
        padding_width = int((diag_length - obs.size(2)) / 2)

        affine_mats_before, affine_mats_after = self.getAffineMatrices(batch_size)
        affine_mats_before = affine_mats_before.to(obs.device)
        affine_mats_after = affine_mats_after.to(obs.device)
        # pad obs
        obs = F.pad(obs, (padding_width, padding_width, padding_width, padding_width), mode='constant', value=0)
        # expand obs into shape (n*num_rot, c, h, w)
        obs = obs.unsqueeze(1).repeat(1, self.n_rot, 1, 1, 1)
        obs = obs.reshape(obs.size(0) * obs.size(1), obs.size(2), obs.size(3), obs.size(4))
        # rotate obs
        flow_grid_before = F.affine_grid(affine_mats_before, obs.size(), align_corners=False)
        rotated_obs = F.grid_sample(obs, flow_grid_before, mode='nearest', align_corners=False)
        # forward network
        conv_output = self.feature(rotated_obs)
        # rotate output
        flow_grid_after = F.affine_grid(affine_mats_after, conv_output.size(), align_corners=False)
        unrotate_output = F.grid_sample(conv_output, flow_grid_after, mode='nearest', align_corners=False)

        rotation_output = unrotate_output.reshape(
            (batch_size, -1, unrotate_output.size(2), unrotate_output.size(3)))
        predictions = rotation_output[:, :, padding_width: -padding_width, padding_width: -padding_width]
        return predictions

class DynamicFilter(nn.Module):
    def __init__(self, in_shape=(1, 24, 24), out_shape=(3, 24, 24)):
        super().__init__()
        self.in_channel = in_shape[0]
        self.out_channel = out_shape[0]
        self.out_size = out_shape[1]
        self.enc = nn.Sequential(OrderedDict([
            ("enc-conv0",  nn.Conv2d(self.in_channel, 32, kernel_size=3, stride=1, padding=1)),
            ("enc-relu0", nn.ReLU(inplace=True)),
            ('enc-res1', BasicBlock(32, 32, dilation=1)),
            ('enc-pool2', nn.MaxPool2d(2)),
            ('enc-res2', BasicBlock(32, 64, downsample=nn.Sequential(nn.Conv2d(32, 64, kernel_size=1, bias=False)))),
            ('enc-pool3', nn.MaxPool2d(2)),
            ('enc-res3', BasicBlock(64, 128, downsample=nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False)))),
            ('enc-pool4', nn.MaxPool2d(2)),
            ('enc-res4', BasicBlock(128, 256, downsample=nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False)))),

            ('upsample6', Interpolate(size=(out_shape[1], out_shape[2]), mode='bilinear', align_corners=False)),
            ("enc-conv6", nn.Conv2d(256, self.out_channel, kernel_size=1, bias=False)),
        ]))

    def forward(self, x):
        return self.enc(x)

class DynamicFilterFC(nn.Module):
    def __init__(self, in_shape=(1, 24, 24), out_n=256):
        super().__init__()
        self.in_channel = in_shape[0]
        self.enc = nn.Sequential(OrderedDict([
            ("enc-conv0",  nn.Conv2d(self.in_channel, 32, kernel_size=3, stride=1, padding=1)),
            ("enc-relu0", nn.ReLU(inplace=True)),
            ('enc-res1', BasicBlock(32, 32, dilation=1)),
            ('enc-pool2', nn.MaxPool2d(2)),
            ('enc-res2', BasicBlock(32, 64, downsample=nn.Sequential(nn.Conv2d(32, 64, kernel_size=1, bias=False)))),
            ('enc-pool3', nn.MaxPool2d(2)),
            ('enc-res3', BasicBlock(64, 128, downsample=nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False)))),
            ('enc-pool4', nn.MaxPool2d(2)),
            ('enc-res4', BasicBlock(128, 256, downsample=nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False)))),
        ]))
        self.fc1 = nn.Linear(int((in_shape[1]/8)*(in_shape[2]/8)*256), 1024)
        self.fc2 = nn.Linear(1024, out_n)

    def forward(self, x):
        x = self.enc(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DynamicFilterFC1(nn.Module):
    def __init__(self, in_shape=(1, 24, 24), out_n=256):
        super().__init__()
        self.in_channel = in_shape[0]
        self.enc = nn.Sequential(OrderedDict([
            ("enc-conv0",  nn.Conv2d(self.in_channel, 32, kernel_size=3, stride=1, padding=1)),
            ("enc-relu0", nn.ReLU(inplace=True)),
            ('enc-res1', BasicBlock(32, 32, dilation=1)),
            ('enc-pool2', nn.MaxPool2d(2)),
            ('enc-res2', BasicBlock(32, 64, downsample=nn.Sequential(nn.Conv2d(32, 64, kernel_size=1, bias=False)))),
            ('enc-pool3', nn.MaxPool2d(2)),
            ('enc-res3', BasicBlock(64, 128, downsample=nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False)))),
            ('enc-res4', BasicBlock(128, 256, downsample=nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False)))),
        ]))
        self.fc1 = nn.Linear(int((in_shape[1]/4)*(in_shape[2]/4)*256), 1024)
        self.fc2 = nn.Linear(1024, out_n)

    def forward(self, x):
        x = self.enc(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DynamicFilterFC2(nn.Module):
    def __init__(self, in_shape=(1, 24, 24), out_n=256):
        super().__init__()
        self.in_channel = in_shape[0]
        self.enc = nn.Sequential(OrderedDict([
            ("enc-conv0",  nn.Conv2d(self.in_channel, 32, kernel_size=3, stride=1, padding=1)),
            ("enc-relu0", nn.ReLU(inplace=True)),

            ('enc-conv1', nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            ("enc-relu1", nn.ReLU(inplace=True)),

            ('enc-pool2', nn.MaxPool2d(2)),
            ('enc-conv2', nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            ("enc-relu2", nn.ReLU(inplace=True)),

            ('enc-pool3', nn.MaxPool2d(2)),
            ('enc-conv3', nn.Conv2d(128, 256, kernel_size=3, padding=1)),
            ("enc-relu3", nn.ReLU(inplace=True)),
        ]))
        self.fc1 = nn.Linear(int((in_shape[1]/4)*(in_shape[2]/4)*256), 1024)
        self.fc2 = nn.Linear(1024, out_n)

    def forward(self, x):
        x = self.enc(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DynamicFilterFC3(nn.Module):
    def __init__(self, in_shape=(1, 24, 24), out_n=256):
        super().__init__()
        self.in_channel = in_shape[0]
        self.enc = nn.Sequential(OrderedDict([
            ("enc-conv0",  nn.Conv2d(self.in_channel, 32, kernel_size=3, stride=1, padding=1)),
            ("enc-relu0", nn.ReLU(inplace=True)),

            ('enc-pool1', nn.MaxPool2d(2)),
            ('enc-conv1', nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            ("enc-relu1", nn.ReLU(inplace=True)),

            ('enc-pool2', nn.MaxPool2d(2)),
            ('enc-conv2', nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            ("enc-relu2", nn.ReLU(inplace=True)),

            ('enc-pool3', nn.MaxPool2d(2)),
            ('enc-conv3', nn.Conv2d(128, 256, kernel_size=3, padding=1)),
            ("enc-relu3", nn.ReLU(inplace=True)),
        ]))
        self.fc1 = nn.Linear(int((in_shape[1]/8)*(in_shape[2]/8)*256), 1024)
        self.fc2 = nn.Linear(1024, out_n)

    def forward(self, x):
        x = self.enc(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    unet = ResUTransport(1)
    inp = torch.zeros((1, 1, 128, 128))
    inh = torch.zeros((1, 1, 24, 24))
    inh[:, :, 10:20, 10:20] = 1
    out = unet(inp, inh)