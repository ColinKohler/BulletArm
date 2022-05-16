import os
import torch
import torch.nn.functional as F
import dill
import numpy as np
from e2cnn import gspaces
from e2cnn import nn
from collections import OrderedDict
from bulletarm_baselines.fc_dqn.networks.models import DynamicFilter, ResU, DynamicFilterFC, DynamicFilterFC1, DynamicFilterFC2, DynamicFilterFC3
from bulletarm_baselines.fc_dqn.networks.r2conv_df import R2ConvDF

class EquiResBlock(torch.nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size, N, flip=False, quotient=False, initialize=True):
        super(EquiResBlock, self).__init__()

        if flip:
            r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            r2_act = gspaces.Rot2dOnR2(N=N)

        if quotient:
            if flip:
                rep = r2_act.quotient_repr((None, 2))
            else:
                rep = r2_act.quotient_repr(2)
        else:
            rep = r2_act.regular_repr

        feat_type_in = nn.FieldType(r2_act, input_channels * [rep])
        feat_type_hid = nn.FieldType(r2_act, hidden_dim * [rep])

        self.layer1 = nn.SequentialModule(
            nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, initialize=initialize),
            nn.ReLU(feat_type_hid, inplace=True)
        )

        self.layer2 = nn.SequentialModule(
            nn.R2Conv(feat_type_hid, feat_type_hid, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, initialize=initialize),

        )
        self.relu = nn.ReLU(feat_type_hid, inplace=True)

        self.upscale = None
        if input_channels != hidden_dim:
            self.upscale = nn.SequentialModule(
                nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, initialize=initialize),
            )

    def forward(self, xx):
        residual = xx
        out = self.layer1(xx)
        out = self.layer2(out)
        if self.upscale:
            out += self.upscale(residual)
        else:
            out += residual
        out = self.relu(out)

        return out

class EquiResBlockMix(torch.nn.Module):
    def __init__(self, input_t, hidden_t, kernel_size, initialize=True):
        super(EquiResBlockMix, self).__init__()

        self.layer1 = nn.SequentialModule(
            nn.R2Conv(input_t, hidden_t, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, initialize=initialize),
            nn.ReLU(hidden_t, inplace=True)
        )

        self.layer2 = nn.SequentialModule(
            nn.R2Conv(hidden_t, hidden_t, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, initialize=initialize),

        )
        self.relu = nn.ReLU(hidden_t, inplace=True)

        self.upscale = None
        if input_t != hidden_t:
            self.upscale = nn.SequentialModule(
                nn.R2Conv(input_t, hidden_t, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, initialize=initialize),
            )

    def forward(self, xx):
        residual = xx
        out = self.layer1(xx)
        out = self.layer2(out)
        if self.upscale:
            out += self.upscale(residual)
        else:
            out += residual
        out = self.relu(out)

        return out

class conv2d(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, N, activation=True, last=False, flip=False, quotient=False, initialize=True):
        super(conv2d, self).__init__()
        if flip:
            r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            r2_act = gspaces.Rot2dOnR2(N=N)

        if quotient:
            if flip:
                rep = r2_act.quotient_repr((None, 2))
            else:
                rep = r2_act.quotient_repr(2)
        else:
            rep = r2_act.regular_repr

        feat_type_in = nn.FieldType(r2_act, input_channels * [rep])
        if last:
            feat_type_hid = nn.FieldType(r2_act, output_channels * [r2_act.trivial_repr])
        else:
            feat_type_hid = nn.FieldType(r2_act, output_channels * [rep])

        if activation:
            self.layer = nn.SequentialModule(
                nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, initialize=initialize),
                nn.ReLU(feat_type_hid, inplace=True)
            )
        else:
            self.layer = nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, initialize=initialize)

    def forward(self, xx):
        return self.layer(xx)

class EquCNNEnc(torch.nn.Module):
    def __init__(self, input_channel, output_channel, N, kernel_size=3, out_size=8, quotient=False, initialize=True):
        assert out_size in [8, 10]
        super().__init__()
        self.input_channel = input_channel
        self.N = N
        self.r2_act = gspaces.Rot2dOnR2(N=N)
        if quotient:
            rep = self.r2_act.quotient_repr(2)
        else:
            rep = self.r2_act.regular_repr

        n1 = int(output_channel/4)
        n2 = int(output_channel/2)

        if out_size == 8:
            last_padding = kernel_size//2-1
        else:
            last_padding = kernel_size // 2

        self.conv = torch.nn.Sequential(OrderedDict([
            ('e2conv-1', nn.R2Conv(nn.FieldType(self.r2_act, input_channel * [self.r2_act.trivial_repr]),
                                   nn.FieldType(self.r2_act, n1 * [rep]),
                                   kernel_size=kernel_size, padding=kernel_size//2-1, initialize=initialize)),
            ('e2relu-1', nn.ReLU(nn.FieldType(self.r2_act, n1 * [rep]), inplace=True)),
            ('e2conv-2', nn.R2Conv(nn.FieldType(self.r2_act, n1 * [rep]),
                                   nn.FieldType(self.r2_act, n2 * [rep]),
                                   kernel_size=kernel_size, padding=kernel_size//2-1, initialize=initialize)),
            ('e2relu-2', nn.ReLU(nn.FieldType(self.r2_act, n2 * [rep]), inplace=True)),
            ('e2maxpool-2', nn.PointwiseMaxPool(nn.FieldType(self.r2_act, n2 * [rep]), 2)),
            ('e2conv-3', nn.R2Conv(nn.FieldType(self.r2_act, n2 * [rep]),
                                   nn.FieldType(self.r2_act, output_channel * [rep]),
                                   kernel_size=kernel_size, padding=last_padding, initialize=initialize)),
            ('e2relu-3', nn.ReLU(nn.FieldType(self.r2_act, output_channel * [rep]), inplace=True)),
        ]))

    def forward(self, x):
        x = nn.GeometricTensor(x, nn.FieldType(self.r2_act, self.input_channel * [self.r2_act.trivial_repr]))
        return self.conv(x)

class EquCNNEncP40(torch.nn.Module):
    def __init__(self, input_channel, output_channel, N, kernel_size=3, quotient=False, initialize=True):
        super().__init__()
        self.input_channel = input_channel
        self.N = N
        self.r2_act = gspaces.Rot2dOnR2(N=N)
        if quotient:
            rep = self.r2_act.quotient_repr(2)
        else:
            rep = self.r2_act.regular_repr

        n1 = int(output_channel/4)
        n2 = int(output_channel/2)

        self.conv = torch.nn.Sequential(OrderedDict([
            ('e2conv-1', nn.R2Conv(nn.FieldType(self.r2_act, input_channel * [self.r2_act.trivial_repr]),
                                   nn.FieldType(self.r2_act, n1 * [rep]),
                                   kernel_size=kernel_size, padding=kernel_size//2, initialize=initialize)),
            ('e2relu-1', nn.ReLU(nn.FieldType(self.r2_act, n1 * [rep]), inplace=True)),
            ('e2maxpool-1', nn.PointwiseMaxPool(nn.FieldType(self.r2_act, n1 * [rep]), 2)),
            ('e2conv-2', nn.R2Conv(nn.FieldType(self.r2_act, n1 * [rep]),
                                   nn.FieldType(self.r2_act, n2 * [rep]),
                                   kernel_size=kernel_size, padding=kernel_size//2, initialize=initialize)),
            ('e2relu-2', nn.ReLU(nn.FieldType(self.r2_act, n2 * [rep]), inplace=True)),
            ('e2maxpool-2', nn.PointwiseMaxPool(nn.FieldType(self.r2_act, n2 * [rep]), 2)),
            ('e2conv-3', nn.R2Conv(nn.FieldType(self.r2_act, n2 * [rep]),
                                   nn.FieldType(self.r2_act, output_channel * [rep]),
                                   kernel_size=kernel_size, padding=kernel_size//2-1, initialize=initialize)),
            ('e2relu-3', nn.ReLU(nn.FieldType(self.r2_act, output_channel * [rep]), inplace=True)),
        ]))

    def forward(self, x):
        x = nn.GeometricTensor(x, nn.FieldType(self.r2_act, self.input_channel * [self.r2_act.trivial_repr]))
        return self.conv(x)

class EquResUNet(torch.nn.Module):
    def __init__(self, n_input_channel=1, n_output_channel=16, n_middle_channels=(16, 32, 64, 128), kernel_size=3, N=8, flip=False, quotient=False, initialize=True):
        super().__init__()
        self.N = N
        self.quotient = quotient
        if flip:
            self.r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            self.r2_act = gspaces.Rot2dOnR2(N=N)

        if quotient:
            if flip:
                self.repr = self.r2_act.quotient_repr((None, 2))
            else:
                self.repr = self.r2_act.quotient_repr(2)
        else:
            self.repr = self.r2_act.regular_repr

        assert len(n_middle_channels) == 4
        self.l1_c = n_middle_channels[0]
        self.l2_c = n_middle_channels[1]
        self.l3_c = n_middle_channels[2]
        self.l4_c = n_middle_channels[3]


        self.conv_down_1 = torch.nn.Sequential(OrderedDict([
            ('enc-e2conv-0', nn.R2Conv(nn.FieldType(self.r2_act, n_input_channel * [self.r2_act.trivial_repr]),
                                       nn.FieldType(self.r2_act, self.l1_c * [self.repr]),
                                       kernel_size=3, padding=1, initialize=initialize)),
            ('enc-e2relu-0', nn.ReLU(nn.FieldType(self.r2_act, self.l1_c * [self.repr]), inplace=True)),
            ('enc-e2res-1', EquiResBlock(self.l1_c, self.l1_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)),
        ]))

        self.conv_down_2 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-2', nn.PointwiseMaxPool(nn.FieldType(self.r2_act, self.l1_c * [self.repr]), 2)),
            ('enc-e2res-2', EquiResBlock(self.l1_c, self.l2_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)),
        ]))
        self.conv_down_4 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-3', nn.PointwiseMaxPool(nn.FieldType(self.r2_act, self.l2_c * [self.repr]), 2)),
            ('enc-e2res-3', EquiResBlock(self.l2_c, self.l3_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)),
        ]))
        self.conv_down_8 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-4', nn.PointwiseMaxPool(nn.FieldType(self.r2_act, self.l3_c * [self.repr]), 2)),
            ('enc-e2res-4', EquiResBlock(self.l3_c, self.l4_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)),
        ]))
        self.conv_down_16 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-5', nn.PointwiseMaxPool(nn.FieldType(self.r2_act, self.l4_c * [self.repr]), 2)),
            ('enc-e2res-5', EquiResBlock(self.l4_c, self.l4_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)),
        ]))

        self.conv_up_8 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-1', EquiResBlock(2*self.l4_c, self.l3_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)),
        ]))
        self.conv_up_4 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-2', EquiResBlock(2*self.l3_c, self.l2_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)),
        ]))
        self.conv_up_2 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-3', EquiResBlock(2*self.l2_c, self.l1_c, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)),
        ]))
        self.conv_up_1 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-4', EquiResBlock(2*self.l1_c, n_output_channel, kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)),
        ]))

        self.upsample_16_8 = nn.R2Upsampling(nn.FieldType(self.r2_act, self.l4_c * [self.repr]), 2)
        self.upsample_8_4 = nn.R2Upsampling(nn.FieldType(self.r2_act, self.l3_c * [self.repr]), 2)
        self.upsample_4_2 = nn.R2Upsampling(nn.FieldType(self.r2_act, self.l2_c * [self.repr]), 2)
        self.upsample_2_1 = nn.R2Upsampling(nn.FieldType(self.r2_act, self.l1_c * [self.repr]), 2)

    def forwardEncoder(self, obs):
        obs_gt = nn.GeometricTensor(obs, nn.FieldType(self.r2_act, obs.shape[1] * [self.r2_act.trivial_repr]))
        feature_map_1 = self.conv_down_1(obs_gt)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)
        return feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16

    def forwardDecoder(self, feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16):
        concat_8 = torch.cat((feature_map_8.tensor, self.upsample_16_8(feature_map_16).tensor), dim=1)
        concat_8 = nn.GeometricTensor(concat_8, nn.FieldType(self.r2_act, 2*self.l4_c * [self.repr]))
        feature_map_up_8 = self.conv_up_8(concat_8)

        concat_4 = torch.cat((feature_map_4.tensor, self.upsample_8_4(feature_map_up_8).tensor), dim=1)
        concat_4 = nn.GeometricTensor(concat_4, nn.FieldType(self.r2_act, 2*self.l3_c * [self.repr]))
        feature_map_up_4 = self.conv_up_4(concat_4)

        concat_2 = torch.cat((feature_map_2.tensor, self.upsample_4_2(feature_map_up_4).tensor), dim=1)
        concat_2 = nn.GeometricTensor(concat_2, nn.FieldType(self.r2_act, 2*self.l2_c * [self.repr]))
        feature_map_up_2 = self.conv_up_2(concat_2)

        concat_1 = torch.cat((feature_map_1.tensor, self.upsample_2_1(feature_map_up_2).tensor), dim=1)
        concat_1 = nn.GeometricTensor(concat_1, nn.FieldType(self.r2_act, 2*self.l1_c * [self.repr]))
        feature_map_up_1 = self.conv_up_1(concat_1)

        return feature_map_up_1

    def forward(self, obs):
        feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16 = self.forwardEncoder(obs)
        return self.forwardDecoder(feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16)


class EquResUNetMix(torch.nn.Module):
    def __init__(self, n_input_channel=1, n_output_channel=16, n_middle_channels=(16, 32, 64, 128), kernel_size=3, N=8,
                 flip=False, quotient=False, out_reflect=False, mix_type=0, initialize=True):
        super().__init__()
        self.N = N
        self.quotient = quotient
        if flip:
            self.r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            self.r2_act = gspaces.Rot2dOnR2(N=N)

        if flip and quotient and out_reflect:
            self.repr = self.r2_act.quotient_repr((None, 2))
        elif flip and quotient and not out_reflect:
            self.repr = self.r2_act.quotient_repr((0, 2))
        elif flip and not quotient and not out_reflect:
            self.repr = self.r2_act.quotient_repr((0, 1))
        elif not flip and quotient:
            self.repr = self.r2_act.quotient_repr(2)
        else:
            self.repr = self.r2_act.regular_repr

        assert len(n_middle_channels) == 4

        self.mix_type = mix_type
        if mix_type == 0:
            self.l1_c = n_middle_channels[0] * [self.repr]
            self.l2_c = n_middle_channels[1] * [self.repr]
            self.l3_c = n_middle_channels[2] * [self.repr]
            self.l4_c = n_middle_channels[3] * [self.repr]
        elif mix_type == 1:
            assert flip and not quotient
            self.l1_c = n_middle_channels[0]//2*[self.r2_act.regular_repr] + n_middle_channels[0]//2*[self.r2_act.quotient_repr((0, 1))]
            self.l2_c = n_middle_channels[1]//2*[self.r2_act.regular_repr] + n_middle_channels[1]//2*[self.r2_act.quotient_repr((0, 1))]
            self.l3_c = n_middle_channels[2]//2*[self.r2_act.regular_repr] + n_middle_channels[2]//2*[self.r2_act.quotient_repr((0, 1))]
            self.l4_c = n_middle_channels[3]//2*[self.r2_act.regular_repr] + n_middle_channels[3]//2*[self.r2_act.quotient_repr((0, 1))]
        elif mix_type == 2:
            assert flip and not quotient
            self.l1_c = 1*n_middle_channels[0]//4*[self.r2_act.regular_repr] + 3*n_middle_channels[0]//4*[self.r2_act.quotient_repr((0, 1))]
            self.l2_c = 1*n_middle_channels[1]//4*[self.r2_act.regular_repr] + 3*n_middle_channels[1]//4*[self.r2_act.quotient_repr((0, 1))]
            self.l3_c = 1*n_middle_channels[2]//4*[self.r2_act.regular_repr] + 3*n_middle_channels[2]//4*[self.r2_act.quotient_repr((0, 1))]
            self.l4_c = 1*n_middle_channels[3]//4*[self.r2_act.regular_repr] + 3*n_middle_channels[3]//4*[self.r2_act.quotient_repr((0, 1))]
        elif mix_type == 3:
            assert flip and not quotient
            self.l1_c = 1*n_middle_channels[0]//8*[self.r2_act.regular_repr] + 7*n_middle_channels[0]//8*[self.r2_act.quotient_repr((0, 1))]
            self.l2_c = 1*n_middle_channels[1]//8*[self.r2_act.regular_repr] + 7*n_middle_channels[1]//8*[self.r2_act.quotient_repr((0, 1))]
            self.l3_c = 1*n_middle_channels[2]//8*[self.r2_act.regular_repr] + 7*n_middle_channels[2]//8*[self.r2_act.quotient_repr((0, 1))]
            self.l4_c = 1*n_middle_channels[3]//8*[self.r2_act.regular_repr] + 7*n_middle_channels[3]//8*[self.r2_act.quotient_repr((0, 1))]

        else:
            raise NotImplementedError

        self.l1_t = nn.FieldType(self.r2_act, self.l1_c)
        self.l2_t = nn.FieldType(self.r2_act, self.l2_c)
        self.l3_t = nn.FieldType(self.r2_act, self.l3_c)
        self.l4_t = nn.FieldType(self.r2_act, self.l4_c)

        self.db_l1_t = nn.FieldType(self.r2_act, 2*self.l1_c)
        self.db_l2_t = nn.FieldType(self.r2_act, 2*self.l2_c)
        self.db_l3_t = nn.FieldType(self.r2_act, 2*self.l3_c)
        self.db_l4_t = nn.FieldType(self.r2_act, 2*self.l4_c)

        self.out_t = nn.FieldType(self.r2_act, n_output_channel * [self.repr])

        self.conv_down_1 = torch.nn.Sequential(OrderedDict([
            ('enc-e2conv-0', nn.R2Conv(nn.FieldType(self.r2_act, n_input_channel * [self.r2_act.trivial_repr]),
                                       self.l1_t, kernel_size=3, padding=1, initialize=initialize)),
            ('enc-e2relu-0', nn.ReLU(self.l1_t, inplace=True)),
            ('enc-e2res-1', EquiResBlockMix(self.l1_t, self.l1_t, kernel_size=kernel_size, initialize=initialize)),
        ]))

        self.conv_down_2 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-2', nn.PointwiseMaxPool(self.l1_t, 2)),
            ('enc-e2res-2',EquiResBlockMix(self.l1_t, self.l2_t, kernel_size=kernel_size, initialize=initialize)),
        ]))
        self.conv_down_4 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-3', nn.PointwiseMaxPool(self.l2_t, 2)),
            ('enc-e2res-3',EquiResBlockMix(self.l2_t, self.l3_t, kernel_size=kernel_size, initialize=initialize)),
        ]))
        self.conv_down_8 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-4', nn.PointwiseMaxPool(self.l3_t, 2)),
            ('enc-e2res-4',EquiResBlockMix(self.l3_t, self.l4_t, kernel_size=kernel_size, initialize=initialize)),
        ]))
        self.conv_down_16 = torch.nn.Sequential(OrderedDict([
            ('enc-pool-5', nn.PointwiseMaxPool(self.l4_t, 2)),
            ('enc-e2res-5',EquiResBlockMix(self.l4_t, self.l4_t, kernel_size=kernel_size, initialize=initialize)),
        ]))

        self.conv_up_8 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-1', EquiResBlockMix(self.db_l4_t, self.l3_t, kernel_size=kernel_size, initialize=initialize)),
        ]))
        self.conv_up_4 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-2', EquiResBlockMix(self.db_l3_t, self.l2_t, kernel_size=kernel_size, initialize=initialize)),
        ]))
        self.conv_up_2 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-3', EquiResBlockMix(self.db_l2_t, self.l1_t, kernel_size=kernel_size, initialize=initialize)),
        ]))
        self.conv_up_1 = torch.nn.Sequential(OrderedDict([
            ('dec-e2res-4', EquiResBlockMix(self.db_l1_t, self.out_t, kernel_size=kernel_size, initialize=initialize)),
        ]))

        self.upsample_16_8 = nn.R2Upsampling(self.l4_t, 2)
        self.upsample_8_4 = nn.R2Upsampling(self.l3_t, 2)
        self.upsample_4_2 = nn.R2Upsampling(self.l2_t, 2)
        self.upsample_2_1 = nn.R2Upsampling(self.l1_t, 2)

    def forwardEncoder(self, obs):
        obs_gt = nn.GeometricTensor(obs, nn.FieldType(self.r2_act, obs.shape[1] * [self.r2_act.trivial_repr]))
        feature_map_1 = self.conv_down_1(obs_gt)
        feature_map_2 = self.conv_down_2(feature_map_1)
        feature_map_4 = self.conv_down_4(feature_map_2)
        feature_map_8 = self.conv_down_8(feature_map_4)
        feature_map_16 = self.conv_down_16(feature_map_8)
        return feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16

    def forwardDecoder(self, feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16):
        concat_8 = torch.cat((feature_map_8.tensor, self.upsample_16_8(feature_map_16).tensor), dim=1)
        concat_8 = nn.GeometricTensor(concat_8, self.db_l4_t)
        feature_map_up_8 = self.conv_up_8(concat_8)

        concat_4 = torch.cat((feature_map_4.tensor, self.upsample_8_4(feature_map_up_8).tensor), dim=1)
        concat_4 = nn.GeometricTensor(concat_4, self.db_l3_t)
        feature_map_up_4 = self.conv_up_4(concat_4)

        concat_2 = torch.cat((feature_map_2.tensor, self.upsample_4_2(feature_map_up_4).tensor), dim=1)
        concat_2 = nn.GeometricTensor(concat_2, self.db_l2_t)
        feature_map_up_2 = self.conv_up_2(concat_2)

        concat_1 = torch.cat((feature_map_1.tensor, self.upsample_2_1(feature_map_up_2).tensor), dim=1)
        concat_1 = nn.GeometricTensor(concat_1, self.db_l1_t)
        feature_map_up_1 = self.conv_up_1(concat_1)

        return feature_map_up_1

    def forward(self, obs):
        feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16 = self.forwardEncoder(obs)
        return self.forwardDecoder(feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16)

# ih raise value
class EquResUExpand(torch.nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 128, 128), N=8,
                 n_middle_channels=(16, 32, 64, 128), kernel_size=3, flip=False, quotient=False, initialize=True):
        super(EquResUExpand, self).__init__()
        self.flip = flip
        self.quotient = quotient
        self.n_middle_channels = n_middle_channels
        self.N = N
        if flip:
            self.r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            self.r2_act = gspaces.Rot2dOnR2(N=N)

        if quotient:
            if flip:
                self.repr = self.r2_act.quotient_repr((None, 2))
            else:
                self.repr = self.r2_act.quotient_repr(2)
        else:
            self.repr = self.r2_act.regular_repr

        # the main unet path
        self.unet = EquResUNet(n_input_channel=n_input_channel, n_output_channel=16,
                               n_middle_channels=n_middle_channels,
                               kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)
        # group pooling for e(s)
        self.group_pool = nn.GroupPooling(nn.FieldType(self.r2_act, n_middle_channels[-1]*[self.repr]))

        self.in_hand_conv = DynamicFilterFC2(patch_shape, n_middle_channels[-1])
        self.cat_conv = conv2d(n_middle_channels[-1]*2, n_middle_channels[-1], kernel_size=3, stride=1, N=N, activation=True, flip=flip, quotient=quotient, initialize=initialize)
        self.output_layer = torch.nn.Sequential(
            conv2d(16, 16, kernel_size=3, stride=1, N=N, last=True, flip=flip, quotient=quotient, initialize=initialize),
            nn.R2Conv(nn.FieldType(self.r2_act, 16 * [self.r2_act.trivial_repr]),
                      nn.FieldType(self.r2_act, n_primitives * [self.r2_act.trivial_repr]),
                      kernel_size=1, initialize=initialize)
        )

    def forward(self, obs, in_hand):
        feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16 = self.unet.forwardEncoder(obs)

        es_i = self.group_pool(feature_map_16).tensor
        es_repeat_n = 256 // es_i.shape[1]
        es = es_i.repeat(1, es_repeat_n, 1, 1)

        in_hand_out = self.in_hand_conv(in_hand)
        ih_repeat_n = self.N
        if self.flip:
            ih_repeat_n *= 2
        if self.quotient:
            ih_repeat_n //= 2
        in_hand_out = in_hand_out.reshape(in_hand_out.shape[0], in_hand_out.shape[1], 1, 1, 1).repeat(1, 1, ih_repeat_n, 8, 8)
        in_hand_out = in_hand_out.reshape(in_hand_out.shape[0], in_hand_out.shape[1] * in_hand_out.shape[2],
                                          in_hand_out.shape[3], in_hand_out.shape[4])
        concat_16 = torch.cat((feature_map_16.tensor, in_hand_out), dim=1)
        concat_16 = nn.GeometricTensor(concat_16, nn.FieldType(self.r2_act, 2 * self.n_middle_channels[-1] * [self.repr]))
        feature_map_up_16 = self.cat_conv(concat_16)

        feature_map_up_1 = self.unet.forwardDecoder(feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_up_16)
        out = self.output_layer(feature_map_up_1).tensor

        return out, es

class EquResUDFReg(torch.nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 128, 128), N=8, df_channel=16,
                 n_middle_channels=(16, 32, 64, 128), kernel_size=3, flip=False, quotient=False, initialize=True):
        assert n_primitives == 2
        super().__init__()
        self.N = N
        if flip:
            self.r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            self.r2_act = gspaces.Rot2dOnR2(N=N)

        if quotient:
            if flip:
                self.repr = self.r2_act.quotient_repr((None, 2))
            else:
                self.repr = self.r2_act.quotient_repr(2)
        else:
            self.repr = self.r2_act.regular_repr

        self.df_channel = df_channel

        # the main unet path
        self.unet = EquResUNet(n_input_channel=n_input_channel, n_output_channel=self.df_channel,
                               n_middle_channels=n_middle_channels,
                               kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)
        # group pooling for e(s)
        self.group_pool = nn.GroupPooling(nn.FieldType(self.r2_act, n_middle_channels[-1]*[self.repr]))
        # dynamic filter
        n_weight = 6
        if quotient and not flip:
            n_weight = 2
        elif quotient and flip:
            n_weight = 4
        elif not quotient and flip:
            n_weight = 12
        self.filter = DynamicFilterFC2(in_shape=patch_shape, out_n=self.df_channel*self.df_channel*n_weight*self.N + self.df_channel)
        self.dynamic_filter = R2ConvDF(nn.FieldType(self.r2_act, self.df_channel * [self.repr]),
                                       nn.FieldType(self.r2_act, self.df_channel * [self.repr]),
                                       kernel_size=3, padding=1)
        # output q values
        self.pick_q_values = torch.nn.Sequential(
            conv2d(self.df_channel, self.df_channel, kernel_size=3, stride=1, N=N, last=True, flip=flip, quotient=quotient, initialize=initialize),
            nn.R2Conv(nn.FieldType(self.r2_act, self.df_channel * [self.r2_act.trivial_repr]),
                      nn.FieldType(self.r2_act, 1 * [self.r2_act.trivial_repr]), kernel_size=1, initialize=initialize)
        )
        self.place_q_values = torch.nn.Sequential(
            conv2d(self.df_channel, self.df_channel, kernel_size=3, stride=1, N=N, last=True, flip=flip, quotient=quotient, initialize=initialize),
            nn.R2Conv(nn.FieldType(self.r2_act, self.df_channel * [self.r2_act.trivial_repr]),
                      nn.FieldType(self.r2_act, 1 * [self.r2_act.trivial_repr]), kernel_size=1, initialize=initialize)
        )

    def forward(self, obs, in_hand):
        feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16 = self.unet.forwardEncoder(obs)

        es_i = self.group_pool(feature_map_16).tensor
        repeat_n = 256 // es_i.shape[1]
        es = es_i.repeat(1, repeat_n, 1, 1)

        feature_map_up_1 = self.unet.forwardDecoder(feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16)

        batch_size = obs.shape[0]
        place_feature = feature_map_up_1
        weight_bias = self.filter(in_hand)

        df_outputs = []
        for i in range(batch_size):
            weight = weight_bias[i, :-self.df_channel]
            bias = weight_bias[i, -self.df_channel:]
            df_outputs.append(self.dynamic_filter.forwardDynamicFilter(place_feature[i:i + 1], weight, bias).tensor)
        place_feature = torch.cat(df_outputs)
        place_feature = F.relu(place_feature)
        place_feature = nn.GeometricTensor(place_feature, nn.FieldType(self.r2_act, self.df_channel * [self.repr]))

        pick_q_values = self.pick_q_values(feature_map_up_1)
        place_q_values = self.place_q_values(place_feature)

        out = torch.cat((pick_q_values.tensor, place_q_values.tensor), dim=1)
        return out, es

class EquResU(torch.nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 128, 128), N=8,
                 n_middle_channels=(16, 32, 64, 128), kernel_size=3, flip=False, quotient=False, initialize=True):
        assert n_primitives == 2
        super().__init__()
        self.N = N
        if flip:
            self.r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            self.r2_act = gspaces.Rot2dOnR2(N=N)

        if quotient:
            if flip:
                self.repr = self.r2_act.quotient_repr((None, 2))
            else:
                self.repr = self.r2_act.quotient_repr(2)
        else:
            self.repr = self.r2_act.regular_repr

        # the main unet path
        self.unet = EquResUNet(n_input_channel=n_input_channel, n_output_channel=16,
                               n_middle_channels=n_middle_channels,
                               kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, initialize=initialize)
        # group pooling for e(s)
        self.group_pool = nn.GroupPooling(nn.FieldType(self.r2_act, n_middle_channels[-1]*[self.repr]))

        # output q values
        self.pick_q_values = torch.nn.Sequential(
            conv2d(16, 16, kernel_size=3, stride=1, N=N, last=True, flip=flip, quotient=quotient, initialize=initialize),
            nn.R2Conv(nn.FieldType(self.r2_act, 16 * [self.r2_act.trivial_repr]),
                      nn.FieldType(self.r2_act, 1 * [self.r2_act.trivial_repr]), kernel_size=1, initialize=initialize)
        )
        self.place_q_values = torch.nn.Sequential(
            conv2d(16, 16, kernel_size=3, stride=1, N=N, last=True, flip=flip, quotient=quotient, initialize=initialize),
            nn.R2Conv(nn.FieldType(self.r2_act, 16 * [self.r2_act.trivial_repr]),
                      nn.FieldType(self.r2_act, 1 * [self.r2_act.trivial_repr]), kernel_size=1, initialize=initialize)
        )

    def forward(self, obs, in_hand):
        feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16 = self.unet.forwardEncoder(obs)

        es_i = self.group_pool(feature_map_16).tensor
        repeat_n = 256 // es_i.shape[1]
        es = es_i.repeat(1, repeat_n, 1, 1)

        feature_map_up_1 = self.unet.forwardDecoder(feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16)

        pick_q_values = self.pick_q_values(feature_map_up_1)
        place_q_values = self.place_q_values(feature_map_up_1)

        out = torch.cat((pick_q_values.tensor, place_q_values.tensor), dim=1)
        return out, es

class EquResUDFRegMix(torch.nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 128, 128), N=8, df_channel=16,
                 n_middle_channels=(16, 32, 64, 128), kernel_size=3, flip=False, quotient=False, mix_type=0, u_out_reflect=True, initialize=True):
        assert n_primitives == 2
        super().__init__()
        self.N = N
        if flip:
            self.r2_act = gspaces.FlipRot2dOnR2(N=N)
        else:
            self.r2_act = gspaces.Rot2dOnR2(N=N)

        if flip and quotient and u_out_reflect:
            self.repr = self.r2_act.quotient_repr((None, 2))
        elif flip and quotient and not u_out_reflect:
            self.repr = self.r2_act.quotient_repr((0, 2))
        elif flip and not quotient and not u_out_reflect:
            self.repr = self.r2_act.quotient_repr((0, 1))
        elif not flip and quotient:
            self.repr = self.r2_act.quotient_repr(2)
        else:
            self.repr = self.r2_act.regular_repr

        self.df_channel = df_channel

        # the main unet path
        self.unet = EquResUNetMix(n_input_channel=n_input_channel, n_output_channel=self.df_channel, n_middle_channels=n_middle_channels,
                                  kernel_size=kernel_size, N=N, flip=flip, quotient=quotient, mix_type=mix_type, out_reflect=u_out_reflect, initialize=initialize)
        # group pooling for e(s)
        if mix_type == 0:
            self.group_pool = nn.GroupPooling(nn.FieldType(self.r2_act, n_middle_channels[-1]*[self.repr]))
        elif mix_type == 1:
            self.group_pool = nn.GroupPooling(nn.FieldType(self.r2_act, n_middle_channels[-1]//2*[self.r2_act.regular_repr]+n_middle_channels[-1]//2*[self.r2_act.quotient_repr((0, 1))]))
        elif mix_type == 2:
            self.group_pool = nn.GroupPooling(nn.FieldType(self.r2_act, 1*n_middle_channels[-1]//4*[self.r2_act.regular_repr]+3*n_middle_channels[-1]//4*[self.r2_act.quotient_repr((0, 1))]))
        elif mix_type == 3:
            self.group_pool = nn.GroupPooling(nn.FieldType(self.r2_act, 1*n_middle_channels[-1]//8*[self.r2_act.regular_repr]+7*n_middle_channels[-1]//8*[self.r2_act.quotient_repr((0, 1))]))
        else:
            raise NotImplementedError
        # dynamic filter
        n_weight = 6
        if quotient and not flip:
            n_weight = 2
        elif quotient and flip and not u_out_reflect:
            n_weight = 4
        elif not quotient and flip and u_out_reflect:
            n_weight = 12
        elif not quotient and flip and not u_out_reflect:
            if N == 16:
                n_weight = 3.125
            elif N == 8:
                n_weight = 3.25
            elif N == 4:
                n_weight = 3.5
            else:
                raise NotImplementedError
        self.filter = DynamicFilterFC2(in_shape=patch_shape, out_n=int(self.df_channel*self.df_channel*n_weight*self.N) + self.df_channel)
        self.dynamic_filter = R2ConvDF(nn.FieldType(self.r2_act, self.df_channel * [self.repr]),
                                       nn.FieldType(self.r2_act, self.df_channel * [self.repr]),
                                       kernel_size=3, padding=1)
        # output q values
        self.pick_q_values = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.r2_act, self.df_channel * [self.repr]),
                      nn.FieldType(self.r2_act, self.df_channel * [self.r2_act.trivial_repr]),
                      kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, self.df_channel * [self.r2_act.trivial_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.r2_act, self.df_channel * [self.r2_act.trivial_repr]),
                      nn.FieldType(self.r2_act, 1 * [self.r2_act.trivial_repr]), kernel_size=1, initialize=initialize)
        )
        self.place_q_values = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.r2_act, self.df_channel * [self.repr]),
                      nn.FieldType(self.r2_act, self.df_channel * [self.r2_act.trivial_repr]),
                      kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, self.df_channel * [self.r2_act.trivial_repr]), inplace=True),
            nn.R2Conv(nn.FieldType(self.r2_act, self.df_channel * [self.r2_act.trivial_repr]),
                      nn.FieldType(self.r2_act, 1 * [self.r2_act.trivial_repr]), kernel_size=1, initialize=initialize)
        )

    def forward(self, obs, in_hand):
        feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16 = self.unet.forwardEncoder(obs)

        es_i = self.group_pool(feature_map_16).tensor
        repeat_n = 256 // es_i.shape[1]
        es = es_i.repeat(1, repeat_n, 1, 1)

        feature_map_up_1 = self.unet.forwardDecoder(feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16)

        batch_size = obs.shape[0]
        place_feature = feature_map_up_1
        weight_bias = self.filter(in_hand)

        df_outputs = []
        for i in range(batch_size):
            weight = weight_bias[i, :-self.df_channel]
            bias = weight_bias[i, -self.df_channel:]
            df_outputs.append(self.dynamic_filter.forwardDynamicFilter(place_feature[i:i + 1], weight, bias).tensor)
        place_feature = torch.cat(df_outputs)
        place_feature = F.relu(place_feature)
        place_feature = nn.GeometricTensor(place_feature, nn.FieldType(self.r2_act, self.df_channel * [self.repr]))

        pick_q_values = self.pick_q_values(feature_map_up_1)
        place_q_values = self.place_q_values(place_feature)

        out = torch.cat((pick_q_values.tensor, place_q_values.tensor), dim=1)
        return out, es

class EquResUDFRegNOut(torch.nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 128, 128), N=8, df_channel=16,
                 n_middle_channels=(16, 32, 64, 128), kernel_size=5, quotient=True, last_quotient=False, out_type='index', initialize=True):
        assert n_primitives == 2
        assert kernel_size in [3, 5]
        assert out_type in ['index', 'sum']
        if last_quotient:
            assert not quotient
        super().__init__()
        self.N = N
        self.n_rotations = N//2
        self.r2_act = gspaces.Rot2dOnR2(N=N)
        self.df_channel = df_channel
        self.quotient = quotient
        self.out_type = out_type
        if quotient:
            self.repr = self.r2_act.quotient_repr(2)
            if kernel_size == 3:
                n_weight = 2
            else:
                n_weight = 3.5
        else:
            self.repr = self.r2_act.regular_repr
            if kernel_size == 3:
                n_weight = 6
            else:
                n_weight = 11

        # the main unet path
        self.unet = EquResUNet(n_input_channel=n_input_channel, n_output_channel=self.df_channel,
                               n_middle_channels=n_middle_channels,
                               kernel_size=kernel_size, N=N, flip=False, quotient=quotient, initialize=initialize)
        # dynamic filter
        self.filter = DynamicFilterFC2(in_shape=patch_shape, out_n=self.df_channel*self.df_channel*int(n_weight*self.N) + self.df_channel)
        self.dynamic_filter = R2ConvDF(nn.FieldType(self.r2_act, self.df_channel * [self.repr]),
                                       nn.FieldType(self.r2_act, self.df_channel * [self.repr]),
                                       kernel_size=kernel_size, padding=kernel_size//2)
        # output q values
        if last_quotient:
            output_repr = 1 * [self.r2_act.quotient_repr(2)]
        else:
            output_repr = 1 * [self.repr]
        self.pick_q_values = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.r2_act, self.df_channel * [self.repr]),
                      nn.FieldType(self.r2_act, output_repr), kernel_size=1, initialize=initialize)
        )
        self.place_q_values = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.r2_act, self.df_channel * [self.repr]),
                      nn.FieldType(self.r2_act, output_repr), kernel_size=1, initialize=initialize)
        )

    def getRegularIdx(self):
        idx = torch.tensor(list(range(0, self.n_rotations)))
        return idx

    def forward(self, obs, in_hand):
        feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16 = self.unet.forwardEncoder(obs)
        feature_map_up_1 = self.unet.forwardDecoder(feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16)

        batch_size = obs.shape[0]
        place_feature = feature_map_up_1
        weight_bias = self.filter(in_hand)

        df_outputs = []
        for i in range(batch_size):
            weight = weight_bias[i, :-self.df_channel]
            bias = weight_bias[i, -self.df_channel:]
            df_outputs.append(self.dynamic_filter.forwardDynamicFilter(place_feature[i:i + 1], weight, bias).tensor)
        place_feature = torch.cat(df_outputs)
        place_feature = F.relu(place_feature)
        place_feature = nn.GeometricTensor(place_feature, nn.FieldType(self.r2_act, self.df_channel * [self.repr]))

        pick_q_values = self.pick_q_values(feature_map_up_1)
        place_q_values = self.place_q_values(place_feature)

        out = torch.stack((pick_q_values.tensor, place_q_values.tensor), dim=1)

        if not self.quotient:
            if self.out_type == 'index':
                idx = torch.tensor(list(range(0, self.n_rotations)))
                out = out[:, :, idx]
            else:
                out = out.reshape(batch_size, 2, 2, self.n_rotations, out.shape[-2], out.shape[-1])
                out = out[:, :, 0] + out[:, :, 1]

        return out, None

class EquResUExpandRegNOut(torch.nn.Module):
    def __init__(self, n_input_channel=1, n_primitives=1, patch_shape=(1, 24, 24), domain_shape=(1, 128, 128), N=8,
                 n_middle_channels=(16, 32, 64, 128), kernel_size=5, quotient=True, last_quotient=False, out_type='index', initialize=True):
        assert n_primitives == 2
        assert kernel_size in [3, 5]
        assert out_type in ['index', 'sum']
        if last_quotient:
            assert not quotient
        super().__init__()
        self.N = N
        self.n_rotations = N//2
        self.r2_act = gspaces.Rot2dOnR2(N=N)
        self.quotient = quotient
        self.out_type = out_type
        self.n_middle_channels = n_middle_channels

        if quotient:
            self.repr = self.r2_act.quotient_repr(2)
        else:
            self.repr = self.r2_act.regular_repr

        # the main unet path
        self.unet = EquResUNet(n_input_channel=n_input_channel, n_output_channel=n_middle_channels[0],
                               n_middle_channels=n_middle_channels,
                               kernel_size=kernel_size, N=N, flip=False, quotient=quotient, initialize=initialize)
        self.in_hand_conv = DynamicFilterFC2(patch_shape, n_middle_channels[-1])
        self.cat_conv = conv2d(n_middle_channels[-1]*2, n_middle_channels[-1], kernel_size=3, stride=1, N=N, activation=True, quotient=quotient, initialize=initialize)

        # output q values
        if last_quotient:
            output_repr = 1 * [self.r2_act.quotient_repr(2)]
        else:
            output_repr = 1 * [self.repr]
        self.pick_q_values = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.r2_act, n_middle_channels[0] * [self.repr]),
                      nn.FieldType(self.r2_act, output_repr), kernel_size=1, initialize=initialize)
        )
        self.place_q_values = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.r2_act, n_middle_channels[0] * [self.repr]),
                      nn.FieldType(self.r2_act, output_repr), kernel_size=1, initialize=initialize)
        )

    def getRegularIdx(self):
        idx = torch.tensor(list(range(0, self.n_rotations)))
        return idx

    def forward(self, obs, in_hand):
        batch_size = obs.shape[0]

        feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_16 = self.unet.forwardEncoder(obs)

        in_hand_out = self.in_hand_conv(in_hand)
        ih_repeat_n = self.N
        in_hand_out = in_hand_out.reshape(in_hand_out.shape[0], in_hand_out.shape[1], 1, 1, 1).repeat(1, 1, ih_repeat_n, 8, 8)
        in_hand_out = in_hand_out.reshape(in_hand_out.shape[0], in_hand_out.shape[1] * in_hand_out.shape[2],
                                          in_hand_out.shape[3], in_hand_out.shape[4])
        concat_16 = torch.cat((feature_map_16.tensor, in_hand_out), dim=1)
        concat_16 = nn.GeometricTensor(concat_16, nn.FieldType(self.r2_act, 2 * self.n_middle_channels[-1] * [self.repr]))
        feature_map_up_16 = self.cat_conv(concat_16)

        feature_map_up_1 = self.unet.forwardDecoder(feature_map_1, feature_map_2, feature_map_4, feature_map_8, feature_map_up_16)


        pick_q_values = self.pick_q_values(feature_map_up_1)
        place_q_values = self.place_q_values(feature_map_up_1)

        out = torch.stack((pick_q_values.tensor, place_q_values.tensor), dim=1)

        if not self.quotient:
            if self.out_type == 'index':
                idx = torch.tensor(list(range(0, self.n_rotations)))
                out = out[:, :, idx]
            else:
                out = out.reshape(batch_size, 2, 2, self.n_rotations, out.shape[-2], out.shape[-1])
                out = out[:, :, 0] + out[:, :, 1]

        return out, None

class EquShiftQ2DF(torch.nn.Module):
    def __init__(self, image_shape, n_rotations, n_primitives, kernel_size=3, df_channel=8, n_hidden=128, quotient=True, last_quotient=False, out_type='index', initialize=True):
        super().__init__()
        assert kernel_size in [3, 5]
        assert out_type in ['index', 'sum']
        if last_quotient:
            assert not quotient
        self.n_rotations = n_rotations
        self.n_primitives = n_primitives
        self.N = n_rotations * 2
        self.r2_act = gspaces.Rot2dOnR2(N=self.N)
        self.quotient = quotient
        self.out_type = out_type
        if quotient:
            self.repr = self.r2_act.quotient_repr(2)
            if kernel_size == 3:
                n_weight = 2
            else:
                n_weight = 3.5
        else:
            self.repr = self.r2_act.regular_repr
            if kernel_size == 3:
                n_weight = 6
            else:
                n_weight = 11
        self.df_channel = df_channel

        if kernel_size == 3:
            self.patch_conv = torch.nn.Sequential(
                EquCNNEnc(1, n_hidden, self.N, kernel_size=3, out_size=8, quotient=quotient, initialize=initialize),
                nn.R2Conv(nn.FieldType(self.r2_act, n_hidden * [self.repr]),
                          nn.FieldType(self.r2_act, self.df_channel * [self.repr]), kernel_size=3, stride=2, initialize=initialize),
                nn.ReLU(nn.FieldType(self.r2_act, self.df_channel * [self.repr]), inplace=True),
            )
        else:
            self.patch_conv = torch.nn.Sequential(
                EquCNNEnc(1, n_hidden, self.N, kernel_size=5, out_size=10, quotient=quotient, initialize=initialize),
                nn.R2Conv(nn.FieldType(self.r2_act, n_hidden * [self.repr]),
                          nn.FieldType(self.r2_act, self.df_channel * [self.repr]), kernel_size=5, padding=2, stride=2, initialize=initialize),
                nn.ReLU(nn.FieldType(self.r2_act, self.df_channel * [self.repr]), inplace=True),
            )

        self.es_fc = torch.nn.Sequential(
            torch.nn.Linear(256*8*8, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
        )
        self.in_hand_conv = torch.nn.Sequential(
            DynamicFilterFC2(in_shape=(image_shape[0]-1, image_shape[1], image_shape[2]), out_n=512),
            torch.nn.ReLU(inplace=True),
        )
        self.df_fc = torch.nn.Sequential(
            torch.nn.Linear(1024, self.df_channel * self.df_channel * int(self.N * n_weight) + self.df_channel)
        )

        self.dynamic_filter = R2ConvDF(nn.FieldType(self.r2_act, self.df_channel * [self.repr]),
                                       nn.FieldType(self.r2_act, self.df_channel * [self.repr]),
                                       kernel_size=kernel_size)
        if last_quotient:
            output_repr = n_primitives * [self.r2_act.quotient_repr(2)]
        else:
            output_repr = n_primitives * [self.repr]
        self.conv_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.r2_act, self.df_channel * [self.repr]),
                      nn.FieldType(self.r2_act, output_repr),
                      kernel_size=1, initialize=initialize)
        )

    def forward(self, obs_encoding, patch):
        batch_size = patch.size(0)
        patch_channel = patch.shape[1]
        image_patch = patch[:, :1]
        in_hand = patch[:, 1:]
        es_vector = self.es_fc(obs_encoding.reshape(obs_encoding.shape[0], -1))
        in_hand_vector = self.in_hand_conv(in_hand)
        df_weight_in = torch.cat((es_vector, in_hand_vector), 1)
        weight_bias = self.df_fc(df_weight_in)

        patch_conv_out = self.patch_conv(image_patch)

        df_outputs = []
        for i in range(batch_size):
            weight = weight_bias[i, :-self.df_channel]
            bias = weight_bias[i, -self.df_channel:]
            df_outputs.append(self.dynamic_filter.forwardDynamicFilter(patch_conv_out[i:i + 1], weight, bias).tensor)
        feature = torch.cat(df_outputs)
        feature = F.relu(feature)

        feature = nn.GeometricTensor(feature, nn.FieldType(self.r2_act, self.df_channel * [self.repr]))

        x = self.conv_2(feature).tensor

        x = x.reshape(batch_size, self.n_primitives, -1)

        if not self.quotient:
            if self.out_type == 'index':
                idx = torch.tensor(list(range(0, self.N)))
                idx = idx[:self.n_rotations]
                x = x[:, :, idx]
            else:
                x = x.reshape(batch_size, self.n_primitives, 2, self.n_rotations)
                x = x[:, :, 0] + x[:, :, 1]
        return x

# 1x1 df
class EquShiftQ2DF3(torch.nn.Module):
    def __init__(self, image_shape, n_rotations, n_primitives, kernel_size=3, df_channel=16, n_hidden=128, quotient=True, last_quotient=False, out_type='index', initialize=True):
        super().__init__()
        assert out_type in ['index', 'sum']
        if last_quotient:
            assert not quotient
        self.n_rotations = n_rotations
        self.n_primitives = n_primitives
        self.N = n_rotations * 2
        self.r2_act = gspaces.Rot2dOnR2(N=self.N)
        self.quotient = quotient
        self.out_type = out_type
        if quotient:
            self.repr = self.r2_act.quotient_repr(2)
            n_weight = 0.5
        else:
            self.repr = self.r2_act.regular_repr
            n_weight = 1
        self.df_channel = df_channel

        self.patch_conv = torch.nn.Sequential(
            EquCNNEnc(1, n_hidden, self.N, kernel_size=kernel_size, out_size=8, quotient=quotient,
                      initialize=initialize),
            nn.R2Conv(nn.FieldType(self.r2_act, n_hidden * [self.repr]),
                      nn.FieldType(self.r2_act, df_channel * 2 * [self.repr]),
                      kernel_size=3, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, df_channel * 2 * [self.repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, df_channel * 2 * [self.repr]), 2),
            nn.R2Conv(nn.FieldType(self.r2_act, df_channel * 2 * [self.repr]),
                      nn.FieldType(self.r2_act, df_channel * [self.repr]),
                      kernel_size=3, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, df_channel * [self.repr]), inplace=True),
        )

        self.es_fc = torch.nn.Sequential(
            torch.nn.Linear(256*8*8, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
        )
        self.in_hand_conv = torch.nn.Sequential(
            DynamicFilterFC2(in_shape=(image_shape[0]-1, image_shape[1], image_shape[2]), out_n=512),
            torch.nn.ReLU(inplace=True),
        )
        self.df_fc = torch.nn.Sequential(
            torch.nn.Linear(1024, self.df_channel * self.df_channel * int(self.N * n_weight) + self.df_channel)
        )

        self.dynamic_filter = R2ConvDF(nn.FieldType(self.r2_act, self.df_channel * [self.repr]),
                                       nn.FieldType(self.r2_act, self.df_channel * [self.repr]),
                                       kernel_size=1)
        if last_quotient:
            output_repr = n_primitives * [self.r2_act.quotient_repr(2)]
        else:
            output_repr = n_primitives * [self.repr]
        self.conv_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.r2_act, self.df_channel * [self.repr]),
                      nn.FieldType(self.r2_act, output_repr),
                      kernel_size=1, initialize=initialize)
        )

    def forward(self, obs_encoding, patch):
        batch_size = patch.size(0)
        patch_channel = patch.shape[1]
        image_patch = patch[:, :1]
        in_hand = patch[:, 1:]
        es_vector = self.es_fc(obs_encoding.reshape(obs_encoding.shape[0], -1))
        in_hand_vector = self.in_hand_conv(in_hand)
        df_weight_in = torch.cat((es_vector, in_hand_vector), 1)
        weight_bias = self.df_fc(df_weight_in)

        patch_conv_out = self.patch_conv(image_patch)

        df_outputs = []
        for i in range(batch_size):
            weight = weight_bias[i, :-self.df_channel]
            bias = weight_bias[i, -self.df_channel:]
            df_outputs.append(self.dynamic_filter.forwardDynamicFilter(patch_conv_out[i:i + 1], weight, bias).tensor)
        feature = torch.cat(df_outputs)
        feature = F.relu(feature)

        feature = nn.GeometricTensor(feature, nn.FieldType(self.r2_act, self.df_channel * [self.repr]))

        x = self.conv_2(feature).tensor

        x = x.reshape(batch_size, self.n_primitives, -1)

        if not self.quotient:
            if self.out_type == 'index':
                idx = torch.tensor(list(range(0, self.N)))
                idx = idx[:self.n_rotations]
                x = x[:, :, idx]
            else:
                x = x.reshape(batch_size, self.n_primitives, 2, self.n_rotations)
                x = x[:, :, 0] + x[:, :, 1]
        return x

# 1x1 df
class EquShiftQ2DF3P40(torch.nn.Module):
    def __init__(self, image_shape, n_rotations, n_primitives, kernel_size=3, df_channel=16, n_hidden=128, quotient=True, last_quotient=False, out_type='index', initialize=True):
        super().__init__()
        assert out_type in ['index', 'sum']
        if last_quotient:
            assert not quotient
        self.n_rotations = n_rotations
        self.n_primitives = n_primitives
        self.N = n_rotations * 2
        self.r2_act = gspaces.Rot2dOnR2(N=self.N)
        self.quotient = quotient
        self.out_type = out_type
        if quotient:
            self.repr = self.r2_act.quotient_repr(2)
            n_weight = 0.5
        else:
            self.repr = self.r2_act.regular_repr
            n_weight = 1
        self.df_channel = df_channel

        self.patch_conv = torch.nn.Sequential(
            EquCNNEncP40(1, n_hidden, self.N, kernel_size=kernel_size, quotient=quotient,
                      initialize=initialize),
            nn.R2Conv(nn.FieldType(self.r2_act, n_hidden * [self.repr]),
                      nn.FieldType(self.r2_act, df_channel * 2 * [self.repr]),
                      kernel_size=3, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, df_channel * 2 * [self.repr]), inplace=True),
            nn.PointwiseMaxPool(nn.FieldType(self.r2_act, df_channel * 2 * [self.repr]), 2),
            nn.R2Conv(nn.FieldType(self.r2_act, df_channel * 2 * [self.repr]),
                      nn.FieldType(self.r2_act, df_channel * [self.repr]),
                      kernel_size=3, initialize=initialize),
            nn.ReLU(nn.FieldType(self.r2_act, df_channel * [self.repr]), inplace=True),
        )

        self.es_fc = torch.nn.Sequential(
            torch.nn.Linear(256*8*8, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
        )
        self.in_hand_conv = torch.nn.Sequential(
            DynamicFilterFC3(in_shape=(image_shape[0]-1, image_shape[1], image_shape[2]), out_n=512),
            torch.nn.ReLU(inplace=True),
        )
        self.df_fc = torch.nn.Sequential(
            torch.nn.Linear(1024, self.df_channel * self.df_channel * int(self.N * n_weight) + self.df_channel)
        )

        self.dynamic_filter = R2ConvDF(nn.FieldType(self.r2_act, self.df_channel * [self.repr]),
                                       nn.FieldType(self.r2_act, self.df_channel * [self.repr]),
                                       kernel_size=1)
        if last_quotient:
            output_repr = n_primitives * [self.r2_act.quotient_repr(2)]
        else:
            output_repr = n_primitives * [self.repr]
        self.conv_2 = torch.nn.Sequential(
            nn.R2Conv(nn.FieldType(self.r2_act, self.df_channel * [self.repr]),
                      nn.FieldType(self.r2_act, output_repr),
                      kernel_size=1, initialize=initialize)
        )

    def forward(self, obs_encoding, patch):
        batch_size = patch.size(0)
        patch_channel = patch.shape[1]
        image_patch = patch[:, :1]
        in_hand = patch[:, 1:]
        es_vector = self.es_fc(obs_encoding.reshape(obs_encoding.shape[0], -1))
        in_hand_vector = self.in_hand_conv(in_hand)
        df_weight_in = torch.cat((es_vector, in_hand_vector), 1)
        weight_bias = self.df_fc(df_weight_in)

        patch_conv_out = self.patch_conv(image_patch)

        df_outputs = []
        for i in range(batch_size):
            weight = weight_bias[i, :-self.df_channel]
            bias = weight_bias[i, -self.df_channel:]
            df_outputs.append(self.dynamic_filter.forwardDynamicFilter(patch_conv_out[i:i + 1], weight, bias).tensor)
        feature = torch.cat(df_outputs)
        feature = F.relu(feature)

        feature = nn.GeometricTensor(feature, nn.FieldType(self.r2_act, self.df_channel * [self.repr]))

        x = self.conv_2(feature).tensor

        x = x.reshape(batch_size, self.n_primitives, -1)

        if not self.quotient:
            if self.out_type == 'index':
                idx = torch.tensor(list(range(0, self.N)))
                idx = idx[:self.n_rotations]
                x = x[:, :, idx]
            else:
                x = x.reshape(batch_size, self.n_primitives, 2, self.n_rotations)
                x = x[:, :, 0] + x[:, :, 1]
        return x

def testQ2():
    obs_encoding = torch.zeros((1, 256, 8, 8)).to('cuda')
    patch = torch.zeros(1, 2, 24, 24).to('cuda')
    patch[0, 0, 10:20, 10:20] = 1

    cnn = EquShiftQ2DF3((2, 24, 24), 16, 2, kernel_size=9, n_hidden=32, quotient=False, last_quotient=True, initialize=False).to('cuda')
    out = cnn(obs_encoding, patch)
    assert out.shape == torch.Size([1, 2, 16])

    cnn = EquShiftQ2DF3((2, 24, 24), 16, 2, kernel_size=7, n_hidden=32, quotient=False, last_quotient=True, initialize=False).to('cuda')
    out = cnn(obs_encoding, patch)
    assert out.shape == torch.Size([1, 2, 16])

    cnn = EquShiftQ2DF3((2, 24, 24), 16, 2, kernel_size=5, n_hidden=64, quotient=False, last_quotient=True, initialize=False).to('cuda')
    out = cnn(obs_encoding, patch)
    assert out.shape == torch.Size([1, 2, 16])

    cnn = EquShiftQ2DF3((2, 24, 24), 16, 2, kernel_size=3, n_hidden=128, quotient=False, last_quotient=True, initialize=False).to('cuda')
    out = cnn(obs_encoding, patch)
    assert out.shape == torch.Size([1, 2, 16])

    cnn = EquShiftQ2DF((2, 24, 24), 8, 2, kernel_size=3, initialize=False).to('cuda')
    out = cnn(obs_encoding, patch)
    assert out.shape == torch.Size([1, 2, 8])

    cnn = EquShiftQ2DF((2, 24, 24), 8, 2, kernel_size=5, initialize=False).to('cuda')
    out = cnn(obs_encoding, patch)
    assert out.shape == torch.Size([1, 2, 8])

    cnn = EquShiftQ2DF((2, 24, 24), 8, 2, kernel_size=3, quotient=False, initialize=False).to('cuda')
    out = cnn(obs_encoding, patch)
    assert out.shape == torch.Size([1, 2, 8])

    cnn = EquShiftQ2DF((2, 24, 24), 8, 2, kernel_size=3, quotient=False, out_type='sum', initialize=False).to('cuda')
    out = cnn(obs_encoding, patch)
    assert out.shape == torch.Size([1, 2, 8])

def testQ1():
    N=16
    obs = torch.zeros((5, 1, 128, 128)).to('cuda')
    inh = torch.zeros((5, 1, 24, 24)).to('cuda')
    obs[0, 0, 30:50, 20:60] = 1
    inh[0, 0, 8:18, 8:18] = 1

    unet = EquResUNet(1, 16, (8, 16, 32, 64), 3, N=N, flip=False, quotient=False, initialize=False).to('cuda')
    out = unet(obs)

    assert out.shape == torch.Size([5, 16*N, 128, 128])

    unet = EquResUExpandRegNOut(1, 2, N=N, n_middle_channels=(16, 16, 32, 64), kernel_size=3, quotient=False, last_quotient=True, initialize=False).to('cuda')
    out = unet(obs, inh)
    assert out[0].shape == torch.Size([5, 2, N // 2, 128, 128])

    unet = EquResUDFRegNOut(1, 2, N=N, n_middle_channels=(16, 16, 32, 64), kernel_size=5, quotient=True, initialize=False).to('cuda')
    out = unet(obs, inh)
    assert out[0].shape == torch.Size([5, 2, N//2, 128, 128])

    unet = EquResUDFRegNOut(1, 2, N=N, n_middle_channels=(16, 16, 32, 64), kernel_size=3, quotient=False, out_type='sum', initialize=False).to('cuda')
    out = unet(obs, inh)
    assert out[0].shape == torch.Size([5, 2, N//2, 128, 128])

    unet = EquResUDFRegNOut(1, 2, N=N, n_middle_channels=(16, 16, 32, 64), kernel_size=3, quotient=False, out_type='index', initialize=False).to('cuda')
    out = unet(obs, inh)
    assert out[0].shape == torch.Size([5, 2, N//2, 128, 128])

    unet = EquResUDFRegNOut(1, 2, N=N, n_middle_channels=(8, 8, 16, 32), kernel_size=3, quotient=False, out_type='sum', initialize=False).to('cuda')
    out = unet(obs, inh)
    assert out[0].shape == torch.Size([5, 2, N//2, 128, 128])

    unet = EquResUDFRegNOut(1, 2, N=N, df_channel=8, n_middle_channels=(8, 8, 16, 32), kernel_size=3, quotient=False, out_type='sum', initialize=False).to('cuda')
    out = unet(obs, inh)
    assert out[0].shape == torch.Size([5, 2, N // 2, 128, 128])

    unet = EquResUDFRegNOut(1, 2, N=N, df_channel=8, n_middle_channels=(8, 8, 16, 32), kernel_size=3, quotient=False, last_quotient=True, initialize=False).to('cuda')
    out = unet(obs, inh)
    assert out[0].shape == torch.Size([5, 2, N // 2, 128, 128])

    N=8
    unet = EquResUDFRegMix(1, 2, N=N, df_channel=8, n_middle_channels=(8, 8, 16, 32), kernel_size=3, flip=True, quotient=False, mix_type=1, initialize=False).to('cuda')
    out = unet(obs, inh)
    assert out[0].shape == torch.Size([5, 2, 128, 128])
    assert out[1].shape == torch.Size([5, 256, 8, 8])

    unet = EquResUDFRegMix(1, 2, N=N, df_channel=8, n_middle_channels=(8, 8, 16, 32), kernel_size=3, flip=True, quotient=False, mix_type=2, initialize=False).to('cuda')
    out = unet(obs, inh)
    assert out[0].shape == torch.Size([5, 2, 128, 128])
    assert out[1].shape == torch.Size([5, 256, 8, 8])

    unet = EquResUDFRegMix(1, 2, N=N, df_channel=8, n_middle_channels=(8, 8, 16, 32), kernel_size=3, flip=True, quotient=False, u_out_reflect=False, mix_type=2, initialize=False).to('cuda')
    out = unet(obs, inh)
    assert out[0].shape == torch.Size([5, 2, 128, 128])
    assert out[1].shape == torch.Size([5, 256, 8, 8])
    print(1)

if __name__ == '__main__':
    testQ1()
