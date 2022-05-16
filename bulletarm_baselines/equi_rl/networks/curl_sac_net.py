import numpy as np
from scipy import ndimage
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from bulletarm_baselines.equi_rl.utils import torch_utils

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def tieWeights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

# similar amount of parameters
class CURLSACEncoder(nn.Module):
    def __init__(self, input_shape=(2, 64, 64), output_dim=50):
        super().__init__()
        self.conv = torch.nn.Sequential(
            # 128x128
            nn.Conv2d(input_shape[0], 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 64x64
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            # 6x6
            nn.MaxPool2d(2),
            # 3x3
            nn.Conv2d(256, 1024, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1024, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x, detach=False):
        h = self.conv(x)
        if detach:
            h = h.detach()
        h_fc = self.fc(h)
        return h_fc

    def copyConvWeightsFrom(self, source):
        for i in range(len(self.conv)):
            if isinstance(self.conv[i], nn.Conv2d):
                tieWeights(src=source.conv[i], trg=self.conv[i])

class CURLSACEncoderOri(nn.Module):
    def __init__(self, input_shape=(2, 64, 64), output_dim=50):
        super().__init__()
        self.conv = torch.nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        x = torch.randn([1] + list(input_shape))
        conv_out_dim = self.conv(x).reshape(-1).shape[-1]

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(conv_out_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x, detach=False):
        h = self.conv(x)
        if detach:
            h = h.detach()
        h_fc = self.fc(h)
        return h_fc

    def copyConvWeightsFrom(self, source):
        for i in range(len(self.conv)):
            if isinstance(self.conv[i], nn.Conv2d):
                tieWeights(src=source.conv[i], trg=self.conv[i])

class CURLSACCritic(nn.Module):
    def __init__(self, encoder, encoder_output_dim=50, hidden_dim=1024, action_dim=5):
        super().__init__()
        self.encoder = encoder
        # Q1
        self.q1 = torch.nn.Sequential(
            torch.nn.Linear(encoder_output_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, 1)
        )

        # Q2
        self.q2 = torch.nn.Sequential(
            torch.nn.Linear(encoder_output_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, act, detach_encoder=False):
        obs_enc = self.encoder(obs, detach=detach_encoder)
        out_1 = self.q1(torch.cat((obs_enc, act), dim=1))
        out_2 = self.q2(torch.cat((obs_enc, act), dim=1))
        return out_1, out_2

class CURLSACGaussianPolicy(nn.Module):
    def __init__(self, encoder, encoder_output_dim=50, hidden_dim=1024, action_dim=5, action_space=None):
        super().__init__()
        self.encoder = encoder
        self.mean_linear = torch.nn.Sequential(
            torch.nn.Linear(encoder_output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, action_dim)
        )
        self.log_std_linear = torch.nn.Sequential(
            torch.nn.Linear(encoder_output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, action_dim)
        )

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

        self.apply(torch_utils.weights_init)

    def forward(self, x, detach_encoder=False):
        x = self.encoder(x, detach=detach_encoder)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, x, detach_encoder=False):
        mean, log_std = self.forward(x, detach_encoder=detach_encoder)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class CURLCNNCom(nn.Module):
    def __init__(self, encoder, encoder_output_dim=50, hidden_dim=1024, n_p=2, n_theta=1):
        super().__init__()
        self.encoder = encoder
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(encoder_output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, 9 * 3 * n_theta * n_p)
        )

        self.apply(torch_utils.weights_init)

    def forward(self, x, detach_encoder=False):
        x = self.encoder(x, detach=detach_encoder)
        q = self.fc(x)
        return q

# code for this class from: https://github.com/PhilipZRH/ferm/blob/71676490fa5442be7185c0772d677c5912659fbc/curl_sac.py#L206
#
# MIT License
#
# Copyright (c) 2020 FERM (A Framework for Efficient Robotic Manipulation) Authors (https://arxiv.org/abs/2012.07975)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
class CURL(nn.Module):
    """
    CURL
    """

    def __init__(self, z_dim, encoder, encoder_target):
        super(CURL, self).__init__()
        self.encoder = encoder

        self.encoder_target = encoder_target

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))

    def encode(self, x, detach=False, ema=False):
        """
        CURLSACEncoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits