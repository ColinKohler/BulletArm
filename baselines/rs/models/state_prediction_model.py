import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.torch_utils import Flatten, makeLayer, BottleneckBlock, UpsamplingBlock, CatConv
from data import constants

class StatePredictionModel(nn.Module):
  def __init__(self, device, out_kernels):
    super(StatePredictionModel, self).__init__()
    self.device = device

    self.obs_prediction_model = ActionPrimativeModel(device, out_kernels)
    self.reward_model = RewardModel(device)

  def forward(self, state, obs, hand_obs, deictic_obs, action):
    batch_size = obs.size(0)

    deictic_obs_, pick_obs_ = self.obs_prediction_model(deictic_obs, hand_obs, obs, action)
    hand_obs_ = utils.getHandObs(deictic_obs).cpu()
    state_ = self.getHandStates(deictic_obs.cpu(), deictic_obs_.cpu())

    if torch.sum((~state_).long()) > 0:
      num_hand_obs_empty = torch.sum((~state_).long()).item()
      empty_hand_obs = torch.zeros(num_hand_obs_empty, 1, self.hand_obs_size, self.hand_obs_size)
      empty_idx = torch.where(state_.int() == 0)
      hand_obs_[empty_idx] = data_utils.convertDepthToOneHot(empty_hand_obs,
                                                             self.config.num_depth_classes)


    reward = self.reward_model(obs_)

    return obs_, hand_obs_, deictic_obs_, reward

  def loadModel(self, model_state_dict):
    self_state = self.state_dict()
    for name, param in model_state_dict.items():
      self_state[name].copy_(param)

class ActionPrimativeModel(nn.Module):
  def __init__(self, device, out_kernels):
    super(ActionPrimativeModel, self).__init__()
    self.device = device

    self.in_kernels = 2 * out_kernels
    self.out_kernels = out_kernels

    input_kernels = 32
    self.conv_1 = nn.Sequential(
      nn.Conv2d(self.in_kernels, input_kernels, 3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(input_kernels),
      nn.LeakyReLU(0.01, inplace=True)
    )

    self.layer_1 = makeLayer(BottleneckBlock, 32, 32, 1, stride=2)
    self.layer_2 = makeLayer(BottleneckBlock, 64, 64, 2, stride=2)
    self.layer_3 = makeLayer(BottleneckBlock, 128, 128, 3, stride=2)
    self.layer_4 = makeLayer(BottleneckBlock, 256, 256, 1, stride=2)

    self.forward_layer = nn.Sequential(
      nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True),
      nn.BatchNorm2d(512),
      nn.LeakyReLU(0.01, inplace=True),
    )

    self.up_proj_4 = UpsamplingBlock(512, 512)
    self.up_proj_3 = UpsamplingBlock(256, 256)
    self.up_proj_2 = UpsamplingBlock(128, 128)
    self.up_proj_1 = UpsamplingBlock(64, 64)

    self.cat_conv_4 = CatConv(512, 256, 256)
    self.cat_conv_3 = CatConv(256, 128, 128)
    self.cat_conv_2 = CatConv(128, 64, 64)
    self.cat_conv_1 = CatConv(64, 32, 32)

    self.out = nn.Conv2d(32, self.out_kernels, kernel_size=3, stride=1, padding=1, bias=True)
    self.softmax = nn.Softmax(dim=1)

    for m in self.modules():
      if isinstance(m, (nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', a=0.01, nonlinearity='leaky_relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def forward(self, deictic_obs, hand_obs, obs, action):
    batch_size = obs.size(0)
    hand_obs = self.padHandObs(hand_obs)

    inp = torch.cat((deictic_obs, hand_obs), dim=1)
    inp = self.conv_1(inp)
    inp = self.bn1(inp)
    inp = self.relu(inp)

    feat_down_1 = self.layer_1(inp)
    feat_down_2 = self.layer_2(feat_down_1)
    feat_down_3 = self.layer_3(feat_down_2)
    feat_down_4 = self.layer_4(feat_down_3)

    forward_feat = self.forward_layer(feat_down_4)

    feat_up_4 = self.cat_conv_4(self.up_proj_4(forward_feat), feat_down_3)
    feat_up_3 = self.cat_conv_3(self.up_proj_3(feat_up_4), feat_down_2)
    feat_up_2 = self.cat_conv_2(self.up_proj_2(feat_up_3), feat_down_1)
    feat_up_1 = self.cat_conv_1(self.up_proj_1(feat_up_2), inp)

    deictic_obs_ = self.out(feat_up_1)
    deictic_obs_ = self.softmax(deictic_obs_)
    obs_ = self.replaceObs(obs, deictic_obs_.detach(), action)

    return deictic_obs_, obs_

  def padHandObs(self, hand_obs):
    batch_size = hand_obs.size(0)

    pad_size = (batch_size, self.out_kernels, constants.DEICTIC_OBS_SIZE, constants.DEICTIC_OBS_SIZE)
    hand_obs_pad = torch.zeros(pad_size).float().to(self.device)
    hand_obs_pad[:,0] = 1.0

    c = round(constants.DEICTIC_OBS_SIZE / 2)
    s = round(hand_obs.size(-1) / 2)
    hand_obs_pad[:, :, c-s:c+s, c-s:c+s] = hand_obs

    return hand_obs_pad

  def replaceObs(self, obs, deictic_obs, actions):
    R = torch.zeros(actions.size(0), 2, 3)
    R[:,0,0] = torch.cos(actions[:,2])
    R[:,0,1] = -torch.sin(actions[:,2])
    R[:,1,0] = torch.sin(actions[:,2])
    R[:,1,1] = torch.cos(actions[:,2])

    grid_shape = (actions.size(0), 1, constants.DEICTIC_OBS_SIZE, constants.DEICTIC_OBS_SIZE)
    grid = F.affine_grid(R, grid_shape, align_corners=True).to(self.device)
    deictic_obs = F.grid_sample(deictic_obs, grid, padding_mode='zeros', align_corners=False, mode='bilinear')

    c = actions[:, :2]
    padding = [int(constants.DEICTIC_OBS_SIZE / 2)] * 4
    padded_obs = F.pad(obs, padding, 'constant', 0.0)
    c = c + int(constants.DEICTIC_OBS_SIZE / 2)

    b = padded_obs.size(0)
    x = torch.clamp(c[:,0].view(b,1) + torch.arange(-s, s).repeat(b, 1).to(self.device), 0, h-1).long()
    y = torch.clamp(c[:,1].view(b,1) + torch.arange(-s, s).repeat(b, 1).to(self.device), 0, h-1).long()

    ind = torch.transpose(((x * h).repeat(1,s*2).view(b,s*2,s*2) + y.view(b,s*2,1)), 1, 2)

    padded_obs = padded_obs.view(b*self.out_kernels,-1)
    padded_obs.scatter_(1, ind.reshape(b,-1).repeat(1, self.out_kernels).view(b*self.out_kernels,-1), deictic_obs.view(b*self.out_kernels, -1))
    padded_obs = padded_obs.view(b, self.out_kernels, h, h)

    start = int(constants.DEICTIC_OBS_SIZE / 2)
    end = obs.size(3) + int(constants.DEICTIC_OBS_SIZE / 2)
    new_obs = padded_obs[:, :, start:end, start:end]

    return new_obs

class RewardModel(nn.Module):
  def __init__(self, device):
    super(StateValueModel, self).__init__()
    self.device = device

    self.obs_feat = nn.Sequential(
      makeLayer(BasicBlock, 1, 32, 1, stride=2, bnorm=False),
      makeLayer(BasicBlock, 32, 64, 1, stride=2, bnorm=False),
      makeLayer(BasicBlock, 64, 128, 1, stride=2, bnorm=False),
      makeLayer(BasicBlock, 128, 256, 1, stride=2, bnorm=False)
    )

    self.hand_feat = nn.Sequential(
      makeLayer(BasicBlock, self.in_channels, 32, 1, stride=2, bnorm=False),
      makeLayer(BasicBlock, 32, 64, 1, stride=2, bnorm=False),
      makeLayer(BasicBlock, 64, 128, 1, stride=2, bnorm=False)
    )

    self.reward_head = nn.Sequential(
      nn.Conv2d(256+128, 256, 3, stride=2, padding=1),
      nn.LeakyReLU(0.01, inplace=True),
      Flatten(),
      nn.Linear(256*4*4, 2048),
      nn.LeakyReLU(0.01, inplace=True),
      nn.Linear(2048, 1028),
      nn.LeakyReLU(0.01, inplace=True),
      nn.Linear(1028, 256),
      nn.LeakyReLU(0.01, inplace=True),
      nn.Linear(256, 1),
    )

    for m in self.modules():
      if isinstance(m, (nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', a=0.01, nonlinearity='leaky_relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def forward(self, obs, hand_obs):
    batch_size = obs.size(0)

    if obs.size(1) > self.in_channels:
      obs = data_utils.convertProbToDepth(obs, obs.size(1))
      hand_obs = data_utils.convertProbToDepth(hand_obs, hand_obs.size(1))
    pad = int((constants.DEICTIC_OBS_SIZE - hand_obs.size(-1)) / 2)
    hand_obs = F.pad(hand_obs, [pad] * 4)

    obs_feat = self.obs_feat(obs)
    hand_feat = self.hand_feat(hand_obs)
    state_feat = torch.cat((obs_feat, hand_feat), dim=1)
    reward = self.reward_head(state_feat)

    return reward
