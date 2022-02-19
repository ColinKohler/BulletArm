import sys
sys.path.append('..')

import os
import numpy as np
import numpy.random as npr
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import functools

from models.state_prediction_model import StatePredictionModel
from data import data_utils
import utils
import models.torch_utils as torch_utils
from models.losses import FocalLoss

class RandomShootingAgent(object):
  def __init__(self, config, device, training=False):
    self.device = device
    self.config = config
    self.training = training

    self.workspace = constants.WORKSPACE
    self.obs_res = constants.OBS_RESOLUTION
    self.obs_size = constants.OBS_SIZE
    self.deictic_obs_size = constants.DEICTIC_OBS_SIZE
    self.hand_obs_size = constants.HAND_OBS_SIZE
    self.rotations = torch.from_numpy(np.linspace(0, np.pi, self.config.num_rots, endpoint=False))
    self.action_sample_pen_size = self.config.init_action_sample_pen_size

    self.preprocessDepth = functools.partial(data_utils.preprocessDepth,
                                             min=0.,
                                             max=self.config.max_height,
                                             num_classes=self.config.num_depth_classes,
                                             round=2,
                                             noise=False)

    self.unnormalizeDepth = functools.partial(data_utils.unnormalizeData,
                                              min=0,
                                              max=self.config.max_height)

    self.forward_model = StatePredictionModel(self.device, self.config.num_depth_classes).to(self.device)

    if self.training:
      self.forward_optimizer = torch.optim.Adam(self.forward_model.parameters(),
                                                lr=self.config.forward_lr_init,
                                                weight_decay=self.config.forward_weight_decay,
                                                betas=(0.9, 0.999))
      self.forward_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.forward_optimizer,
                                                                      self.config.lr_decay)
      self.focal_loss = FocalLoss(self.device,
                                  alpha=torch.ones(self.config.num_depth_classes),
                                  gamma=0.0,
                                  smooth=1e-5,
                                  size_average=True)
      self.forward_model.train()
    else:
      self.forward_model.eval()

  def selectAction(self, obs, normalize_obs=True):
    state, hand_obs, obs = obs

    obs = torch.Tensor(obs.astype(np.float32)).view(1, 1, self.config.obs_size, self.config.obs_size)
    hand_obs = torch.Tensor(hand_obs.astype(np.float32)).view(1, 1, self.config.hand_obs_size, self.config.hand_obs_size)
    if normalize_obs:
      obs = data_utils.convertDepthToOneHot(self.preprocessDepth(obs),
                                            self.config.num_depth_classes)
      hand_obs = data_utils.convertDepthToOneHot(self.preprocessDepth(hand_obs),
                                                 self.config.num_depth_classes)
    else:
      obs = data_utils.convertDepthToOneHot(obs, self.config.num_depth_classes)
      hand_obs = data_utils.convertDepthToOneHot(hand_obs, self.config.num_depth_classes)

    action, pred_obs = self.getRandomShootingAction((state, hand_obs, obs))

    return torch.zeros((1, 1, 128, 128)), action, pred_obs[2]

  def getRandomShootingAction(self, obs):
    state = torch.tensor([obs[0]]).repeat(self.config.num_sampled_actions)
    hand_obs = obs[1].repeat(self.config.num_sampled_actions, 1, 1, 1)
    obs = obs[2].repeat(self.config.num_sampled_actions, 1, 1, 1)
    reward = torch.tensor([0]).repeat(self.config.num_sampled_actions)

    traj_actions = list()
    traj = [[state, hand_obs, obs, reward]]
    for d in range(self.config.depth):
      q_map, actions = self.sampleActionsAroundObjects(state, obs)
      state, obs, hand_obs, reward = self.getNextState(state, obs, hand_obs, actions)

      traj_actions.append(actions.cpu())
      traj.append([state.cpu(), hand_obs.cpu(), obs.cpu(), reward.squeeze().cpu()])

    best_traj = torch.argmax(traj[:,3])
    best_obs = [traj[1][0][best_traj], traj[1][1][best_traj], traj[1][2][best_traj]]
    return traj_actions[0][best_traj].cpu(), best_obs

  def getNextState(self, state, obs, hand_obs, pixel_actions):
    rotations = self.rotations.to(self.device)
    deictic_pixel_actions = torch.stack([pixel_actions[:,2],
                                         pixel_actions[:,1],
                                         rotations[pixel_actions[:,3].long()]])
    deictic_pixel_actions = deictic_pixel_actions.permute(1,0)

    pixel_actions_w_rot = torch.stack([pixel_actions[:,1],
                                       pixel_actions[:,2],
                                       rotations[pixel_actions[:,3].long()]])
    pixel_actions_w_rot = pixel_actions_w_rot.permute(1,0)

    deictic_obs = utils.getDeicticActions(obs, deictic_pixel_actions)

    with torch.no_grad():
      deictic_obs_, obs_ = self.forward_model(deictic_obs.to(self.device),
                                              hand_obs.to(self.device),
                                              obs.to(self.device),
                                              pixel_actions_w_rot.to(self.device))
    deictic_obs_ = deictic_obs_[torch.arange(deictic_obs_.size(0)), state.long()]
    obs_ = obs_[torch.arange(obs_.size(0)), state.long()]

    hand_obs_ = utils.getHandObs(deictic_obs).cpu()
    state_ = self.getHandStates(deictic_obs.cpu(), deictic_obs_.cpu())

    if torch.sum((~state_).long()) > 0:
      num_hand_obs_empty = torch.sum((~state_).long()).item()
      empty_hand_obs = torch.zeros(num_hand_obs_empty, 1, self.hand_obs_size, self.hand_obs_size)
      empty_idx = torch.where(state_.int() == 0)
      hand_obs_[empty_idx] = data_utils.convertDepthToOneHot(empty_hand_obs,
                                                             self.config.num_depth_classes)

    return state_, obs_, hand_obs_

  def sampleActions(self, state):
    x = torch.randint(self.config.obs_size, (self.config.num_sampled_actions,))
    y = torch.randint(self.config.obs_size, (self.config.num_sampled_actions,))
    r = torch.randint(self.config.num_rots, (self.config.num_sampled_actions,))

    action = torch.vstack((state, x, y, r))
    return action.permute(1,0)

  def sampleActionsFromObjects(self, state, obs):
    if obs.size(1) == self.config.num_depth_classes:
      obs = data_utils.convertProbToDepth(obs, self.config.num_depth_classes)

    object_masks = (obs > 0).float() + 1e-20
    pixel_actions = torch_utils.sample2d(object_masks, k=1)
    rots = torch.randint(self.config.num_rots, (n, 1))
    pixel_actions = torch.cat((state.view(n, 1), pixel_actions, rots))

    return object_masks, torch.tensor(pixel_actions).squeeze()

  def sampleActionsAroundObjects(self, state, obs, p=15):
    n = obs.size(0)

    if obs.size(1) == self.config.num_depth_classes:
      obs = data_utils.convertProbToDepth(obs, self.config.num_depth_classes)

    # Pad area around objects using kernel and then take a mask of this area to sample actions from
    kernel = torch.ones((n, 1, p, p)).to(self.device)
    obs = F.conv2d(obs.permute(1,0,2,3).to(self.device), kernel, padding='same', groups=n).permute(1,0,2,3)
    object_masks = (obs > 0).float() + 1e-20

    pixel_actions = torch_utils.sample2d(object_masks, k=1)
    rots = torch.randint(self.config.num_rots, (n, 1))
    pixel_actions = torch.cat((state.view(n, 1).to(self.device), pixel_actions, rots.to(self.device)), dim=1)
    return object_masks, pixel_actions

  def updateWeights(self, batch, class_weight):
    # Check that training mode was enabled at init
    if not self.training:
      return None

    # Process batch and load onto device
    state_batch, hand_obs_batch, obs_batch, action_batch, target_state_value, target_q_value, target_reward, is_expert, weight_batch, = batch
    td_error = np.zeros((self.config.batch_size, obs_batch.size(1)-1))

    state_batch = state_batch.to(self.device)
    hand_obs_batch = hand_obs_batch.to(self.device)
    obs_batch = obs_batch.to(self.device)
    action_batch = action_batch.to(self.device)
    target_reward = target_reward.to(self.device)
    weight_batch = weight_batch.to(self.device)
    rotations = self.rotations.to(self.device)

    obs = obs_batch[:,0]
    forward_loss, reward_loss = 0, 0
    for t in range(1, action_batch.size(1)):
      # Reward prediction
      _, reward = self.reward_model(obs, hand_obs_batch[:,t-1])
      reward_loss += F.smooth_l1_loss(reward.squeeze(), target_reward[:,t-1].squeeze(), reduction='none')

      # Dynamics prediction
      action = torch.stack([action_batch[:,t,0],
                            action_batch[:,t,2],
                            action_batch[:,t,1],
                            rotations[action_batch[:,t,3].long()]]).permute(1,0)

      deictic_obs = utils.getDeicticActions(obs, action[:,1:])
      deictic_obs_, obs, = self.forward_model(deictic_obs,
                                              hand_obs_batch[:,t-1],
                                              obs,
                                              action[:,1:])
      deictic_obs_ = deictic_obs_[torch.arange(self.config.batch_size), state_batch[:,t-1]]
      obs = obs[torch.arange(self.config.batch_size), state_batch[:,t-1]]

      deictic_target = utils.getDeicticActions(obs_batch[:,t], action[:,1:])
      deictic_label = data_utils.convertProbToLabel(deictic_target, self.config.num_depth_classes)

      f_loss = self.focal_loss(deictic_obs_, deictic_label, alpha=class_weight.to(self.device))
      forward_loss += f_loss

      td_error[:,t-1] = f_loss.detach().cpu().numpy()

    # Compute the weighted losses
    forward_loss = (forward_loss * weight_batch).mean()
    reward_loss = (reward_loss * weight_batch).mean()

    # Optimize
    self.forward_optimizer.zero_grad()
    forward_loss.backward()
    self.forward_optimizer.step()

    self.reward_optimizer.zero_grad()
    reward_loss.backward()
    self.reward_optimizer.step()

    return (td_error,
            forward_loss.item(),
            reward_loss.item())

  def updateLR(self):
    self.forward_scheduler.step()
    self.reward_scheduler.step()

  def getLR(self):
    return (self.forward_optimizer.param_groups[0]['lr'],
            self.reward_optimizer.param_groups[0]['lr'])

  def getWeights(self):
    return (torch_utils.dictToCpu(self.forward_model.state_dict()),
            torch_utils.dictToCpu(self.reward_model.state_dict()))

  def setWeights(self, weights):
    if weights is not None:
      self.forward_model.load_state_dict(weights[0])
      self.reward_model.load_state_dict(weights[1])

  def getOptimizerState(self):
    return (torch_utils.dictToCpu(self.forward_optimizer.state_dict()),
            torch_utils.dictToCpu(self.reward_optimizer.state_dict()))

  def setOptimizerState(self, state):
    self.forward_optimizer.load_state_dict(copy.deepcopy(state[0]))
    self.reward_optimizer.load_state_dict(copy.deepcopy(state[1]))
