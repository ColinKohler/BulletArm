import torch
import numpy as np
import numpy.random as npr

from bulletarm_baselines.vtt.vtt import torch_utils
from bulletarm_baselines.vtt.vtt.models.sac import GaussianPolicy, TwinnedQNetwork
from bulletarm_baselines.vtt.vtt.models.latent import LatentModel

class Agent(object):
  '''
  Soft Actor-Critic Agent.

  Args:
    config (dict): Task config.
    device (torch.Device): Device to use for inference (cpu or gpu)
  '''
  def __init__(self, config, device, latent=None, actor=None, critic=None, initialize_models=True):
    self.config = config
    self.device = device

    self.p_range = torch.tensor([0, 1])
    self.dx_range = torch.tensor([-self.config.dpos, self.config.dpos])
    self.dy_range = torch.tensor([-self.config.dpos, self.config.dpos])
    self.dz_range = torch.tensor([-self.config.dpos, self.config.dpos])
    self.dtheta_range = torch.tensor([-self.config.drot, self.config.drot])
    self.action_shape = 5

    if actor:
      self.actor = actor
    else:
      # self.actor = GaussianPolicy()
      self.actor = GaussianPolicy([5], 8, 36)
      self.actor.to(self.device)
      self.actor.train()

    if critic:
      self.critic = critic
    else:
      # self.critic = TwinnedQNetwork()
      self.critic = TwinnedQNetwork([5], 2, 2)
      self.critic.to(self.device)
      self.critic.train()

    if latent:
      self.latent = latent
    else:
      # self.latent = LatentModel()
      self.latent = LatentModel([1, 288], [5])
      self.latent.to(self.device)
      self.latent.train()

  def getAction(self, vision, force, proprio, evaluate=False):
    '''
    Get the action from the policy.

    Args:
      evalute (bool):

    Returns:
      (numpy.array, double) : (Action, Q-Value)
    '''
    vision = torch.Tensor(vision.astype(np.float32)).view(vision.shape[0], vision.shape[1], vision.shape[2], vision.shape[3]).to(self.device)
    vision = torch_utils.centerCrop(vision, out=self.config.vision_size)
    force = torch.Tensor(torch_utils.normalizeForce(force, self.config.max_force)).view(vision.shape[0], self.config.force_history, self.config.force_dim).to(self.device)
    proprio = torch.Tensor(proprio).view(vision.shape[0], self.config.proprio_dim).to(self.device)

    with torch.no_grad():
      z, _, _ = self.latent.encoder(vision, force)
      if evaluate:
        _, _, action = self.actor.sample(z)
      else:
        action, _, _ = self.actor.sample(z)

    action = action.cpu()
    action_idx, action = self.decodeActions(*[action[:,i] for i in range(self.action_shape)])
    with torch.no_grad():
      value = self.critic((vision, force, proprio), action_idx.to(self.device))

    value = torch.min(torch.hstack((value[0], value[1])), dim=1)[0]
    return action_idx, action, value

  def decodeActions(self, unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz, unscaled_dtheta):
    '''
    Convert action from model to environment action.

    Args:
      unscaled_p (double):
      unscaled_dx (double):
      unscaled_dy (double):
      unscaled_dz (double):
      unscaled_dtheta (double):

    Returns:
      (torch.Tensor, torch.Tensor) : Unscaled actions, scaled actions
    '''
    p = 0.5 * (unscaled_p + 1) * (self.p_range[1] - self.p_range[0]) + self.p_range[0]
    dx = 0.5 * (unscaled_dx + 1) * (self.dx_range[1] - self.dx_range[0]) + self.dx_range[0]
    dy = 0.5 * (unscaled_dy + 1) * (self.dy_range[1] - self.dy_range[0]) + self.dy_range[0]
    dz = 0.5 * (unscaled_dz + 1) * (self.dz_range[1] - self.dz_range[0]) + self.dz_range[0]

    dtheta = 0.5 * (unscaled_dtheta + 1) * (self.dtheta_range[1] - self.dtheta_range[0]) + self.dtheta_range[0]
    actions = torch.stack([p, dx, dy, dz, dtheta], dim=1)
    unscaled_actions = torch.stack([unscaled_p, unscaled_dx, unscaled_dy, unscaled_dz, unscaled_dtheta], dim=1)

    return unscaled_actions, actions

  def convertPlanAction(self, plan_action):
    '''
    Convert actions from planner to agent actions by unscalling/scalling them.

    Args:
      plan_action (numpy.array): Action received from planner

    Returns:
      (torch.Tensor, torch.Tensor) : Unscaled actions, scaled actions
    '''
    p = plan_action[:, 0].clamp(*self.p_range)
    dx = plan_action[:, 1].clamp(*self.dx_range)
    dy = plan_action[:, 2].clamp(*self.dy_range)
    dz = plan_action[:, 3].clamp(*self.dz_range)
    dtheta = plan_action[:, 4].clamp(*self.dtheta_range)

    return self.decodeActions(
      self.getUnscaledActions(p, self.p_range),
      self.getUnscaledActions(dx, self.dx_range),
      self.getUnscaledActions(dy, self.dy_range),
      self.getUnscaledActions(dz, self.dz_range),
      self.getUnscaledActions(dtheta, self.dtheta_range)
    )

  def getUnscaledActions(self, action, action_range):
    '''
    Convert action to the unscalled version using the given range.

    Args:
      action (double): Action
      action_range (list[double]): Min and max range for the given action

    Returns:
      double: The unscalled action
    '''
    return 2 * (action - action_range[0]) / (action_range[1] - action_range[0]) - 1

  def getWeights(self):
    '''
    '''
    return (self.latent.state_dict(),
            self.actor.state_dict(),
            self.critic.state_dict())

  def setWeights(self, weights):
    '''
    Load given weights into the actor and critic

    Args:
      weights (dict, dict): (actor weights, critic weights)
    '''
    if weights is not None:
      sefllatent.load_state_dict(weights[0])
      self.actor.load_state_dict(weights[1])
      self.critic.load_state_dict(weights[2])
