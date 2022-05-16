import torch
import numpy as np
import numpy.random as npr

from models import torch_utils

from data import constants
import utils

from data.configs.block_stacking_3 import BlockStacking3Config
from data.configs.house_building_2 import HouseBuilding2Config
from data.configs.bottle_tray import BottleTrayConfig
from data.configs.bin_packing import BinPackingConfig

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

task_config_dict = {
  'block_stacking_3' : BlockStacking3Config,
  'house_building_2' : HouseBuilding2Config,
  'bottle_tray' : BottleTrayConfig,
  'bin_packing' : BinPackingConfig,
}

def getTaskConfig(task, num_gpus, results_path=None):
  try:
    config = task_config_dict[task](num_gpus, results_path=results_path)
  except:
    raise ValueError('Invalid task specified')

  return config

def getPlannerConfig(pick_noise, place_noise, rand_action_prob, random_orientation, planner_type=None):
  config = {
    'pick_noise': pick_noise,
    'place_noise': place_noise,
    'rand_place_prob': rand_action_prob,
    'rand_pick_prob': rand_action_prob,
    'random_orientation': random_orientation
  }

  if planner_type:
    config['planner_type'] = planner_type

  return config

def getEnvConfig(env_type, use_rot, use_planner_noise=False, render=False):
  if env_type in constants.ENV_CONFIGS:
    env_config = constants.ENV_CONFIGS[env_type]
    env_type = constants.ENV_TYPES[env_type]
  else:
    raise ValueError('Invalid env type specified')

  env_config['render'] = render
  env_config['random_orientation'] = use_rot

  if use_planner_noise:
    planner_config = getPlannerConfig([0, 0.01], [0, 0.01], 0., use_rot)
  else:
    planner_config = getPlannerConfig([0, 0], [0, 0.], 0, use_rot)

  return env_type, env_config, planner_config

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def preprocessDepth(depth, min, max, num_classes, round=2, noise=False):
  depth = normalizeData(depth, min, max)
  if type(depth) is np.ndarray:
    depth = np.round(depth, round)
    depth = utils.smoothDepth(depth, num_classes)
    if noise:
      depth += npr.rand(*depth.shape)
  elif type(depth) is torch.Tensor:
    depth = torch_utils.roundTensor(depth, round)
    depth = utils.smoothDepthTorch(depth, num_classes)
    if noise:
      depth += torch.rand_like(depth) * 0.01
  else:
    ValueError('Data not numpy array or torch tensor')

  return depth

def normalizeData(data, min, max, eps=1e-8):
  return (data - min) / (max - min + eps)

def unnormalizeData(data, min, max, eps=1e-8):
  return data * (max - min + eps) + min

def convertDepthToOneHot(depth, num_labels):
  if depth.ndim == 2 or depth.ndim == 3:
    num_depth = 1
  elif depth.ndim == 4:
    num_depth = depth.shape[0]
  depth_size = depth.shape[-1]

  inds = torch.from_numpy(convertDepthToLabel(depth, num_labels))
  inds = inds.view(num_depth, 1, depth_size, depth_size).long()

  x = torch.FloatTensor(num_depth, num_labels, depth_size, depth_size)
  x.zero_()
  x.scatter_(1, inds, 1)

  return x

def convertDepthToLabel(depth, num_labels):
  bins = np.linspace(0., 1., num_labels, dtype=np.float32)
  inds = np.digitize(depth, bins, right=True)

  return inds

def convertProbToLabel(prob, num_labels):
  return torch.argmax(prob, dim=1).squeeze()

def convertProbToDepth(prob, num_depth_classes):
  if prob.dim() == 3:
    n = 1
    prob_one_hot = prob.argmax(0).float()
  elif prob.dim() == 4:
    n = prob.size(0)
    prob_one_hot = prob.argmax(1).float()
  depth = prob_one_hot * (1 / (num_depth_classes - 1))

  s = prob.size(-1)
  return depth.view(n, 1, s, s)

def normalizeProb(prob):
  if prob.dim() == 3:
    n = 1
    d = prob.size(0)
    s = prob.size(1)
  elif prob.dim() == 4:
    n = prob.size(0)
    d = prob.size(1)
    s = prob.size(2)

  prob = prob.view(n, d, -1)
  max_prob = torch.sum(prob, dim=1)
  return (prob / max_prob).view(n, d, s, s)
