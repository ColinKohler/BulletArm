import os
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from data import constants
from data import data_utils

def generate2dGaussian(size, mu=0, sigma=1):
  x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
  dst = np.sqrt(x*x+y*y)
  gauss = torch.from_numpy(np.exp(-((dst-mu)**2 / (2.0 * sigma**2))))

  return gauss

def pixelwiseEntropy(pt):
  n = pt.size(0)
  entropy = (-pt * torch.log2(pt)).view(n, -1)
  return torch.sum(entropy, dim=1) / pt.size(-1) ** 2

def getDeicticActions(obs, actions, num_depth_classes=None):
  device = obs.device

  x_offset = (actions[:,0].float() - (constants.OBS_SIZE / 2.0)) / (constants.OBS_SIZE / 2.0)
  y_offset = (actions[:,1].float() - (constants.OBS_SIZE / 2.0)) / (constants.OBS_SIZE / 2.0)
  zoom = constants.DEICTIC_OBS_SIZE / constants.OBS_SIZE

  R = torch.eye(3, 3)
  R = R.repeat(actions.size(0), 1, 1)
  R[:,0,0] = torch.cos(-actions[:,2])
  R[:,0,1] = -torch.sin(-actions[:,2])
  R[:,1,0] = torch.sin(-actions[:,2])
  R[:,1,1] = torch.cos(-actions[:,2])

  S = torch.eye(3, 3)
  S = S.repeat(actions.size(0), 1, 1)
  S[:,0,0] = zoom
  S[:,1,1] = zoom

  T = torch.eye(3,3)
  T = T.repeat(actions.size(0), 1, 1)
  T[:,0,2] = x_offset
  T[:,1,2] = y_offset

  theta = torch.matmul(T, torch.matmul(S, R))

  grid_shape = (actions.size(0), 1, constants.DEICTIC_OBS_SIZE, constants.DEICTIC_OBS_SIZE)
  grid = F.affine_grid(theta[:,:2,:], grid_shape, align_corners=False).to(device)

  crops = F.grid_sample(obs, grid, padding_mode='zeros', mode='bilinear', align_corners=False)

  # Padding prob maps with zeroes is bad so we edit them to be one hots for the zero depth classs
  if obs.size(1) > 1:
    s, x, y = torch.where(torch.sum(crops, dim=1) == 0)
    crops[s,0,x,y] = 1.0
    crops[s,1:,x,y] = 0.0

  if num_depth_classes:
    crops = smoothDepthTorch(crops, num_depth_classes)
  return crops

def getHandObs(obs, num_depth_classes=None):
  device = obs.device

  zoom = constants.HAND_OBS_SIZE / constants.DEICTIC_OBS_SIZE
  theta = torch.Tensor([[zoom, 0.,    0.],
                        [0.,   zoom,  0.]]).view(1,2,3)
  theta = theta.repeat(obs.size(0), 1, 1)
  grid_shape = (obs.size(0), 1, constants.HAND_OBS_SIZE, constants.HAND_OBS_SIZE)
  grid = F.affine_grid(theta, grid_shape, align_corners=True).to(device)

  crops = F.grid_sample(obs, grid, padding_mode='zeros', align_corners=True, mode='nearest')

  # Padding prob maps with zeroes is bad so we edit them to be one hots for the zero depth classs
  if obs.size(1) > 1:
    s, x, y = torch.where(torch.sum(crops, dim=1) == 0)
    crops[s,0,x,y] = 1.0
    crops[s,1:,x,y] = 0.0

  if num_depth_classes:
    crops = smoothDepthTorch(crops, num_depth_classes)
  return crops

def rotateObs(obs, rots):
  device = obs.device

  R = torch.zeros(rots.size(0), 2, 3)
  R[:,0,0] = torch.cos(-rots)
  R[:,0,1] = -torch.sin(-rots)
  R[:,1,0] = torch.sin(-rots)
  R[:,1,1] = torch.cos(-rots)

  grid_shape = (rots.size(0), 1, constants.OBS_SIZE, constants.OBS_SIZE)
  grid = F.affine_grid(R, grid_shape, align_corners=True).to(device)
  rot_obs = F.grid_sample(obs, grid, padding_mode='zeros', align_corners=True, mode='nearest')

  return rot_obs

def getPixelAction(action, workspace, res, obs_size):
  pixel_action = np.array([action[0],
                           np.clip(round((action[2].item() - workspace[1,0]) / res), 0, obs_size-1),
                           np.clip(round((action[1].item() - workspace[0,0]) / res), 0, obs_size-1)])
  return pixel_action

def getPixelActions(state, action_res):
  return torch.stack(torch.meshgrid(torch.tensor(state),
                                    torch.arange(0, constants.OBS_SIZE, action_res, dtype=torch.float32),
                                    torch.arange(0, constants.OBS_SIZE, action_res, dtype=torch.float32)), -1).view(-1, 3)


def getWorkspaceAction(action, workspace, res, rotations):
  return torch.tensor([action[0],
                       (action[1] * res) + workspace[0,0],
                       (action[2] * res) + workspace[1,0],
                       rotations[int(action[3])]])

def cropObs(obs, center, rot, crop_size):
  adding = [int(crop_size/2)] * 4
  padded_obs = F.pad(obs, padding, 'constant', 0.0)
  center = center + int(crop_size / 2)

  x_min = int(center[0] - crop_size / 2)
  x_max = int(center[0] + crop_size / 2)
  y_min = int(center[1] - crop_size / 2)
  y_max = int(center[1] + crop_size / 2)
  crop = padded_obs[x_min:x_max, y_min:y_max]

  return crop

def replaceObs(obs, deictic_obs, center):
  padding = [int(constants.DEICTIC_OBS_SIZE/2)] * 4
  padded_obs = F.pad(obs, padding, 'constant', 0.0)
  center = center + int(constants.DEICTIC_OBS_SIZE / 2)

  x_min = (center[:,0] - constants.DEICTIC_OBS_SIZE / 2).int()
  x_max = (center[:,0] + constants.DEICTIC_OBS_SIZE / 2).int()
  y_min = (center[:,1] - constants.DEICTIC_OBS_SIZE / 2).int()
  y_max = (center[:,1] + constants.DEICTIC_OBS_SIZE / 2).int()

  # TODO: This loops makes things bad/slow
  for i in range(padded_obs.size(0)):
    padded_obs[i, :, x_min[i]:x_max[i], y_min[i]:y_max[i]] = deictic_obs[i]
  start = int(constants.DEICTIC_OBS_SIZE/2)
  end = obs.size(3) + int(constants.DEICTIC_OBS_SIZE/2)
  new_obs = padded_obs[:, :, start:end, start:end]

  return new_obs

def padTensor(tensor, max_len):
  if tensor.ndim == 2:
    padded_tensor = torch.zeros_like(tensor[0]).repeat(max_len, 1)
  elif tensor.ndim == 3:
    padded_tensor = torch.zeros_like(tensor[0]).repeat(max_len, 1, 1)
  elif tensor.ndim == 4:
    padded_tensor = torch.zeros_like(tensor[0]).repeat(max_len, 1, 1, 1)
  padded_tensor[:tensor.size(0)] = tensor

  return padded_tensor

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def smoothDepth(depth, num_labels):
  num_depth = depth.shape[0]
  depth_size = depth.shape[-1]
  bins = np.linspace(0, 1, num_labels)
  out = closestArgmin(depth.reshape(-1), bins)
  if depth.ndim > 2:
    return out.reshape(num_depth, 1, depth_size, depth_size)
  else:
    return out.reshape(depth_size, depth_size)

def closestArgmin(A, B):
  L = B.size
  sidx_B = B.argsort()
  sorted_B = B[sidx_B]
  sorted_idx = np.searchsorted(sorted_B, A)
  sorted_idx[sorted_idx==L] = L-1
  mask = (sorted_idx > 0) & \
         ((np.abs(A - sorted_B[sorted_idx-1]) < np.abs(A - sorted_B[sorted_idx])))
  return sorted_B[sorted_idx - mask]

def smoothDepthTorch(depth, num_labels):
  device = depth.device
  num_depth = depth.size(0)
  depth_size = depth.size(-1)
  bins = torch.linspace(0, 1, num_labels).to(device)
  out = closestArgminTorch(depth.contiguous().view(-1), bins)
  if depth.ndim > 2:
    return out.view(num_depth, 1, depth_size, depth_size)
  else:
    return out.view(depth_size, depth_size)

def closestArgminTorch(A, B):
  sidx_B = B.argsort()
  sorted_B = B[sidx_B]
  sorted_idx = torch.searchsorted(sorted_B, A)
  mask = (sorted_idx > 0) & \
         ((torch.abs(A - sorted_B[sorted_idx-1]) < torch.abs(A - sorted_B[sorted_idx])))

  return sorted_B[sorted_idx - mask.int()]

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def plotProbs(probs):
  probs = probs.squeeze()

  fig, ax = plt.subplots(nrows=4, ncols=3)
  fig.suptitle('Depth Classification Probability')

  depths = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']

  for i in range(4):
    for j in range(3):
      ind = (i * 3) + j
      if ind >= len(depths):
        fig.delaxes(ax[i][j])
        continue
      ax[i][j].set_title(depths[ind])
      ax[i][j].axis('off')
      im = ax[i][j].imshow(probs[ind].detach().cpu(), vmin=0, vmax=1)

  fig.subplots_adjust(right=0.8, hspace=0.5)
  cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
  fig.colorbar(im, cax=cbar_ax)

  plt.show()

def plotObs(obs, hand_obs, value=None):
  if obs.size(1) > 1:
    obs = data_utils.convertProbToDepth(obs, obs.size(1))
    hand_obs = data_utils.convertProbToDepth(hand_obs, hand_obs.size(1))

  fig, ax = plt.subplots(1,2)
  if value:
    fig.suptitle('Value: {:.3f}'.format(value))
  ax[0].imshow(obs.cpu().squeeze(), cmap='gray')
  ax[1].imshow(hand_obs.cpu().squeeze(), cmap='gray')
  plt.show()

def plot(tensors):
  fig, ax = plt.subplots(1,len(tensors))
  for i, t in enumerate(tensors):
    ax[i].imshow(data_utils.convertProbToDepth(t, t.size(1)).cpu().squeeze(), cmap='gray')
  plt.show()

def saveObs(obs, hand_obs, filename):
  if obs.size(1) > 1:
    obs = data_utils.convertProbToDepth(obs, obs.size(1))
    hand_obs = data_utils.convertProbToDepth(hand_obs, hand_obs.size(1))

  fig, ax = plt.subplots(1,2)
  ax[0].imshow(obs.cpu().squeeze(), cmap='gray', vmin=0.0, vmax=1.0)
  ax[1].imshow(hand_obs.cpu().squeeze(), cmap='gray', vmin=0.0, vmax=1.0)
  fig.savefig(filename)
  plt.close(fig)

def removeFiles(path):
  for root, dirs, files in os.walk(path):
    for file in files:
      os.remove(os.path.join(root, file))
