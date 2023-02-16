import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import escnn.nn as enn
import numpy as np
import numpy.random as npr
import scipy.ndimage

def detachGeoTensor(geo, t):
  return enn.GeometricTensor(geo.tensor.detach(), t)

def dictToCpu(state_dict):
  cpu_dict = dict()
  for k, v in state_dict.items():
    if isinstance(v, torch.Tensor):
      cpu_dict[k] = v.cpu()
    elif isinstance(v, dict):
      cpu_dict[k] = dictToCpu(v)
    elif isinstance(v, enn.EquivariantModule):
      cpu_dict[k] = v.cpu()
    else:
      cpu_dict[k] = v

  return cpu_dict

def clipGradNorm(optimizer, max_norm=None, norm_type=2):
  for param_group in optimizer.param_groups:
    max_norm_x = max_norm
    if max_norm_x is None and 'n_params' in param_group:
      max_norm_x = 1e-1 * np.sqrt(param_group['n_params'])
    if max_norm_x is not None:
      nn.utils.clip_grad.clip_grad_norm_(param_groups['params'],
                                         max_norm=max_norm_x,
                                         norm_type=norm_type)

def normalizeDepth(depth):
  depth = np.clip(depth, 0, 0.32)
  depth = depth / 0.4 * 255
  depth = depth.astype(np.uint8)

  return depth

def unnormalizeDepth(depth):
  return depth / 255 * 0.4

def normalizeForce(force, max_force):
  return np.clip(force, -max_force, max_force) / max_force

def sampleGaussian(mu, var):
  eps = Normal(0, 1).sample(mu.size())
  z = mu + torch.sqrt(var) * eps.cuda()

  return z

def gaussianParameters(h, dim=-1):
  mu, h = torch.split(h, h.size(dim) // 2, dim=dim)
  var = F.softplus(h) + 1e-8

  return mu, var

def productOfExperts(mu_vect, var_vect):
  t_vect = 1.0 / var_vect
  mu = (mu_vect * t_vect).sum(2) * (1 / t_vect.sum(2))
  var = 1 / t_vect.sum(2)

  return mu, var

def duplicate(x, rep):
  return x.expand(rep, *x.shape).reshape(-1, *x.shape[1:])

def klNormal(qm, qv, pm, pv):
  element_wise = 0.5 * (
    torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1
  )
  kl = element_wise.sum(-1)

  return kl

def perturb(depth, depth_, set_theta_zero=False, set_trans_zero=False):
  depth_size = depth.shape[-2:]

  # Compute random rigid transform
  theta, trans, pivot = getRandomImageTransformParams(depth_size)
  if set_theta_zero:
    theta = 0.
  if set_trans_zero:
    trans = [0., 0.]
  transform = getImageTransform(theta, trans, pivot)
  transform_params = theta, trans, pivot

  # Apply rigid transform to depth
  depth = scipy.ndimage.affine_transform(depth, np.linalg.inv(transform), mode='nearest', order=1)
  depth_ = scipy.ndimage.affine_transform(depth_, np.linalg.inv(transform), mode='nearest', order=1)

  return depth, depth_, transform_params

def getRandomImageTransformParams(depth_size):
  ''''''
  theta = npr.rand() * 2 * np.pi
  trans = npr.randint(0, depth_size[0] // 10, 2) - depth_size[0] // 20
  pivot = (depth_size[1] / 2, depth_size[0] / 2)

  return theta, trans, pivot

def getImageTransform(theta, trans, pivot=(0,0)):
  ''''''
  pivot_t_image = np.array([[1., 0., -pivot[0]],
                            [0., 1., -pivot[1]],
                            [0., 0., 1.]])
  image_t_pivot = np.array([[1., 0., pivot[0]],
                            [0., 1., pivot[1]],
                            [0., 0., 1.]])
  transform = np.array([[np.cos(theta), -np.sin(theta), trans[0]],
                        [np.sin(theta), np.cos(theta),  trans[1]],
                        [0., 0., 1.]])
  return np.dot(image_t_pivot, np.dot(transform, pivot_t_image))

def randomCrop(imgs, out=64):
  n, c, h, w = imgs.shape
  crop_max = h - out + 1
  w1 = np.random.randint(0, crop_max, n)
  h1 = np.random.randint(0, crop_max, n)
  cropped = np.empty((n, c, out, out), dtype=imgs.dtype)
  for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
    cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
  return cropped

def centerCrop(imgs, out=64):
  n, c, h, w = imgs.shape
  top = (h - out) // 2
  left = (w - out) // 2

  imgs = imgs[:, :, top:top + out, left:left + out]
  return imgs

def rotateVector(vec, theta):
  rot = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)],
  ])

  rot_vec = (rot @ vec).T
  return rot_vec
