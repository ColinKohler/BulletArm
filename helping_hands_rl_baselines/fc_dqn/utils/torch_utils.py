import torch
import torch.nn as nn
import math
import cv2
import numpy as np
import collections
from tqdm import tqdm
from collections import OrderedDict

ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert')

def featureExtractor():
  '''Creates a CNN module used for feature extraction'''
  return nn.Sequential(OrderedDict([
    ('conv0', nn.Conv2d(1, 16, kernel_size=7)),
    ('relu0', nn.ReLU(True)),
    ('pool0', nn.MaxPool2d(2)),
    ('conv1', nn.Conv2d(16, 32, kernel_size=7)),
    ('relu1', nn.ReLU(True)),
    ('pool1', nn.MaxPool2d(2)),
    ('conv2', nn.Conv2d(32, 64, kernel_size=5)),
    ('relu2', nn.ReLU(True)),
    ('pool2', nn.MaxPool2d(2))
  ]))

# def rotate(tensor, rad):
#   """
#   rotate the input tensor with the given rad
#   Args:
#     tensor: 1 x d x d image tensor
#     rad: degree in rad
#
#   Returns: 1 x d x d image tensor after rotation
#
#   """
#   img = transforms.ToPILImage()(tensor)
#   angle = 180./np.pi * rad
#   img = TF.rotate(img, angle)
#   return transforms.ToTensor()(img)

class TransformationMatrix(nn.Module):
  def __init__(self):
    super(TransformationMatrix, self).__init__()

    self.scale = torch.eye(3,3)
    self.rotation = torch.eye(3,3)
    self.translation = torch.eye(3,3)

  def forward(self, scale, rotation, translation):
    scale_matrix = self.scale.repeat(scale.size(0), 1, 1)
    rotation_matrix = self.rotation.repeat(rotation.size(0), 1, 1)
    translation_matrix = self.translation.repeat(translation.size(0), 1, 1)

    scale_matrix[:,0,0] = scale[:,0]
    scale_matrix[:,1,1] = scale[:,1]

    rotation_matrix[:,0,0] = torch.cos(rotation)
    rotation_matrix[:,0,1] = -torch.sin(rotation)
    rotation_matrix[:,1,0] = torch.sin(rotation)
    rotation_matrix[:,1,1] = torch.cos(rotation)

    translation_matrix[:,0,2] = translation[:,0]
    translation_matrix[:,1,2] = translation[:,1]

    return torch.bmm(translation_matrix, torch.bmm(rotation_matrix, scale_matrix))

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.scale = self.scale.to(*args, **kwargs)
    self.rotation = self.rotation.to(*args, **kwargs)
    self.translation = self.translation.to(*args, **kwargs)
    return self

class WeightedHuberLoss(nn.Module):
  ''' Compute weighted Huber loss for use with Pioritized Expereince Replay '''
  def __init__(self):
	  super(WeightedHuberLoss, self).__init__()

  def forward(self, input, target, weights, mask):
    batch_size = input.size(0)
    batch_loss = (torch.abs(input - target) < 1).float() * (input - target)**2 + \
                 (torch.abs(input - target) >= 1).float() * (torch.abs(input - target) - 0.5)
    batch_loss *= mask
    weighted_batch_loss = weights * batch_loss.view(batch_size, -1).sum(dim=1)
    weighted_loss = weighted_batch_loss.sum() / batch_size

    return weighted_loss

def clip(tensor, min, max):
  '''
  Clip the given tensor to the min and max values given

  Args:
    - tensor: PyTorch tensor to clip
    - min: List of min values to clip to
    - max: List of max values to clip to

  Returns: PyTorch tensor like given tensor clipped to bounds
  '''
  clipped_tensor = torch.zeros_like(tensor)
  for i in range(len(min)):
    clipped_tensor[:,i] = torch.max(torch.min(tensor[:,i], torch.tensor(max[i])), torch.tensor(min[i]))
  return clipped_tensor

def argmax2d(tensor):
  '''
  Find the index of the maximum value in a 2d tensor.

  Args:
    - tensor: PyTorch tensor of size (n x 1 x d x d)

  Returns: nx2 PyTorch tensor containing indexes of max values
  '''
  n = tensor.size(0)
  d = tensor.size(2)
  m = tensor.view(n, -1).argmax(1)
  return torch.cat(((m / d).view(-1, 1), (m % d).view(-1, 1)), dim=1)

def argmax3d(tensor):
  n = tensor.size(0)
  c = tensor.size(1)
  d = tensor.size(2)
  m = tensor.contiguous().view(n, -1).argmax(1)
  return torch.cat(((m/(d*d)).view(-1, 1), ((m%(d*d))/d).view(-1, 1), ((m%(d*d))%d).view(-1, 1)), dim=1)

def argmax4d(tensor):
  n = tensor.size(0)
  c1 = tensor.size(1)
  c2 = tensor.size(2)
  d = tensor.size(3)
  m = tensor.view(n, -1).argmax(1)

  d0 = (m/(d*d*c2)).view(-1, 1)
  d1 = ((m%(d*d*c2))/(d*d)).view(-1, 1)
  d2 = (((m%(d*d*c2))%(d*d))/d).view(-1, 1)
  d3 = (((m%(d*d*c2))%(d*d))%d).view(-1, 1)

  return torch.cat((d0, d1, d2, d3), dim=1)

def softUpdate(target_net, source_net, tau):
  '''
  Move target  net to source net a small amount

  Args:
    - target_net: net to copy weights into
    - source_net: net to copy weights from
    - tau: Amount to update weights
  '''
  for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
    target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

def hardUpdate(target_net, source_net):
  '''
  Copy all weights from source net to target net

  Args:
    - target_net: net to copy weights into
    - source_net: net to copy weights from
  '''
  target_net.load_state_dict(source_net.state_dict())

def rand_perlin_2d(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3, device='cuda'):
  delta = (res[0] / shape[0], res[1] / shape[1])
  d = (shape[0] // res[0], shape[1] // res[1])

  grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim=-1).to(device) % 1
  angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1).to(device)
  gradients = torch.stack((torch.cos(angles).to(device), torch.sin(angles).to(device)), dim=-1)

  tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0],
                                                                                                            0).repeat_interleave(
    d[1], 1)
  dot = lambda grad, shift: (
            torch.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                        dim=-1) * grad[:shape[0], :shape[1]]).sum(dim=-1)

  n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
  n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
  n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
  n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
  t = fade(grid[:shape[0], :shape[1]])
  return (math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])).cpu()


def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5):
  noise = torch.zeros(shape)
  frequency = 1
  amplitude = 1
  for _ in range(octaves):
    noise += amplitude * rand_perlin_2d(shape, (frequency * res[0], frequency * res[1]))
    frequency *= 2
    amplitude *= persistence
  return noise

def bbox(img, threshold=0.011):
  rows = np.any(img>threshold, axis=1)
  cols = np.any(img>threshold, axis=0)
  rmin, rmax = np.where(rows)[0][[0, -1]]
  cmin, cmax = np.where(cols)[0][[0, -1]]

  return rmin, rmax, cmin, cmax

def get_image_transform(theta, trans, pivot=(0, 0)):
  """Compute composite 2D rigid transformation matrix."""
  # Get 2D rigid transformation matrix that rotates an image by theta (in
  # radians) around pivot (in pixels) and translates by trans vector (in
  # pixels)
  pivot_t_image = np.array([[1., 0., -pivot[0]], [0., 1., -pivot[1]],
                            [0., 0., 1.]])
  image_t_pivot = np.array([[1., 0., pivot[0]], [0., 1., pivot[1]],
                            [0., 0., 1.]])
  transform = np.array([[np.cos(theta), -np.sin(theta), trans[0]],
                        [np.sin(theta), np.cos(theta), trans[1]], [0., 0., 1.]])
  return np.dot(image_t_pivot, np.dot(transform, pivot_t_image))


def get_random_image_transform_params(image_size, theta_dis_n=32):
  theta_sigma = 2 * np.pi / 6
  # theta = np.random.normal(0, theta_sigma)
  theta = np.random.choice(np.linspace(0, 2*np.pi, theta_dis_n, False))

  trans_sigma = np.min(image_size) / 20
  trans = np.random.normal(0, trans_sigma, size=2)  # [x, y]
  pivot = (image_size[1] / 2, image_size[0] / 2)
  return theta, trans, pivot

# Code for this function modified from https://github.com/google-research/ravens
def perturb(current_image, next_image, pixels, set_theta_zero=False, set_trans_zero=False, theta_dis_n=32):
  image_size = current_image.shape[:2]

  bbox_current = bbox(current_image)

  if np.any(np.array([bbox_current[1]-bbox_current[0], bbox_current[3]-bbox_current[2]]) > image_size[0] - 10):
    set_theta_zero = True

  pixels.extend([(bbox_current[0], bbox_current[2]),
                 (bbox_current[0], bbox_current[3]),
                 (bbox_current[1], bbox_current[2]),
                 (bbox_current[1], bbox_current[3]),
                 ])

  # Compute random rigid transform.
  while True:
    theta, trans, pivot = get_random_image_transform_params(image_size, theta_dis_n)
    if set_theta_zero:
      theta = 0.
    if set_trans_zero:
      trans = [0., 0.]
    transform = get_image_transform(theta, trans, pivot)
    transform_params = theta, trans, pivot

    # Ensure pixels remain in the image after transform.
    is_valid = True
    new_pixels = []
    new_rounded_pixels = []
    for pixel in pixels:
      pixel = np.float32([pixel[1], pixel[0], 1.]).reshape(3, 1)

      rounded_pixel = np.int32(np.round(transform @ pixel))[:2].squeeze()
      rounded_pixel = np.flip(rounded_pixel)

      pixel = (transform @ pixel)[:2].squeeze()
      pixel = np.flip(pixel)

      in_fov_rounded = rounded_pixel[0] < image_size[0] and rounded_pixel[
        1] < image_size[1]
      in_fov = pixel[0] < image_size[0] and pixel[1] < image_size[1]

      is_valid = is_valid and np.all(rounded_pixel >= 0) and np.all(
        pixel >= 0) and in_fov_rounded and in_fov

      new_pixels.append(pixel)
      new_rounded_pixels.append(rounded_pixel)
    if is_valid:
      break

  new_pixels = new_pixels[:-4]
  new_rounded_pixels = new_rounded_pixels[:-4]

  # Apply rigid transform to image and pixel labels.
  current_image = cv2.warpAffine(
    current_image,
    transform[:2, :], (image_size[1], image_size[0]),
    flags=cv2.INTER_NEAREST)
  if next_image is not None:
    next_image = cv2.warpAffine(
      next_image,
      transform[:2, :], (image_size[1], image_size[0]),
      flags=cv2.INTER_NEAREST)

  return current_image, next_image, new_pixels, new_rounded_pixels, transform_params

def perturbWithTheta(current_image, next_image, pixels, theta):
  """Data augmentation on images."""
  image_size = current_image.shape[:2]
  trans = (0, 0)
  pivot = (image_size[0]//2, image_size[1]//2)
  transform = get_image_transform(theta, trans, pivot)
  transform_params = theta, trans, pivot

  # Ensure pixels remain in the image after transform.
  new_pixels = []
  new_rounded_pixels = []
  for pixel in pixels:
    pixel = np.float32([pixel[1], pixel[0], 1.]).reshape(3, 1)

    rounded_pixel = np.int32(np.round(transform @ pixel))[:2].squeeze()
    rounded_pixel = np.flip(rounded_pixel)

    pixel = (transform @ pixel)[:2].squeeze()
    pixel = np.flip(pixel)

    new_pixels.append(pixel)
    new_rounded_pixels.append(rounded_pixel)

  # Apply rigid transform to image and pixel labels.
  current_image = cv2.warpAffine(
    current_image,
    transform[:2, :], (image_size[1], image_size[0]),
    flags=cv2.INTER_NEAREST)
  next_image = cv2.warpAffine(
    next_image,
    transform[:2, :], (image_size[1], image_size[0]),
    flags=cv2.INTER_NEAREST)

  return current_image, next_image, new_pixels, new_rounded_pixels, transform_params

def augmentBuffer(buffer, aug_n, rzs):
  num_rz = len(rzs)
  aug_list = []
  for i, d in enumerate(buffer):
    for _ in range(aug_n):
      dtheta = rzs[1] - rzs[0]
      theta_dis_n = int(2 * np.pi / dtheta)
      obs, next_obs, _, (trans_pixel,), transform_params = perturb(d.obs[0].clone().numpy(),
                                                                   d.next_obs[0].clone().numpy(),
                                                                   [d.action[:2].clone().numpy()],
                                                                   theta_dis_n=theta_dis_n)
      action_theta = d.action[2].clone()
      trans_theta, _, _ = transform_params
      if trans_theta >= dtheta:
        action_theta -= (trans_theta // dtheta).long()
        action_theta %= num_rz
      if trans_theta <= -dtheta:
        action_theta += (trans_theta // dtheta).long()
        action_theta %= num_rz
      trans_action = torch.tensor([trans_pixel[0], trans_pixel[1], action_theta])
      aug_list.append(ExpertTransition(d.state, (torch.tensor(obs), d.obs[1]), trans_action, d.reward, d.next_state,
                              (torch.tensor(next_obs), d.next_obs[1]), d.done, d.step_left, d.expert))
  for d in aug_list:
    buffer.add(d)

def augmentBufferD4(buffer, rzs):
  num_rz = len(rzs)
  aug_list = []
  for i, d in enumerate(buffer):
    for j, rot in enumerate(np.linspace(0, 2 * np.pi, 4, endpoint=False)):
      dtheta = rzs[1] - rzs[0]
      obs, next_obs, _, (trans_pixel,), transform_params = perturbWithTheta(d.obs[0].clone().numpy(), d.next_obs[0].clone().numpy(), [d.action[:2].clone().numpy()], theta=rot)
      action_theta = d.action[2].clone()
      trans_theta, _, _ = transform_params
      if trans_theta >= dtheta:
        action_theta -= (trans_theta // dtheta).long()
        action_theta %= num_rz
      if trans_theta <= -dtheta:
        action_theta += (trans_theta // dtheta).long()
        action_theta %= num_rz
      trans_action = torch.tensor([trans_pixel[0], trans_pixel[1], action_theta])
      aug_list.append(ExpertTransition(d.state, (torch.tensor(obs), d.obs[1]), trans_action, d.reward, d.next_state,
                                       (torch.tensor(next_obs), d.next_obs[1]), d.done, d.step_left, d.expert))

      flipped_obs = np.flip(obs, 0)
      flipped_next_obs = np.flip(next_obs, 0)
      flipped_xy = trans_pixel.copy()
      flipped_xy[0] = flipped_obs.shape[-1] - 1 - flipped_xy[0]
      flipped_theta = action_theta.clone()
      flipped_theta = (-flipped_theta) % num_rz
      flipped_action = torch.tensor([flipped_xy[0], flipped_xy[1], flipped_theta])
      aug_list.append(ExpertTransition(d.state, (torch.tensor(flipped_obs.copy()), d.obs[1]), flipped_action, d.reward, d.next_state,
                                       (torch.tensor(flipped_next_obs.copy()), d.next_obs[1]), d.done, d.step_left, d.expert))
  for d in aug_list:
    buffer.add(d)

def addPerlinToBuffer(buffer, perlin_c, heightmap_size, in_hand_mode, no_bar):
    patch_size = buffer[0].obs[1].shape[-1]
    if perlin_c == 0:
        return
    if not no_bar:
      loop = tqdm(range(len(buffer)))
    else:
      loop = range(len(buffer))

    for i in loop:
        t = buffer[i]
        obs_w_perlin = t.obs[0] + (
                perlin_c * rand_perlin_2d((heightmap_size, heightmap_size), (getPerlinFade(heightmap_size), getPerlinFade(heightmap_size))))
        in_hand_w_perlin = t.obs[1] + (
                perlin_c * rand_perlin_2d((patch_size, patch_size), (getPerlinFade(patch_size), getPerlinFade(patch_size))))
        n_obs_w_perlin = t.next_obs[0] + (
                perlin_c * rand_perlin_2d((heightmap_size, heightmap_size), (getPerlinFade(heightmap_size), getPerlinFade(heightmap_size))))
        n_in_hand_w_perlin = t.next_obs[1] + (
                perlin_c * rand_perlin_2d((patch_size, patch_size), (getPerlinFade(patch_size), getPerlinFade(patch_size))))

        if in_hand_mode == 'proj':
            noisy_obs = (obs_w_perlin, t.obs[1])
            noisy_next_obs = (n_obs_w_perlin, t.next_obs[1])
        else:
            noisy_obs = (obs_w_perlin, in_hand_w_perlin)
            noisy_next_obs = (n_obs_w_perlin, n_in_hand_w_perlin)
        t = ExpertTransition(t.state, noisy_obs, t.action, t.reward, t.next_state, noisy_next_obs, t.done,
                             t.step_left, t.expert)
        buffer[i] = t

def addGaussianToBuffer(buffer, gaussian_c, in_hand_mode, no_bar):
  if gaussian_c == 0:
    return
  if not no_bar:
    loop = tqdm(range(len(buffer)))
  else:
    loop = range(len(buffer))

  for i in loop:
    t = buffer[i]
    obs_w_gaussian = t.obs[0] + (torch.randn_like(t.obs[0])) * gaussian_c
    in_hand_w_gaussian = t.obs[1] + (torch.randn_like(t.obs[1])) * gaussian_c
    n_obs_w_gaussian = t.next_obs[0] + (torch.randn_like(t.next_obs[0])) * gaussian_c
    n_in_hand_w_gaussian = t.next_obs[1] + (torch.randn_like(t.next_obs[1])) * gaussian_c

    if in_hand_mode == 'proj':
      noisy_obs = (obs_w_gaussian, t.obs[1])
      noisy_next_obs = (n_obs_w_gaussian, t.next_obs[1])
    else:
      noisy_obs = (obs_w_gaussian, in_hand_w_gaussian)
      noisy_next_obs = (n_obs_w_gaussian, n_in_hand_w_gaussian)
    t = ExpertTransition(t.state, noisy_obs, t.action, t.reward, t.next_state, noisy_next_obs, t.done,
                         t.step_left, t.expert)
    buffer[i] = t

def getPerlinFade(img_size):
  if img_size == 128:
    return int(np.random.choice([1, 2, 4, 8], 1)[0])
  elif img_size == 90:
    return int(np.random.choice([1, 2, 3, 5, 6], 1)[0])
  elif img_size == 24:
    return int(np.random.choice([1, 2], 1)[0])
  elif img_size == 40:
    return int(np.random.choice([1, 2, 4, 5], 1)[0])

def getDrQAugmentedTransition(obs, action_idx=None, rzs=(0, np.pi/2), aug_type='shift'):
    theta_dis_n = 2 * np.pi // (rzs[1] - rzs[0])
    num_rz = len(rzs)
    heightmap_size = obs.shape[-1]
    if aug_type in ['cn', 't', 'se2']:
        if action_idx is None:
            pixels = [[0, 0]]
        else:
            pixels = [action_idx[:2]]
        if aug_type == 'cn':
            set_trans_zero = True
            set_theta_zero = False
        elif aug_type == 't':
            set_trans_zero = False
            set_theta_zero = True
        elif aug_type == 'se2':
            set_trans_zero = False
            set_theta_zero = False
        else:
            raise NotImplementedError
        aug_obs, _, _, (trans_pixel, ), (trans_theta, _, _) = perturb(obs, None, pixels, set_trans_zero=set_trans_zero,
                                                                      set_theta_zero=set_theta_zero,
                                                                      theta_dis_n=theta_dis_n)
        if action_idx is not None:
            action_theta = action_idx[2]
            if trans_theta >= rzs[1] - rzs[0]:
                action_theta -= (trans_theta // rzs[1] - rzs[0]).long()
                action_theta %= num_rz
            if trans_theta <= -rzs[1] - rzs[0]:
                action_theta += (trans_theta // rzs[1] - rzs[0]).long()
                action_theta %= num_rz
            trans_action = [trans_pixel[0], trans_pixel[1], action_theta]
        else:
            trans_action = None
        return aug_obs, trans_action
    elif aug_type == 'shift':
        while True:
            padded_obs = np.pad(obs, [4, 4], mode='edge')
            mag_x = np.random.randint(8)
            mag_y = np.random.randint(8)
            aug_obs = padded_obs[mag_x:mag_x+heightmap_size, mag_y:mag_y+heightmap_size]
            if action_idx is None:
                trans_action = None
                break
            else:
                trans_action = [action_idx[0]-mag_x+4, action_idx[1]-mag_y+4, action_idx[2]]
                if (np.array(trans_action[:2]) > 0).all() and (np.array(trans_action[:2]) < heightmap_size).all():
                    break
        return aug_obs, trans_action