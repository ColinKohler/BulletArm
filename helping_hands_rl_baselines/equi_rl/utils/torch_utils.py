import torch
import torch.nn as nn
import math
import torch.nn.functional as F
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as TF
from torch.autograd import Variable
import numpy as np
import cv2
import collections
from helping_hands_rl_baselines.equi_rl.utils.parameters import crop_size

from collections import OrderedDict
from scipy.ndimage import affine_transform

ExpertTransition = collections.namedtuple('ExpertTransition', 'state obs action reward next_state next_obs done step_left expert')

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
    return torch.cat(((m // d).view(-1, 1), (m % d).view(-1, 1)), dim=1)

def argmax3d(tensor):
    n = tensor.size(0)
    c = tensor.size(1)
    d = tensor.size(2)
    m = tensor.contiguous().view(n, -1).argmax(1)
    return torch.cat(((m//(d*d)).view(-1, 1), ((m%(d*d))//d).view(-1, 1), ((m%(d*d))%d).view(-1, 1)), dim=1)

def argmax4d(tensor):
    n = tensor.size(0)
    c1 = tensor.size(1)
    c2 = tensor.size(2)
    c3 = tensor.size(3)
    c4 = tensor.size(4)
    m = tensor.reshape(n, -1).argmax(1)

    d0 = (m//(c4*c3*c2)).reshape(-1, 1)
    d1 = ((m%(c4*c3*c2))//(c4*c3)).reshape(-1, 1)
    d2 = (((m%(c4*c3*c2))%(c4*c3))//c4).reshape(-1, 1)
    d3 = (((m%(c4*c3*c2))%(c4*c3))%c4).reshape(-1, 1)

    return torch.cat((d0, d1, d2, d3), dim=1)

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

def randomCrop(imgs, out=64):
    n, c, h, w = imgs.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = torch.empty((n, c, out, out), dtype=imgs.dtype).to(imgs.device)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    return cropped

def centerCrop(imgs, out=64):
    n, c, h, w = imgs.shape
    top = (h - out) // 2
    left = (w - out) // 2

    imgs = imgs[:, :, top:top + out, left:left + out]
    return imgs

def bbox(img, threshold=0.011):
    rows = np.any(img>threshold, axis=1)
    cols = np.any(img>threshold, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

# code for this function from: https://github.com/google-research/ravens/blob/d11b3e6d35be0bd9811cfb5c222695ebaf17d28a/ravens/utils/utils.py#L353
#
# coding=utf-8
# Copyright 2021 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

# code for this function from: https://github.com/google-research/ravens/blob/d11b3e6d35be0bd9811cfb5c222695ebaf17d28a/ravens/utils/utils.py#L418
#
# coding=utf-8
# Copyright 2021 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def get_random_image_transform_params(image_size):
    theta = np.random.random() * 2*np.pi
    trans = np.random.randint(0, image_size[0]//10, 2) - image_size[0]//20
    pivot = (image_size[1] / 2, image_size[0] / 2)
    return theta, trans, pivot

# code for this function modified from: https://github.com/google-research/ravens/blob/d11b3e6d35be0bd9811cfb5c222695ebaf17d28a/ravens/utils/utils.py#L428
#
# coding=utf-8
# Copyright 2021 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The original version:
#
# def perturb(input_image, pixels, set_theta_zero=False):
#   """Data augmentation on images."""
#   image_size = input_image.shape[:2]
#
#   # Compute random rigid transform.
#   while True:
#     theta, trans, pivot = get_random_image_transform_params(image_size)
#     if set_theta_zero:
#       theta = 0.
#     transform = get_image_transform(theta, trans, pivot)
#     transform_params = theta, trans, pivot
#
#     # Ensure pixels remain in the image after transform.
#     is_valid = True
#     new_pixels = []
#     new_rounded_pixels = []
#     for pixel in pixels:
#       pixel = np.float32([pixel[1], pixel[0], 1.]).reshape(3, 1)
#
#       rounded_pixel = np.int32(np.round(transform @ pixel))[:2].squeeze()
#       rounded_pixel = np.flip(rounded_pixel)
#
#       pixel = (transform @ pixel)[:2].squeeze()
#       pixel = np.flip(pixel)
#
#       in_fov_rounded = rounded_pixel[0] < image_size[0] and rounded_pixel[
#           1] < image_size[1]
#       in_fov = pixel[0] < image_size[0] and pixel[1] < image_size[1]
#
#       is_valid = is_valid and np.all(rounded_pixel >= 0) and np.all(
#           pixel >= 0) and in_fov_rounded and in_fov
#
#       new_pixels.append(pixel)
#       new_rounded_pixels.append(rounded_pixel)
#     if is_valid:
#       break
#
#   # Apply rigid transform to image and pixel labels.
#   input_image = cv2.warpAffine(
#       input_image,
#       transform[:2, :], (image_size[1], image_size[0]),
#       flags=cv2.INTER_NEAREST)
#   return input_image, new_pixels, new_rounded_pixels, transform_params
def perturb(current_image, next_image, dxy, set_theta_zero=False, set_trans_zero=False):
    image_size = current_image.shape[-2:]

    # Compute random rigid transform.
    theta, trans, pivot = get_random_image_transform_params(image_size)
    if set_theta_zero:
        theta = 0.
    if set_trans_zero:
        trans = [0., 0.]
    transform = get_image_transform(theta, trans, pivot)
    transform_params = theta, trans, pivot

    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_dxy = rot.dot(dxy)
    rotated_dxy = np.clip(rotated_dxy, -1, 1)

    # Apply rigid transform to image and pixel labels.
    current_image = affine_transform(current_image, np.linalg.inv(transform), mode='nearest', order=1)
    if next_image is not None:
        next_image = affine_transform(next_image, np.linalg.inv(transform), mode='nearest', order=1)

    return current_image, next_image, rotated_dxy, transform_params

def perturbVec(current_state, next_state, dxy, set_theta_zero=False, set_trans_zero=False):
    assert not set_theta_zero
    assert set_trans_zero

    aug_current_state = current_state.copy()
    aug_next_state = next_state.copy()

    n_pose = (current_state.shape[0] - 1) // 4

    theta = np.random.random() * 2 * np.pi - np.pi
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    rotated_dxy = rot.dot(dxy)
    rotated_dxy = np.clip(rotated_dxy, -1, 1)

    for i in range(n_pose):
        aug_current_state[1+i*4: 1+i*4+2] = rot.dot(current_state[1+i*4: 1+i*4+2])
        aug_next_state[1+i*4: 1+i*4+2] = rot.dot(next_state[1+i*4: 1+i*4+2])

        scaled_current_theta = current_state[1+i*4+3]
        unscaled_current_theta = (scaled_current_theta+1) * np.pi
        unscaled_aug_current_theta = unscaled_current_theta + theta
        if unscaled_aug_current_theta > np.pi:
            unscaled_aug_current_theta -= 2* np.pi
        if unscaled_aug_current_theta < -np.pi:
            unscaled_aug_current_theta += 2* np.pi
        aug_current_state[1 + i * 4 + 3] = 2 * (unscaled_aug_current_theta - -np.pi) / (2*np.pi) - 1

        scaled_next_theta = next_state[1+i*4+3]
        unscaled_next_theta = (scaled_next_theta+1) * np.pi
        unscaled_aug_next_theta = unscaled_next_theta + theta
        if unscaled_aug_next_theta > np.pi:
            unscaled_aug_next_theta -= 2* np.pi
        if unscaled_aug_next_theta < -np.pi:
            unscaled_aug_next_theta += 2* np.pi
        aug_next_state[1 + i * 4 + 3] = 2 * (unscaled_aug_next_theta - -np.pi) / (2*np.pi) - 1


        # aug_current_state[1+i*4+3] = (current_state[1+i*4+3]+1) * np.pi + theta
        # if aug_current_state[1+i*4+3] > np.pi:
        #     aug_current_state[1 + i * 4 + 3] -= 2* np.pi
        # if aug_current_state[1+i*4+3] < -np.pi:
        #     aug_current_state[1 + i * 4 + 3] += 2* np.pi
        # aug_next_state[1+i*4+3] = next_state[1+i*4+3] + theta

    return aug_current_state, aug_next_state, rotated_dxy, theta

def augmentDQNTransitionC4(d):
    t1_map = np.array([6, 3, 0,
                       7, 4, 1,
                       8, 5, 2])
    t2_map = np.array([8, 7, 6,
                       5, 4, 3,
                       2, 1, 0])
    t3_map = np.array([2, 5, 8,
                       1, 4, 7,
                       0, 3, 6])
    current_image = d.obs[0].copy()
    next_image = d.next_obs[0].copy()
    image_size = current_image.shape[-2:]

    # Compute random rigid transform.
    theta_id = np.random.randint(0, 4)
    theta = theta_id * np.pi/2
    trans = [0., 0.]
    pivot = (image_size[1] / 2, image_size[0] / 2)
    transform = get_image_transform(theta, trans, pivot)

    # Apply rigid transform to image and pixel labels.
    current_image = affine_transform(current_image, np.linalg.inv(transform), mode='nearest', order=1)
    if next_image is not None:
        next_image = affine_transform(next_image, np.linalg.inv(transform), mode='nearest', order=1)

    action = d.action.copy()
    if theta_id == 1:
        action[1] = t1_map[action[1]]
    elif theta_id == 2:
        action[1] = t2_map[action[1]]
    elif theta_id == 3:
        action[1] = t3_map[action[1]]
    obs = current_image.reshape(1, *current_image.shape)
    next_obs = next_image.reshape(1, *next_image.shape)
    return ExpertTransition(d.state, obs, action, d.reward, d.next_state,
                            next_obs, d.done, d.step_left, d.expert)

def augmentTransitionSO2(d):
    obs, next_obs, dxy, transform_params = perturb(d.obs[0].copy(),
                                                   d.next_obs[0].copy(),
                                                   d.action[1:3].copy(),
                                                   set_trans_zero=True)
    obs = obs.reshape(1, *obs.shape)
    next_obs = next_obs.reshape(1, *next_obs.shape)
    action = d.action.copy()
    action[1] = dxy[0]
    action[2] = dxy[1]
    return ExpertTransition(d.state, obs, action, d.reward, d.next_state,
                            next_obs, d.done, d.step_left, d.expert)


def augmentTransitionSE2(d):
    obs, next_obs, dxy, transform_params = perturb(d.obs[0].copy(),
                                                   d.next_obs[0].copy(),
                                                   d.action[1:3].copy())
    obs = obs.reshape(1, *obs.shape)
    next_obs = next_obs.reshape(1, *next_obs.shape)
    action = d.action.clone()
    action[1] = dxy[0]
    action[2] = dxy[1]
    return ExpertTransition(d.state, obs, action, d.reward, d.next_state,
                            next_obs, d.done, d.step_left, d.expert)


def augmentTransitionTranslate(d):
    obs, next_obs, dxy, transform_params = perturb(d.obs[0].copy(),
                                                   d.next_obs[0].copy(),
                                                   d.action[1:3].copy(),
                                                   set_theta_zero=True)
    obs = obs.reshape(1, *obs.shape)
    next_obs = next_obs.reshape(1, *next_obs.shape)
    return ExpertTransition(d.state, obs, d.action, d.reward, d.next_state,
                            next_obs, d.done, d.step_left, d.expert)

def augmentTransitionSO2Vec(d):
    obs, next_obs, dxy, transform_params = perturbVec(d.obs.copy(),
                                                      d.next_obs.copy(),
                                                      d.action[1:3].copy(),
                                                      set_trans_zero=True)
    action = d.action.copy()
    action[1] = dxy[0]
    action[2] = dxy[1]
    return ExpertTransition(d.state, obs, action, d.reward, d.next_state,
                            next_obs, d.done, d.step_left, d.expert)

def augmentTransitionShift(d):
    obs = d.obs[0]
    next_obs = d.next_obs[0]
    heightmap_size = obs.shape[-1]
    padded_obs = np.pad(obs, [4, 4], mode='edge')
    padded_next_obs = np.pad(next_obs, [4, 4], mode='edge')
    mag_x = np.random.randint(8)
    mag_y = np.random.randint(8)
    obs = padded_obs[mag_x:mag_x + heightmap_size, mag_y:mag_y + heightmap_size]
    next_obs = padded_next_obs[mag_x:mag_x + heightmap_size, mag_y:mag_y + heightmap_size]
    obs = obs.reshape(1, *obs.shape)
    next_obs = next_obs.reshape(1, *next_obs.shape)
    return ExpertTransition(d.state, obs, d.action, d.reward, d.next_state,
                            next_obs, d.done, d.step_left, d.expert)

def augmentTransitionCrop(d):
    obs = d.obs[0]
    next_obs = d.next_obs[0]
    heightmap_size = obs.shape[-1]

    crop_max = heightmap_size - crop_size + 1
    w1 = np.random.randint(0, crop_max)
    h1 = np.random.randint(0, crop_max)
    obs = obs[w1:w1 + crop_size, h1:h1 + crop_size]
    next_obs = next_obs[w1:w1 + crop_size, h1:h1 + crop_size]
    obs = obs.reshape(1, *obs.shape)
    next_obs = next_obs.reshape(1, *next_obs.shape)
    return ExpertTransition(d.state, obs, d.action, d.reward, d.next_state,
                            next_obs, d.done, d.step_left, d.expert)


def augmentTransition(d, aug_type):
    if aug_type == 'se2':
        return augmentTransitionSE2(d)
    elif aug_type == 'so2':
        return augmentTransitionSO2(d)
    elif aug_type == 't':
        return augmentTransitionTranslate(d)
    elif aug_type == 'dqn_c4':
        return augmentDQNTransitionC4(d)
    elif aug_type == 'so2_vec':
        return augmentTransitionSO2Vec(d)
    elif aug_type == 'shift':
        return augmentTransitionShift(d)
    elif aug_type == 'crop':
        return augmentTransitionCrop(d)
    else:
        raise NotImplementedError

def normalizeTransition(d: ExpertTransition):
    obs = np.clip(d.obs, 0, 0.32)
    obs = obs/0.4*255
    obs = obs.astype(np.uint8)

    next_obs = np.clip(d.next_obs, 0, 0.32)
    next_obs = next_obs/0.4*255
    next_obs = next_obs.astype(np.uint8)

    return ExpertTransition(d.state, obs, d.action, d.reward, d.next_state, next_obs, d.done, d.step_left, d.expert)

def augmentBuffer(buffer, aug_t, aug_n):
    aug_list = []
    for i, d in enumerate(buffer):
        for _ in range(aug_n):
            aug_list.append(augmentTransition(d, aug_t))
    for d in aug_list:
        buffer.add(d)