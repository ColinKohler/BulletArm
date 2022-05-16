import numpy as np
import cupy as cp
import torch
import time
from functools import wraps
from bulletarm_baselines.fc_dqn.agents.agents_6d.dqn_6d_asr_5l import DQN6DASR5L
from cupyx.scipy.ndimage import median_filter

def timer(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            end_ = time.time() - start
            print('{}: {}'.format(func.__name__, end_))
    return _time_it


class DQN6DASR5LDeictic(DQN6DASR5L):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7 * np.pi / 8), num_ry=8, ry_range=(0, 7 * np.pi / 8), num_rx=8,
                 rx_range=(0, 7 * np.pi / 8), num_zs=16, z_range=(0.02, 0.12)):
        super().__init__(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range,
                         num_ry, ry_range, num_rx, rx_range, num_zs, z_range)

    def initTMap(self):
        super().initTMap()
        self.map = cp.array(self.map)

    def getProjAll(self, obs, center_pixel, rz, z, ry, rx, batch_dimension='ry'):
        batch_size = obs.shape[0]
        patch = self.getPatch(obs, center_pixel, torch.zeros_like(rz.squeeze(1)))
        patch = np.round(patch.cpu().numpy(), 5)
        patch = cp.array(patch)
        size = self.patch_size
        if batch_dimension in ('ry', 'rx'):
            zs = cp.array(z.numpy()) + cp.array([(-size / 2 + j) * self.heightmap_resolution for j in range(size)])
            zs = zs.reshape((zs.shape[0], 1, 1, zs.shape[1]))
            zs = zs.repeat(size, 1).repeat(size, 2)
            c = patch.reshape(patch.shape[0], self.patch_size, self.patch_size, 1).repeat(size, 3)
            ori_occupancy = c > zs
            # transform into points
            point_w_d = cp.argwhere(ori_occupancy)

            if batch_dimension == 'ry':
                rz_id = (rz.expand(-1, self.num_rz) - self.rzs).abs().argmin(1).unsqueeze(0).expand(self.num_ry, -1)
                ry_id = torch.arange(self.num_ry).unsqueeze(1).expand(-1, batch_size)
                rx_id = (rx.expand(-1, self.num_rx) - self.rxs).abs().argmin(1).unsqueeze(0).expand(self.num_ry, -1)
                extra_d_size = self.num_ry
            elif batch_dimension == 'rx':
                rz_id = (rz.expand(-1, self.num_rz) - self.rzs).abs().argmin(1).unsqueeze(0).expand(self.num_ry, -1)
                ry_id = (ry.expand(-1, self.num_ry) - self.rys).abs().argmin(1).unsqueeze(0).expand(self.num_ry, -1)
                rx_id = torch.arange(self.num_rx).unsqueeze(1).expand(-1, batch_size)
                extra_d_size = self.num_rx
            else:
                raise NotImplementedError

            dimension = point_w_d[:, 0]
            point = point_w_d[:, 1:4]
            dimension = cp.tile(dimension, extra_d_size)
            point = cp.tile(point, (extra_d_size, 1))
            extra_dimension = cp.tile(cp.arange(extra_d_size).reshape(-1, 1), (1, point_w_d.shape[0])).reshape(-1)

        elif batch_dimension == 'z':
            add = cp.array([(-size / 2 + j) * self.heightmap_resolution for j in range(size)])
            add = cp.tile(add.reshape((1, 1, size)), (batch_size, self.num_zs, 1))
            zs = cp.tile(self.zs.reshape(1, -1, 1), (batch_size, 1, size))
            zs = zs + add
            zs = zs.reshape((batch_size, self.num_zs, 1, 1, size))
            zs = cp.tile(zs, (1, 1, size, size, 1))
            c = patch.reshape(batch_size, 1, size, size, 1)
            c = cp.tile(c, (1, self.num_zs, 1, 1, size))
            ori_occupancy = c > zs
            point_w_d = cp.argwhere(ori_occupancy)
            dimension = point_w_d[:, 0]
            extra_dimension = point_w_d[:, 1]
            point = point_w_d[:, -3:]
            rz_id = (rz.expand(-1, self.num_rz) - self.rzs).abs().argmin(1).unsqueeze(0).expand(self.num_zs, -1)
            ry_id = (ry.expand(-1, self.num_ry) - self.rys).abs().argmin(1).unsqueeze(0).expand(self.num_zs, -1)
            rx_id = (rx.expand(-1, self.num_rx) - self.rxs).abs().argmin(1).unsqueeze(0).expand(self.num_zs, -1)
            extra_d_size = self.num_zs
        else:
            raise NotImplementedError
        rz_id = cp.array(rz_id)
        ry_id = cp.array(ry_id)
        rx_id = cp.array(rx_id)
        mapped_point = self.map[rz_id[extra_dimension, dimension], ry_id[extra_dimension, dimension], rx_id[extra_dimension, dimension], point[:, 0], point[:, 1], point[:, 2]].T
        valid_point_mask = (cp.logical_and(0 < mapped_point.T, mapped_point.T < size)).all(1)
        rotated_point = mapped_point.T[valid_point_mask]
        batch_dimension = dimension[valid_point_mask].T.astype(int)
        extra_dimension = extra_dimension[valid_point_mask]
        occupancy = cp.zeros((extra_d_size, patch.shape[0], size, size, size))
        if rotated_point.shape[0] > 0:
            occupancy[extra_dimension, batch_dimension, rotated_point[:, 0], rotated_point[:, 1], rotated_point[:, 2]] = 1
        occupancy = median_filter(occupancy, size=(1, 1, 2, 2, 2))
        occupancy = cp.ceil(occupancy)
        projection = cp.stack((occupancy.sum(2), occupancy.sum(3), occupancy.sum(4)), 2)
        return torch.tensor(projection).float().to(self.device).reshape(projection.shape[0]*projection.shape[1], projection.shape[2], projection.shape[3], projection.shape[4])

    def getQ2Input(self, obs, center_pixel):
        patch = []
        for rz in self.rzs:
            patch.append(self.getPatch(obs, center_pixel, torch.ones(center_pixel.size(0))*rz))
        patch = torch.cat(patch)
        patch = self.normalizePatch(patch)
        return patch

    def getQ3Input(self, obs, center_pixel, rz):
        proj = self.getProjAll(obs, center_pixel, rz, torch.zeros_like(rz), torch.zeros_like(rz), torch.zeros_like(rz), batch_dimension='z')
        return proj

    def getQ4Input(self, obs, center_pixel, rz, z):
        proj = self.getProjAll(obs, center_pixel, rz, z, torch.zeros_like(rz), torch.zeros_like(rz), batch_dimension='ry')
        return proj

    def getQ5Input(self, obs, center_pixel, rz, z, ry):
        proj = self.getProjAll(obs, center_pixel, rz, z, ry, torch.zeros_like(rz), batch_dimension='rx')
        return proj

    def forwardQ2(self, states, in_hand, obs, obs_encoding, pixels, target_net=False, to_cpu=False):
        obs_encoding = obs_encoding.repeat(self.num_rz, 1, 1, 1)
        in_hand = in_hand.repeat(self.num_rz, 1, 1, 1)
        patch = self.getQ2Input(obs.to(self.device), pixels.to(self.device))
        patch = self.encodeInHand(patch, in_hand.to(self.device))

        q2 = self.q2 if not target_net else self.target_q2
        q2_output = q2(obs_encoding, patch).reshape(self.num_rz, states.size(0), self.num_primitives).permute(1, 2, 0)[
            torch.arange(0, states.size(0)),
            states.long()]
        if to_cpu:
            q2_output = q2_output.cpu()
        return q2_output

    def forwardQ3(self, states, in_hand, obs, obs_encoding, pixels, a2_id, target_net=False, to_cpu=False):
        obs_encoding = obs_encoding.repeat(self.num_zs, 1, 1, 1)
        in_hand = in_hand.repeat(self.num_zs, 1, 1, 1)

        a2_id, a2 = self.decodeA2(a2_id)
        patch = self.getQ3Input(obs.to(self.device), pixels.to(self.device), a2)
        patch = self.encodeInHand(patch, in_hand.to(self.device))

        q3 = self.q3 if not target_net else self.target_q3
        q3_output = q3(obs_encoding, patch).reshape(self.num_zs, states.size(0), self.num_primitives).permute(1, 2, 0)[
            torch.arange(0, states.size(0)),
            states.long()]
        if to_cpu:
            q3_output = q3_output.cpu()
        return q3_output

    def forwardQ4(self, states, in_hand, obs, obs_encoding, pixels, a2_id, a3_id, target_net=False, to_cpu=False):
        obs_encoding = obs_encoding.repeat(self.num_ry, 1, 1, 1)
        in_hand = in_hand.repeat(self.num_ry, 1, 1, 1)

        a2_id, a2 = self.decodeA2(a2_id)
        a3_id, a3 = self.decodeA3(a3_id)
        patch = self.getQ4Input(obs.to(self.device), pixels.to(self.device), a2, a3)
        patch = self.encodeInHand(patch, in_hand.to(self.device))

        q4 = self.q4 if not target_net else self.target_q4
        q4_output = q4(obs_encoding, patch).reshape(self.num_ry, states.size(0), self.num_primitives).permute(1, 2, 0)[
            torch.arange(0, states.size(0)),
            states.long()]
        if to_cpu:
            q4_output = q4_output.cpu()
        return q4_output

    def forwardQ5(self, states, in_hand, obs, obs_encoding, pixels, a2_id, a3_id, a4_id, target_net=False, to_cpu=False):
        obs_encoding = obs_encoding.repeat(self.num_rx, 1, 1, 1)
        in_hand = in_hand.repeat(self.num_rx, 1, 1, 1)

        a2_id, a2 = self.decodeA2(a2_id)
        a3_id, a3 = self.decodeA3(a3_id)
        a4_id, a4 = self.decodeA4(a4_id)
        patch = self.getQ5Input(obs.to(self.device), pixels.to(self.device), a2, a3, a4)
        patch = self.encodeInHand(patch, in_hand.to(self.device))

        q5 = self.q5 if not target_net else self.target_q5
        q5_output = q5(obs_encoding, patch).reshape(self.num_rx, states.size(0), self.num_primitives).permute(1, 2, 0)[
            torch.arange(0, states.size(0)),
            states.long()]
        if to_cpu:
            q5_output = q5_output.cpu()
        return q5_output