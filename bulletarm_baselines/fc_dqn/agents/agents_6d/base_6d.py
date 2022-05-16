import numpy as np
import torch
# from scipy.ndimage import median_filter
from cupyx.scipy.ndimage import median_filter
import cupy as cp
from bulletarm_baselines.fc_dqn.agents.base_agent import BaseAgent
from bulletarm_baselines.fc_dqn.utils import transformations


class Base6D(BaseAgent):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7*np.pi/8), num_ry=8, ry_range=(0, 7*np.pi/8), num_rx=8,
                 rx_range=(0, 7*np.pi/8), num_zs=16, z_range=(0.02, 0.12)):
        super().__init__(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size)

        self.num_rz = num_rz
        self.rzs = torch.from_numpy(np.linspace(rz_range[0], rz_range[1], num_rz)).float()

        self.num_zs = num_zs
        self.zs = torch.from_numpy(np.linspace(z_range[0], z_range[1], num_zs)).float()

        self.num_ry = num_ry
        self.rys = torch.from_numpy(np.linspace(ry_range[0], ry_range[1], num_ry)).float()

        self.num_rx = num_rx
        self.rxs = torch.from_numpy(np.linspace(rx_range[0], rx_range[1], num_rx)).float()

        self.map = None
        self.initTMap()

    def initTMap(self):
        maps = []
        for rz in self.rzs:
            for ry in self.rys:
                for rx in self.rxs:
                    occupancy = np.ones((self.patch_size, self.patch_size, self.patch_size))
                    point = np.argwhere(occupancy)
                    point = point - self.patch_size / 2
                    R = transformations.euler_matrix(rx, ry, rz)[:3, :3].T
                    rotated_point = R.dot(point.T)
                    rotated_point = rotated_point + self.patch_size / 2
                    rotated_point = np.round(rotated_point).astype(int)
                    rotated_point = rotated_point.T.reshape(1, self.patch_size, self.patch_size, self.patch_size, 3)
                    maps.append(rotated_point)
        self.map = np.concatenate(maps).reshape((self.num_rz, self.num_ry, self.num_rx, self.patch_size, self.patch_size, self.patch_size, 3))
        self.map = cp.array(self.map)

    def getProj(self, obs, center_pixel, rz, z, ry, rx):
        patch = self.getPatch(obs, center_pixel, torch.zeros(center_pixel.size(0)))
        patch = np.round(patch.cpu().numpy(), 5)
        patch = cp.array(patch)
        projections = []
        size = self.patch_size
        zs = cp.array(z.numpy()) + cp.array([(-size / 2 + j) * self.heightmap_resolution for j in range(size)])
        zs = zs.reshape((zs.shape[0], 1, 1, zs.shape[1]))
        zs = zs.repeat(size, 1).repeat(size, 2)
        c = patch.reshape(patch.shape[0], self.patch_size, self.patch_size, 1).repeat(size, 3)
        ori_occupancy = c > zs
        # transform into points
        point_w_d = cp.argwhere(ori_occupancy)

        rz_id = (rz.expand(-1, self.num_rz) - self.rzs).abs().argmin(1)
        ry_id = (ry.expand(-1, self.num_ry) - self.rys).abs().argmin(1)
        rx_id = (rx.expand(-1, self.num_rx) - self.rxs).abs().argmin(1)

        dimension = point_w_d[:, 0]
        point = point_w_d[:, 1:4]

        rz_id = cp.array(rz_id)
        ry_id = cp.array(ry_id)
        rx_id = cp.array(rx_id)
        mapped_point = self.map[rz_id[dimension], ry_id[dimension], rx_id[dimension], point[:, 0], point[:, 1], point[:, 2]].T
        rotated_point = mapped_point.T[(cp.logical_and(0 < mapped_point.T, mapped_point.T < size)).all(1)]
        d = dimension[(cp.logical_and(0 < mapped_point.T, mapped_point.T < size)).all(1)].T.astype(int)

        for i in range(patch.shape[0]):
            point = rotated_point[d==i].T
            occupancy = cp.zeros((size, size, size))
            if point.shape[0] > 0:
                occupancy[point[0], point[1], point[2]] = 1

            occupancy = median_filter(occupancy, size=2)
            occupancy = cp.ceil(occupancy)

            projection = cp.stack((occupancy.sum(0), occupancy.sum(1), occupancy.sum(2)))
            projections.append(projection)

        return torch.tensor(cp.stack(projections)).float().to(self.device)