import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from bulletarm_baselines.fc_dqn.agents.agents_3d.base_3d import Base3D
from bulletarm_baselines.fc_dqn.utils import torch_utils


class DQN3DFCN(Base3D):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7 * np.pi / 8)):
        super().__init__(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range)

    def initNetwork(self, fcn):
        self.fcn = fcn
        self.target_fcn = deepcopy(fcn)
        self.fcn_optimizer = torch.optim.Adam(self.fcn.parameters(), lr=self.lr, weight_decay=1e-5)
        self.networks.append(self.fcn)
        self.target_networks.append(self.target_fcn)
        self.optimizers.append(self.fcn_optimizer)
        self.updateTarget()

    def getAffineMatrices(self, n, specific_rotations):
        if specific_rotations is None:
            rotations = [self.rzs for _ in range(n)]
        else:
            rotations = specific_rotations
        affine_mats_before = []
        affine_mats_after = []
        for i in range(n):
            for rotate_theta in rotations[i]:
                # counter clockwise
                affine_mat_before = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],
                                                [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                affine_mat_before.shape = (2, 3, 1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float().to(self.device)
                affine_mats_before.append(affine_mat_before)

                affine_mat_after = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
                                               [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                affine_mat_after.shape = (2, 3, 1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float().to(self.device)
                affine_mats_after.append(affine_mat_after)

        affine_mats_before = torch.cat(affine_mats_before)
        affine_mats_after = torch.cat(affine_mats_after)
        return affine_mats_before, affine_mats_after

    def forwardFCN(self, states, in_hand, obs, target_net=False, to_cpu=False, specific_rotation_idxes=None):
        fcn = self.fcn if not target_net else self.target_fcn
        if specific_rotation_idxes is None:
            rotations = [self.rzs for _ in range(obs.size(0))]
        else:
            rotations = self.rzs[specific_rotation_idxes]
        diag_length = float(obs.size(2)) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 32) * 32
        padding_width = int((diag_length - obs.size(2)) / 2)
        obs = obs.to(self.device)
        in_hand = in_hand.to(self.device)
        # pad obs
        obs = F.pad(obs, (padding_width, padding_width, padding_width, padding_width), mode='constant', value=0)
        # expand obs into shape (n*num_rot, c, h, w)
        obs = obs.unsqueeze(1).repeat(1, len(rotations[0]), 1, 1, 1)
        in_hand = in_hand.unsqueeze(1).repeat(1, len(rotations[0]), 1, 1, 1)
        obs = obs.reshape(obs.size(0) * obs.size(1), obs.size(2), obs.size(3), obs.size(4))
        in_hand = in_hand.reshape(in_hand.size(0) * in_hand.size(1), in_hand.size(2), in_hand.size(3), in_hand.size(4))

        affine_mats_before, affine_mats_after = self.getAffineMatrices(states.size(0), specific_rotation_idxes)
        # rotate obs
        flow_grid_before = F.affine_grid(affine_mats_before, obs.size(), align_corners=False)
        rotated_obs = F.grid_sample(obs, flow_grid_before, mode='bilinear', align_corners=False)
        # forward network
        conv_output, _ = fcn(rotated_obs, in_hand)
        # rotate output
        flow_grid_after = F.affine_grid(affine_mats_after, conv_output.size(), align_corners=False)
        unrotate_output = F.grid_sample(conv_output, flow_grid_after, mode='bilinear', align_corners=False)

        rotation_output = unrotate_output.reshape(
            (states.shape[0], -1, unrotate_output.size(1), unrotate_output.size(2), unrotate_output.size(3)))
        rotation_output = rotation_output.permute(0, 2, 1, 3, 4)
        predictions = rotation_output[torch.arange(0, states.size(0)), states.long()]
        predictions = predictions[:, :, padding_width: -padding_width, padding_width: -padding_width]
        if to_cpu:
            predictions = predictions.cpu()
        return predictions

    def getEGreedyActions(self, states, in_hand, obs, eps, coef=0.):
        obs = obs.to(self.device)
        in_hand = in_hand.to(self.device)
        with torch.no_grad():
            q_value_maps = self.forwardFCN(states, in_hand, obs, to_cpu=True)
        q_value_maps += torch.randn_like(q_value_maps) * eps * coef
        action_idx = torch_utils.argmax3d(q_value_maps).long()
        pixels = action_idx[:, 1:]
        rot_idx = action_idx[:, 0:1]

        rand = torch.tensor(np.random.uniform(0, 1, states.size(0)))
        rand_mask = rand < eps
        for i, m in enumerate(rand_mask):
            if m:
                pixel_candidates = torch.nonzero(obs[i, 0]>0.01)
                rand_pixel = pixel_candidates[np.random.randint(pixel_candidates.size(0))]
                pixels[i] = rand_pixel

        rand_phi = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.num_rz)
        rot_idx[rand_mask, 0] = rand_phi.long()

        rot = self.rzs[rot_idx]
        x = (pixels[:, 0].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(states.size(0), 1)
        y = (pixels[:, 1].float() * self.heightmap_resolution + self.workspace[1][0]).reshape(states.size(0), 1)

        actions = torch.cat((x, y, rot), dim=1)
        action_idx = torch.cat((pixels, rot_idx), dim=1)
        return q_value_maps, action_idx, actions

    def calcTDLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        if self.sl:
            q_target = self.gamma ** step_lefts
        else:
            with torch.no_grad():
                q_map_prime = self.forwardFCN(next_states, next_obs[1], next_obs[0], target_net=True)
                q_prime = q_map_prime.reshape((batch_size, -1)).max(1)[0]
                q_target = rewards + self.gamma * q_prime * non_final_masks

        self.loss_calc_dict['q_target'] = q_target

        q_output = self.forwardFCN(states, obs[1], obs[0])
        q_pred = q_output[torch.arange(0, batch_size), action_idx[:, 2], action_idx[:, 0], action_idx[:, 1]]

        self.loss_calc_dict['q1_output'] = q_output

        td_loss = F.smooth_l1_loss(q_pred, q_target)
        with torch.no_grad():
            td_error = torch.abs(q_pred - q_target)

        return td_loss, td_error

    def update(self, batch):
        self._loadBatchToDevice(batch)
        td_loss, td_error = self.calcTDLoss()

        self.fcn_optimizer.zero_grad()
        td_loss.backward()

        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        self.loss_calc_dict = {}

        return td_loss.item(), td_error
