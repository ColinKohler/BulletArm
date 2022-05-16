import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from bulletarm_baselines.fc_dqn.agents.agents_6d.base_6d import Base6D
from bulletarm_baselines.fc_dqn.utils import torch_utils


class DQN6DASR5L(Base6D):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7*np.pi/8), num_ry=8, ry_range=(0, 7*np.pi/8), num_rx=8,
                 rx_range=(0, 7*np.pi/8), num_zs=16, z_range=(0.02, 0.12)):
        super().__init__(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz,
                         rz_range, num_ry, ry_range, num_rx, rx_range, num_zs, z_range)
        self.a2_size = num_rz
        self.a3_size = num_zs
        self.a4_size = num_ry
        self.a5_size = num_rx

        self.q2 = None
        self.q3 = None
        self.q4 = None
        self.q5 = None
        self.target_q2 = None
        self.target_q3 = None
        self.target_q4 = None
        self.target_q5 = None
        self.q2_optimizer = None
        self.q3_optimizer = None
        self.q4_optimizer = None
        self.q5_optimizer = None

    def initNetwork(self, q1, q2, q3, q4, q5):
        self.fcn = q1
        self.q2 = q2
        self.q3 = q3
        self.q4 = q4
        self.q5 = q5
        self.target_fcn = deepcopy(q1)
        self.target_q2 = deepcopy(q2)
        self.target_q3 = deepcopy(q3)
        self.target_q4 = deepcopy(q4)
        self.target_q5 = deepcopy(q5)
        self.fcn_optimizer = torch.optim.Adam(self.fcn.parameters(), lr=self.lr, weight_decay=1e-5)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=self.lr, weight_decay=1e-5)
        self.q3_optimizer = torch.optim.Adam(self.q3.parameters(), lr=self.lr, weight_decay=1e-5)
        self.q4_optimizer = torch.optim.Adam(self.q4.parameters(), lr=self.lr, weight_decay=1e-5)
        self.q5_optimizer = torch.optim.Adam(self.q5.parameters(), lr=self.lr, weight_decay=1e-5)
        self.networks.extend([self.fcn, self.q2, self.q3, self.q4, self.q5])
        self.target_networks.extend([self.target_fcn, self.target_q2, self.target_q3, self.target_q4, self.target_q5])
        self.optimizers.extend([self.fcn_optimizer, self.q2_optimizer, self.q3_optimizer, self.q4_optimizer, self.q5_optimizer])
        self.updateTarget()

    def getQ2Input(self, obs, center_pixel):
        patch = self.getPatch(obs, center_pixel, torch.zeros(center_pixel.size(0)))
        patch = self.normalizePatch(patch)
        return patch

    def getQ3Input(self, obs, center_pixel, rz):
        patch = self.getPatch(obs, center_pixel, rz)
        patch = self.normalizePatch(patch)
        return patch

    def getQ4Input(self, obs, center_pixel, rz, z):
        return self.getProj(obs, center_pixel, rz, z, torch.zeros_like(z), torch.zeros_like(z))

    def getQ5Input(self, obs, center_pixel, rz, z, ry):
        return self.getProj(obs, center_pixel, rz, z, ry, torch.zeros_like(z))

    def forwardQ2(self, states, in_hand, obs, obs_encoding, pixels, target_net=False, to_cpu=False):
        patch = self.getQ2Input(obs.to(self.device), pixels.to(self.device))
        patch = self.encodeInHand(patch, in_hand.to(self.device))

        q2 = self.q2 if not target_net else self.target_q2
        q2_output = q2(obs_encoding, patch).reshape(states.size(0), self.num_primitives, -1)[
            torch.arange(0, states.size(0)),
            states.long()]
        if to_cpu:
            q2_output = q2_output.cpu()
        return q2_output

    def forwardQ3(self, states, in_hand, obs, obs_encoding, pixels, a2_id, target_net=False, to_cpu=False):
        a2_id, a2 = self.decodeA2(a2_id)
        patch = self.getQ3Input(obs.to(self.device), pixels.to(self.device), a2.squeeze(1))
        patch = self.encodeInHand(patch, in_hand.to(self.device))

        q3 = self.q3 if not target_net else self.target_q3
        q3_output = q3(obs_encoding, patch).reshape(states.size(0), self.num_primitives, -1)[
            torch.arange(0, states.size(0)),
            states.long()]
        if to_cpu:
            q3_output = q3_output.cpu()
        return q3_output

    def forwardQ4(self, states, in_hand, obs, obs_encoding, pixels, a2_id, a3_id, target_net=False, to_cpu=False):
        a2_id, a2 = self.decodeA2(a2_id)
        a3_id, a3 = self.decodeA3(a3_id)
        patch = self.getQ4Input(obs.to(self.device), pixels.to(self.device), a2, a3)
        patch = self.encodeInHand(patch, in_hand.to(self.device))

        q4 = self.q4 if not target_net else self.target_q4
        q4_output = q4(obs_encoding, patch).reshape(states.size(0), self.num_primitives, -1)[
            torch.arange(0, states.size(0)),
            states.long()]
        if to_cpu:
            q4_output = q4_output.cpu()
        return q4_output

    def forwardQ5(self, states, in_hand, obs, obs_encoding, pixels, a2_id, a3_id, a4_id, target_net=False, to_cpu=False):
        a2_id, a2 = self.decodeA2(a2_id)
        a3_id, a3 = self.decodeA3(a3_id)
        a4_id, a4 = self.decodeA4(a4_id)
        patch = self.getQ5Input(obs.to(self.device), pixels.to(self.device), a2, a3, a4)
        patch = self.encodeInHand(patch, in_hand.to(self.device))

        q5 = self.q5 if not target_net else self.target_q5
        q5_output = q5(obs_encoding, patch).reshape(states.size(0), self.num_primitives, -1)[
            torch.arange(0, states.size(0)),
            states.long()]
        if to_cpu:
            q5_output = q5_output.cpu()
        return q5_output

    def decodeA2(self, a2_id):
        rz_id = a2_id.reshape(a2_id.size(0), 1)
        rz = self.rzs[rz_id].reshape(a2_id.size(0), 1)
        return rz_id, rz

    def decodeA3(self, a3_id):
        z_id = a3_id.reshape(a3_id.size(0), 1)
        z = self.zs[z_id].reshape(a3_id.size(0), 1)
        return z_id, z

    def decodeA4(self, a4_id):
        ry_id = a4_id.reshape(a4_id.size(0), 1)
        ry = self.rys[ry_id].reshape(a4_id.size(0), 1)
        return ry_id, ry

    def decodeA5(self, a5_id):
        rx_id = a5_id.reshape(a5_id.size(0), 1)
        rx = self.rxs[rx_id].reshape(a5_id.size(0), 1)
        return rx_id, rx

    def decodeActions(self, pixels, a2_id, a3_id, a4_id, a5_id):
        rz_id, rz = self.decodeA2(a2_id)
        z_id, z = self.decodeA3(a3_id)
        ry_id, ry = self.decodeA4(a4_id)
        rx_id, rx = self.decodeA5(a5_id)

        x = (pixels[:, 0].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(pixels.size(0), 1)
        y = (pixels[:, 1].float() * self.heightmap_resolution + self.workspace[1][0]).reshape(pixels.size(0), 1)
        actions = torch.cat((x, y, z, rz, ry, rx), dim=1)
        action_idx = torch.cat((pixels, z_id, rz_id, ry_id, rx_id), dim=1)
        return action_idx, actions

    def getEGreedyActions(self, states, in_hand, obs, eps, coef=0.):
        with torch.no_grad():
            q_value_maps, obs_encoding = self.forwardFCN(states, in_hand, obs, to_cpu=True)
            pixels = torch_utils.argmax2d(q_value_maps).long()
            q2_output = self.forwardQ2(states, in_hand, obs, obs_encoding, pixels, to_cpu=True)
            a2_id = torch.argmax(q2_output, 1)
            q3_output = self.forwardQ3(states, in_hand, obs, obs_encoding, pixels, a2_id, to_cpu=True)
            a3_id = torch.argmax(q3_output, 1)
            q4_output = self.forwardQ4(states, in_hand, obs, obs_encoding, pixels, a2_id, a3_id, to_cpu=True)
            a4_id = torch.argmax(q4_output, 1)
            q5_output = self.forwardQ5(states, in_hand, obs, obs_encoding, pixels, a2_id, a3_id, a4_id, to_cpu=True)
            a5_id = torch.argmax(q5_output, 1)

        rand = torch.tensor(np.random.uniform(0, 1, states.size(0)))
        rand_mask = rand < eps

        if type(obs) is tuple:
            hm, ih = obs
        else:
            hm = obs
        for i, m in enumerate(rand_mask):
            if m:
                pixel_candidates = torch.nonzero(hm[i, 0]>0.01)
                rand_pixel = pixel_candidates[np.random.randint(pixel_candidates.size(0))]
                pixels[i] = rand_pixel

        rand_a2 = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.a2_size)
        a2_id[rand_mask] = rand_a2.long()
        rand_a3 = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.a3_size)
        a3_id[rand_mask] = rand_a3.long()
        rand_a4 = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.a4_size)
        a4_id[rand_mask] = rand_a4.long()
        rand_a5 = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.a5_size)
        a5_id[rand_mask] = rand_a5.long()

        action_idx, actions = self.decodeActions(pixels, a2_id, a3_id, a4_id, a5_id)

        return q_value_maps, action_idx, actions

    def getActionFromPlan(self, plan):
        x = plan[:, 0:1]
        y = plan[:, 1:2]
        z = plan[:, 2:3]
        rz = plan[:, 3:4]
        ry = plan[:, 4:5]
        rx = plan[:, 5:6]
        states = plan[:, 6:7]
        pixel_x = ((x - self.workspace[0][0]) / self.heightmap_resolution).long()
        pixel_y = ((y - self.workspace[1][0]) / self.heightmap_resolution).long()
        pixel_x = torch.clamp(pixel_x, 0, self.heightmap_size - 1)
        pixel_y = torch.clamp(pixel_y, 0, self.heightmap_size - 1)

        z_id = (z.expand(-1, self.num_zs) - self.zs).abs().argmin(1)
        diff = (rz.expand(-1, self.num_rz) - self.rzs).abs()
        diff2 = (diff - np.pi).abs()
        rz_id = torch.min(diff, diff2).argmin(1).unsqueeze(1)
        mask = diff.min(1)[0] > diff2.min(1)[0]
        ry[mask] = -ry[mask]
        rx[mask] = -rx[mask]
        # rz_id = (rz.expand(-1, self.num_rz) - self.rzs).abs().argmin(1)
        ry_id = (ry.expand(-1, self.num_ry) - self.rys).abs().argmin(1)
        rx_id = (rx.expand(-1, self.num_rx) - self.rxs).abs().argmin(1)

        action_idx, actions = self.decodeActions(torch.cat((pixel_x, pixel_y), dim=1), rz_id, z_id, ry_id, rx_id)
        return action_idx, actions

    def calcTDLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        pixel = action_idx[:, 0:2]
        a3_idx = action_idx[:, 2:3]
        a2_idx = action_idx[:, 3:4]
        a4_idx = action_idx[:, 4:5]
        a5_idx = action_idx[:, 5:6]

        with torch.no_grad():
            q1_map_prime, obs_prime_encoding = self.forwardFCN(next_states, next_obs[1], next_obs[0], target_net=True)
            x_star = torch_utils.argmax2d(q1_map_prime)
            q2_prime = self.forwardQ2(next_states, next_obs[1], next_obs[0], obs_prime_encoding, x_star, target_net=True)
            a2_star = torch.argmax(q2_prime, 1)
            q3_prime = self.forwardQ3(next_states, next_obs[1], next_obs[0], obs_prime_encoding, x_star, a2_star, target_net=True)
            a3_star = torch.argmax(q3_prime, 1)
            q4_prime = self.forwardQ4(next_states, next_obs[1], next_obs[0], obs_prime_encoding, x_star, a2_star, a3_star, target_net=True)
            a4_star = torch.argmax(q4_prime, 1)
            q5_prime = self.forwardQ5(next_states, next_obs[1], next_obs[0], obs_prime_encoding, x_star, a2_star, a3_star, a4_star, target_net=True)

            q5 = q5_prime.max(1)[0]
            q_prime = q5
            q_target = rewards + self.gamma * q_prime * non_final_masks

        self.loss_calc_dict['q_target'] = q_target

        q1_output, obs_encoding = self.forwardFCN(states, obs[1], obs[0])
        q1_pred = q1_output[torch.arange(0, batch_size), pixel[:, 0], pixel[:, 1]]
        q2_output = self.forwardQ2(states, obs[1], obs[0], obs_encoding, pixel)
        q2_pred = q2_output[torch.arange(batch_size), a2_idx[:, 0]]
        q3_output = self.forwardQ3(states, obs[1], obs[0], obs_encoding, pixel, a2_idx)
        q3_pred = q3_output[torch.arange(batch_size), a3_idx[:, 0]]
        q4_output = self.forwardQ4(states, obs[1], obs[0], obs_encoding, pixel, a2_idx, a3_idx)
        q4_pred = q4_output[torch.arange(batch_size), a4_idx[:, 0]]
        q5_output = self.forwardQ5(states, obs[1], obs[0], obs_encoding, pixel, a2_idx, a3_idx, a4_idx)
        q5_pred = q5_output[torch.arange(batch_size), a5_idx[:, 0]]

        self.loss_calc_dict['q1_output'] = q1_output
        self.loss_calc_dict['q2_output'] = q2_output
        self.loss_calc_dict['q3_output'] = q3_output
        self.loss_calc_dict['q4_output'] = q4_output
        self.loss_calc_dict['q5_output'] = q5_output

        q1_td_loss = F.smooth_l1_loss(q1_pred, q_target)
        q2_td_loss = F.smooth_l1_loss(q2_pred, q_target)
        q3_td_loss = F.smooth_l1_loss(q3_pred, q_target)
        q4_td_loss = F.smooth_l1_loss(q4_pred, q_target)
        q5_td_loss = F.smooth_l1_loss(q5_pred, q_target)
        td_loss = q1_td_loss + q2_td_loss + q3_td_loss + q4_td_loss + q5_td_loss

        with torch.no_grad():
            td_error = torch.abs(q5_pred - q_target)

        return td_loss, td_error

    def update(self, batch):
        self._loadBatchToDevice(batch)
        td_loss, td_error = self.calcTDLoss()

        self.fcn_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        self.q3_optimizer.zero_grad()
        self.q4_optimizer.zero_grad()
        self.q5_optimizer.zero_grad()
        td_loss.backward()

        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        for param in self.q2.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q2_optimizer.step()

        for param in self.q3.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q3_optimizer.step()

        for param in self.q4.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q4_optimizer.step()

        for param in self.q5.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q5_optimizer.step()

        self.loss_calc_dict = {}

        return td_loss.item(), td_error
