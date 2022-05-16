import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from bulletarm_baselines.fc_dqn.agents.agents_3d.base_3d import Base3D
from bulletarm_baselines.fc_dqn.utils import torch_utils


class DQN3DASR(Base3D):
    def __init__(self, workspace, heightmap_size, device, lr=1e-4, gamma=0.9, sl=False, num_primitives=1,
                 patch_size=24, num_rz=8, rz_range=(0, 7*np.pi/8)):
        super().__init__(workspace, heightmap_size, device, lr, gamma, sl, num_primitives, patch_size, num_rz, rz_range)
        self.num_rz = num_rz
        self.rzs = torch.from_numpy(np.linspace(rz_range[0], rz_range[1], num_rz)).float()
        self.a2_size = num_rz

        self.q2 = None
        self.target_q2 = None
        self.q2_optimizer = None

    def initNetwork(self, q1, q2):
        self.fcn = q1
        self.q2 = q2
        self.target_fcn = deepcopy(q1)
        self.target_q2 = deepcopy(q2)
        self.fcn_optimizer = torch.optim.Adam(self.fcn.parameters(), lr=self.lr, weight_decay=1e-5)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=self.lr, weight_decay=1e-5)
        self.networks.append(self.fcn)
        self.networks.append(self.q2)
        self.target_networks.append(self.target_fcn)
        self.target_networks.append(self.target_q2)
        self.optimizers.append(self.fcn_optimizer)
        self.optimizers.append(self.q2_optimizer)
        self.updateTarget()

    def getQ2Input(self, obs, center_pixel):
        patch = self.getPatch(obs, center_pixel, torch.zeros(center_pixel.size(0)))
        return patch

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

    def decodeA2(self, a2_id):
        rz_id = a2_id.reshape(a2_id.size(0), 1)
        rz = self.rzs[rz_id].reshape(a2_id.size(0), 1)
        return rz_id, rz

    def decodeActions(self, pixels, a2_id):
        rz_id, rz = self.decodeA2(a2_id)
        x = (pixels[:, 0].float() * self.heightmap_resolution + self.workspace[0][0]).reshape(pixels.size(0), 1)
        y = (pixels[:, 1].float() * self.heightmap_resolution + self.workspace[1][0]).reshape(pixels.size(0), 1)
        actions = torch.cat((x, y, rz), dim=1)
        action_idx = torch.cat((pixels, rz_id), dim=1)
        return action_idx, actions

    def getEGreedyActions(self, states, in_hand, obs, eps, coef=0.):
        with torch.no_grad():
            q_value_maps, obs_encoding = self.forwardFCN(states, in_hand, obs, to_cpu=True)
            pixels = torch_utils.argmax2d(q_value_maps).long()
            q2_output = self.forwardQ2(states, in_hand, obs, obs_encoding, pixels, to_cpu=True)
            a2_id = torch.argmax(q2_output, 1)

        rand = torch.tensor(np.random.uniform(0, 1, states.size(0)))
        rand_mask = rand < eps

        if type(obs) is tuple:
            hm, ih = obs
        else:
            hm = obs
        for i, m in enumerate(rand_mask):
            if m:
                pixel_candidates = torch.nonzero(hm[i, 0]>-0.01)
                rand_pixel = pixel_candidates[np.random.randint(pixel_candidates.size(0))]
                pixels[i] = rand_pixel

        rand_a2 = torch.randint_like(torch.empty(rand_mask.sum()), 0, self.a2_size)
        a2_id[rand_mask] = rand_a2.long()

        action_idx, actions = self.decodeActions(pixels, a2_id)

        return q_value_maps, action_idx, actions

    def calcTDLoss(self):
        batch_size, states, obs, action_idx, rewards, next_states, next_obs, non_final_masks, step_lefts, is_experts = self._loadLossCalcDict()
        pixel = action_idx[:, 0:2]
        a2_idx = action_idx[:, 2:3]

        if self.sl:
            q_target = self.gamma ** step_lefts
        else:
            with torch.no_grad():
                q1_map_prime, obs_prime_encoding = self.forwardFCN(next_states, next_obs[1], next_obs[0], target_net=True)
                x_star = torch_utils.argmax2d(q1_map_prime)
                q2_prime = self.forwardQ2(next_states, next_obs[1], next_obs[0], obs_prime_encoding, x_star, target_net=True)

                q_prime = q2_prime.max(1)[0]
                q_target = rewards + self.gamma * q_prime * non_final_masks

        self.loss_calc_dict['q_target'] = q_target

        q1_output, obs_encoding = self.forwardFCN(states, obs[1], obs[0])
        q1_pred = q1_output[torch.arange(0, batch_size), pixel[:, 0], pixel[:, 1]]
        q2_output = self.forwardQ2(states, obs[1], obs[0], obs_encoding, pixel)
        q2_pred = q2_output[torch.arange(batch_size), a2_idx[:, 0]]

        self.loss_calc_dict['q1_output'] = q1_output
        self.loss_calc_dict['q2_output'] = q2_output

        q1_td_loss = F.smooth_l1_loss(q1_pred, q_target)
        q2_td_loss = F.smooth_l1_loss(q2_pred, q_target)
        td_loss = q1_td_loss + q2_td_loss

        with torch.no_grad():
            if self.per_td_error == 'last':
                td_error = torch.abs(q2_pred - q_target)
            else:
                td_error = (torch.abs(q1_pred - q_target) + torch.abs(q2_pred - q_target))/2

        return td_loss, td_error

    def update(self, batch):
        self._loadBatchToDevice(batch)
        td_loss, td_error = self.calcTDLoss()

        self.fcn_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        td_loss.backward()

        for param in self.fcn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.fcn_optimizer.step()

        for param in self.q2.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q2_optimizer.step()

        self.loss_calc_dict = {}

        return td_loss.item(), td_error
